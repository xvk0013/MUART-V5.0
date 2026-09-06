"""A2 s42 comparison on original Train: shared evidence plus small interaction.

Each epoch logs single-class/set metrics, NLL and confidence diagnostics.
A fixed unaugmented Train subset is evaluated with the SAME EMA as Val.
Monitoring preserves RNG state and cannot alter training sampling.
Primary selection remains maximum Val EMR (patience 20); best NLL is saved
separately as a diagnostic, never silently substituted for the primary result.
"""
from __future__ import annotations
import argparse
import copy
import csv
import hashlib
import json
import math
import random
from contextlib import contextmanager
from pathlib import Path

# Import order matters for the user's Windows scientific Python runtime.
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from config_v6a2 import ConfigV6A2, validate_v6a2_config
from dataset import create_loaders, collate_multi
from experiment_snapshot import build_experiment_snapshot, write_experiment_snapshot
from model_v6a import LEGAL_SET_NAMES, multihot_to_set_targets, count_parameters
from model_v6a2 import MUARTV6A2
from train_v6a import (
    ModelEMA, WarmupCosineScheduler, seed_everything, compute_metrics,
    query_orthogonal_loss, _move_batch, _prepare_empty_output, _atomic_torch_save,
)


def loss_components(outputs, labels, presence_weight=0.5):
    targets = multihot_to_set_targets(labels)
    nll = F.nll_loss(outputs['set_log_probs'], targets)
    # All rows, all classes: noise negatives, single positives/negatives,
    # double positives/negative. No per-class weighting or teacher target.
    presence = F.binary_cross_entropy_with_logits(
        outputs['presence_logits'].float(), labels.float())
    return nll + presence_weight * presence, nll, presence, targets


@contextmanager
def preserve_rng():
    state = (random.getstate(), np.random.get_state(), torch.get_rng_state(),
             torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None)
    try:
        yield
    finally:
        random.setstate(state[0])
        np.random.set_state(state[1])
        torch.set_rng_state(state[2])
        if state[3] is not None:
            torch.cuda.set_rng_state_all(state[3])


def make_train_monitor(train_dataset, config):
    if train_dataset.split != 'Train':
        raise ValueError('Monitoring subset must come from Train')
    groups = [[] for _ in range(7)]
    for index, (_, label, _) in enumerate(train_dataset.samples):
        target = int(multihot_to_set_targets(torch.tensor([label])).item())
        groups[target].append(index)
    rng = np.random.RandomState(config.monitor_seed)
    selected = []
    for group in groups:
        if not group:
            raise ValueError('Train monitor requires all seven legal sets')
        selected.extend(rng.choice(group, min(len(group), config.monitor_per_set),
                                   replace=False).tolist())
    selected.sort()
    ds = copy.copy(train_dataset)
    ds.samples = [train_dataset.samples[i] for i in selected]
    ds.is_train = False  # disables random cropping as well as augmentation
    ds.waveform_augment = None
    ds.spec_augment = None
    loader = DataLoader(ds, batch_size=config.batch_size, shuffle=False,
                        num_workers=0, collate_fn=collate_multi,
                        generator=torch.Generator().manual_seed(config.monitor_seed))
    return loader


@torch.no_grad()
def evaluate(model, loader, device, config):
    model.eval()
    logs, labels_all, types_all = [], [], []
    sums = dict(nll=0.0, presence_bce=0.0, pair_interaction_abs=0.0)
    conditional = {'single': [0, 0], 'double': [0, 0]}
    for batch in loader:
        x, d, y, types = _move_batch(batch, device, config.use_demon)
        outputs = model(x, d)
        _, nll, bce, target = loss_components(outputs, y, config.presence_loss_weight)
        n = len(y)
        sums['nll'] += float(nll) * n
        sums['presence_bce'] += float(bce) * n
        sums['pair_interaction_abs'] += float(outputs['pair_interaction'].abs().mean()) * n
        counts = y.sum(1).long()
        for name, count, key, offset in (
                ('single', 1, 'single_logits', 1), ('double', 2, 'pair_logits', 4)):
            mask = counts == count
            conditional[name][0] += int((outputs[key][mask].argmax(1) ==
                                         target[mask] - offset).sum())
            conditional[name][1] += int(mask.sum())
        logs.append(outputs['set_log_probs'].cpu())
        labels_all.append(y.cpu())
        types_all.append(types.cpu())
    log_probs = torch.cat(logs)
    labels = torch.cat(labels_all)
    targets = multihot_to_set_targets(labels)
    metrics = compute_metrics(log_probs, labels, torch.cat(types_all), config.class_names)
    metrics.update({key: value / len(labels) for key, value in sums.items()})
    probabilities = log_probs.exp()
    confidence, predicted = probabilities.max(1)
    wrong = predicted != targets
    metrics['wrong_mean_confidence'] = float(confidence[wrong].mean()) if wrong.any() else None
    metrics['set_brier'] = float(((probabilities - F.one_hot(targets, 7))**2).sum(1).mean())
    metrics['conditional_identity_accuracy'] = {
        name: {'n': n, 'correct': hits, 'accuracy': hits/n if n else None}
        for name, (hits, n) in conditional.items()}
    metrics['conditional_identity_note'] = 'Uses true count only for subgroup diagnosis; primary EMR uses predicted joint set.'
    return metrics, probabilities, targets


def train_epoch(model, loader, optimizer, scheduler, ema, scaler, device, config):
    model.train()
    total = dict(n=0, nll=0., bce=0., loss=0., correct=0, skipped_amp_steps=0)
    for step, batch in enumerate(loader, 1):
        x, d, y, _ = _move_batch(batch, device, config.use_demon)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=scaler is not None):
            outputs = model(x, d)
            loss, nll, bce, target = loss_components(outputs, y, config.presence_loss_weight)
            loss = loss + config.ortho_loss_weight * query_orthogonal_loss(
                model.classifier.query_attn.queries)
        if not torch.isfinite(loss):
            raise FloatingPointError('Non-finite training loss')
        updated = True
        if scaler is not None:
            scale = scaler.get_scale()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.clip_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            updated = scaler.get_scale() >= scale
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.clip_grad_norm,
                                           error_if_nonfinite=True)
            optimizer.step()
        if updated:
            scheduler.step()
            if ema is not None:
                ema.update(model)
        else:
            total['skipped_amp_steps'] += 1
        n = len(y)
        total['n'] += n
        for key, value in (('loss', loss), ('nll', nll), ('bce', bce)):
            total[key] += float(value.detach()) * n
        total['correct'] += int((outputs['set_log_probs'].argmax(1) == target).sum())
        if step % 100 == 0:
            print(f"  [{step}/{len(loader)}] loss={total['loss']/total['n']:.4f}", flush=True)
    return dict(loss=total['loss']/total['n'], nll=total['nll']/total['n'],
                presence_bce=total['bce']/total['n'], emr=total['correct']/total['n'],
                skipped_amp_steps=total['skipped_amp_steps'])


def write_csv(path, rows):
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def write_predictions(path, loader, probs, targets, root):
    if len(loader.dataset.samples) != len(targets):
        raise ValueError('Prediction rows do not align with dataset')
    predicted = probs.argmax(1)
    rows = []
    for i, (wav, _, _) in enumerate(loader.dataset.samples):
        row = dict(sample_id=Path(wav).resolve().relative_to(root).as_posix(),
                   true_set=LEGAL_SET_NAMES[int(targets[i])],
                   pred_set=LEGAL_SET_NAMES[int(predicted[i])],
                   correct=int(targets[i] == predicted[i]))
        row.update({f'prob_{name}': float(probs[i, k]) for k, name in enumerate(LEGAL_SET_NAMES)})
        rows.append(row)
    write_csv(path, rows)


def protocol_check(snapshot, reference_dir):
    if reference_dir is None:
        raise ValueError('Formal A2 comparison requires the A1 reference directory')
    reference = json.loads((reference_dir/'experiment_config.json').read_text(encoding='utf-8-sig'))
    baseline = json.loads((reference_dir/'val_metrics.json').read_text(encoding='utf-8-sig'))
    if baseline.get('seed') != 42 or baseline.get('test_loaded') is not False:
        raise ValueError('Reference must be the A1 s42 Train/Val-only run')
    if snapshot['dataset_manifest']['path_size_sha256'] != reference['dataset_manifest']['path_size_sha256']:
        raise ValueError('Dataset path/size manifest differs from A1; do not run a mixed-protocol comparison')
    for key in ('global_mean', 'global_std'):
        if not np.allclose(snapshot['normalization_stats'][key], reference['normalization_stats'][key],
                           rtol=1e-6, atol=1e-6):
            raise ValueError(f'Train normalization differs: {key}')
    keys = ('batch_size', 'num_epochs', 'early_stop_patience', 'lr', 'lr_min',
            'warmup_epochs', 'weight_decay', 'ema_decay', 'clip_grad_norm',
            'normalization_mode', 'use_demon', 'demon_mode', 'feature_type',
            'use_spec_gate', 'dual_stream_fusion', 'freq_shift_mode', 'waveform_noise_mode',
            'use_speed_perturb', 'use_waveform_aug', 'use_specaug', 'ortho_loss_weight',
            'backbone_dims', 'backbone_blocks', 'query_dropout', 'query_attn_dropout',
            'class_names', 'multitarget_pos_to_label', 'multitarget_types', 'sample_rate',
            'audio_len', 'lofar_n_fft', 'lofar_hop', 'lofar_win', 'lofar_freq_max_hz',
            'wideband_n_fft', 'wideband_hop', 'demon_bp_low', 'demon_bp_high',
            'demon_n_bins', 'demon_remove_dc', 'specaug_n_freq_masks', 'specaug_n_time_masks',
            'specaug_freq_param', 'specaug_time_param', 'aug_time_stretch', 'aug_gain_db')
    for key in keys:
        if snapshot['resolved_config'].get(key) != reference['resolved_config'].get(key):
            raise ValueError(f'Comparison setting differs from A1: {key}')
    return baseline


def overfit_signal(history, config):
    best = min(history, key=lambda r: r['val_nll'])
    recent = history[-config.monitor_nll_patience:]
    warning = (len(recent) == config.monitor_nll_patience and
               all(row['val_nll'] > best['val_nll'] + config.monitor_nll_rise for row in recent) and
               recent[-1]['train_eval_nll'] < best['train_eval_nll'])
    return dict(suspected_overfit=bool(warning), best_nll_epoch=best['epoch'],
                val_nll_rise=history[-1]['val_nll']-best['val_nll'],
                note='Heuristic alert, not a significance test; primary best-EMR selection unchanged.')


def run(config, args):
    validate_v6a2_config(config)
    out = Path(config.output_dir).resolve()
    _prepare_empty_output(out)
    device = torch.device(config.device)
    train_loader, val_loader, forbidden_loader, info = create_loaders(config, include_test=False)
    if forbidden_loader is not None or info['n_test'] is not None:
        raise RuntimeError('Test must not be constructed')
    monitor = make_train_monitor(train_loader.dataset, config)
    root = Path(config.data_combined).resolve()
    snapshot = build_experiment_snapshot(config, info, vars(args))
    baseline = protocol_check(snapshot, Path(args.reference_dir).resolve()) if not args.smoke_test else None
    snapshot['a2_protocol'] = dict(
        primary_selection='maximum Val EMR; strict improvement; patience 20',
        auxiliary_selection='minimum Val set NLL, diagnostic only',
        loss='set_NLL + 0.5 * all-sample mean BCE + 0.1 * query_ortho',
        pair='e_i + e_j + 0.5*tanh(interaction32); zero-initialized output',
        monitor='same EMA, no augmentation, fixed stratified Train subset; in-sample diagnostic',
        monitor_ids=[Path(w).resolve().relative_to(root).as_posix() for w, _, _ in monitor.dataset.samples],
        test_loaded=False, training_from_scratch=True)
    code_dir = Path(__file__).resolve().parent
    names = ('config.py','model.py','dataset.py','experiment_snapshot.py', 'config_v6a.py',
             'model_v6a.py','train_v6a.py','config_v6a2.py','model_v6a2.py','train_v6a2.py')
    snapshot['code_sha256'] = {name: hashlib.sha256((code_dir/name).read_bytes()).hexdigest() for name in names}
    write_experiment_snapshot(snapshot, str(out/'experiment_config.json'))
    model = MUARTV6A2(config).to(device)
    print(f'A2 parameters={count_parameters(model):,}; Train={info["n_train"]}; Val={info["n_val"]}; Test NOT LOADED', flush=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    for group in optimizer.param_groups:
        group['initial_lr'] = group['lr']
    scheduler = WarmupCosineScheduler(optimizer, config.warmup_epochs, config.num_epochs,
                                     config.lr_min, len(train_loader))
    ema = ModelEMA(model, config.ema_decay)
    scaler = torch.amp.GradScaler('cuda', enabled=config.use_amp and device.type == 'cuda')
    scaler = scaler if scaler.is_enabled() else None

    if args.smoke_test:
        # Seven real Train examples ensure noise/single/double paths all execute.
        batch = collate_multi([monitor.dataset[next(
            i for i, (_, label, _) in enumerate(monitor.dataset.samples)
            if int(multihot_to_set_targets(torch.tensor([label])).item()) == k)] for k in range(7)])
        tr = train_epoch(model, [batch], optimizer, scheduler, ema, scaler, device, config)
        with preserve_rng():
            val, _, _ = evaluate(ema.ema, [next(iter(val_loader))], device, config)
        # Exact checkpoint round trip on fixed synthetic input, no dataset write.
        _atomic_torch_save({'architecture': config.architecture_name,
                            'model_state_dict': ema.ema.state_dict()}, out/'smoke_checkpoint.pt')
        restored = MUARTV6A2(config).to(device).eval()
        restored.load_state_dict(torch.load(out/'smoke_checkpoint.pt', map_location=device,
                                           weights_only=False)['model_state_dict'])
        x, d, _, _ = _move_batch(batch, device, config.use_demon)
        with torch.no_grad():
            expected = ema.ema(x,d)['set_log_probs']
            actual = restored(x,d)['set_log_probs']
        if not torch.allclose(expected, actual, atol=1e-6, rtol=1e-6):
            raise RuntimeError('Checkpoint round-trip mismatch')
        result = dict(status='PASS', purpose='code smoke only; not research performance',
                      train_step=tr, val_forward_n=val['n'], checkpoint_roundtrip=True, test_loaded=False)
        write_experiment_snapshot(result, str(out/'smoke_result.json'))
        print(f'SMOKE PASS: {out}', flush=True)
        return result

    history, epoch_metrics = [], []
    best_emr, best_nll, patience = -1., math.inf, 0
    best_result = None
    for epoch in range(1, config.num_epochs+1):
        tr = train_epoch(model, train_loader, optimizer, scheduler, ema, scaler, device, config)
        with preserve_rng():
            val, probs, targets = evaluate(ema.ema, val_loader, device, config)
            probe, _, _ = evaluate(ema.ema, monitor, device, config)
        row = dict(epoch=epoch, train_loss=tr['loss'], train_nll=tr['nll'],
                   train_presence_bce=tr['presence_bce'], train_aug_emr=tr['emr'],
                   train_eval_emr=probe['emr'], train_eval_nll=probe['nll'],
                   val_emr=val['emr'], val_nll=val['nll'], val_presence_bce=val['presence_bce'],
                   val_count_accuracy=val['count_accuracy'], val_wrong_confidence=val['wrong_mean_confidence'],
                   val_brier=val['set_brier'], lr=optimizer.param_groups[0]['lr'],
                   skipped_amp_steps=tr['skipped_amp_steps'])
        for name in LEGAL_SET_NAMES:
            row['val_set_'+name] = val['by_legal_set'][name]['emr']
        for kind in ('single','double'):
            row['val_'+kind+'_emr'] = val['by_sample_type'][kind]['emr']
            row['train_eval_'+kind+'_emr'] = probe['by_sample_type'][kind]['emr']
        history.append(row)
        alert = overfit_signal(history, config)
        epoch_metrics.append(dict(epoch=epoch, val=val, train_eval=probe, overfit=alert))
        write_csv(out/'history.csv', history)
        write_experiment_snapshot(epoch_metrics, str(out/'epoch_metrics.json'))
        write_experiment_snapshot(alert, str(out/'overfit_watch.json'))
        print(f'Epoch {epoch}: Val EMR={val["emr"]:.4f} NLL={val["nll"]:.4f}; '
              f'TrainEval EMR={probe["emr"]:.4f} NLL={probe["nll"]:.4f}', flush=True)
        print('  Single sets: '+', '.join(f'{name}={val["by_legal_set"][name]["emr"]:.4f}'
                                         for name in ('Tanker','Cargo','Tug')), flush=True)
        print(f'  single={row["val_single_emr"]:.4f}; double={row["val_double_emr"]:.4f}', flush=True)
        if alert['suspected_overfit']:
            print('  OVERFIT WATCH: TrainEval NLL improves while Val NLL remains above its minimum.', flush=True)
        improved = val['emr'] > best_emr
        nll_improved = val['nll'] < best_nll
        if improved or nll_improved:
            payload = dict(architecture=config.architecture_name, epoch=epoch,
                           legal_set_names=list(LEGAL_SET_NAMES), model_state_dict=ema.ema.state_dict(),
                           ema_state_dict=ema.ema.state_dict(), raw_model_state_dict=model.state_dict(),
                           optimizer_state_dict=optimizer.state_dict(), val_acc=val['emr'],
                           val_metrics=val, config=snapshot['resolved_config'],
                           normalization_stats=snapshot['normalization_stats'],
                           experiment_snapshot=snapshot, test_loaded=False)
            if improved:
                best_emr = val['emr']
                best_result = dict(stage='V6-A2', seed=args.seed, best_epoch=epoch,
                                   best_val_emr=best_emr, reloaded_val_metrics=val,
                                   test_loaded=False, test_evaluated=False,
                                   selection='maximum Val EMR; EMA weights')
                _atomic_torch_save(payload, out/'best_model.pt')
                write_predictions(out/'val_predictions.csv', val_loader, probs, targets, root)
                write_experiment_snapshot(best_result, str(out/'val_metrics.json'))
            if nll_improved:
                best_nll = val['nll']
                _atomic_torch_save(payload, out/'best_nll_model.pt')
                write_experiment_snapshot(dict(epoch=epoch, metrics=val,
                                               purpose='diagnostic only; not primary EMR selection'),
                                          str(out/'best_nll_metrics.json'))
        patience = 0 if improved else patience+1
        if patience >= config.early_stop_patience:
            print(f'Early stop at epoch {epoch}: EMR patience={patience}', flush=True)
            break
    saved = torch.load(out/'best_model.pt', map_location=device, weights_only=False)
    model.load_state_dict(saved['model_state_dict'])
    with preserve_rng():
        reloaded, _, _ = evaluate(model, val_loader, device, config)
    if reloaded['confusion_matrix'] != best_result['reloaded_val_metrics']['confusion_matrix']:
        raise RuntimeError('Best checkpoint evaluation changed after reload')
    best_result['reloaded_val_metrics'] = reloaded
    best_result['checkpoint_reload_verified'] = True
    write_experiment_snapshot(best_result, str(out/'val_metrics.json'))
    comparison = comparison_report(best_result, baseline, history)
    (out/'comparison.md').write_text(comparison, encoding='utf-8')
    print(f'COMPLETE: best Val={best_emr:.6f}. Read comparison.md and overfit_watch.json. Test NOT LOADED.', flush=True)
    return best_result


def comparison_report(result, baseline, history):
    current = result['reloaded_val_metrics']
    previous = baseline['reloaded_val_metrics']
    lines = ['# V6-A2 s42 versus V6-A1 s42', '',
             'Same original Train/Val, one model, no distillation. One-seed development comparison, not a significance claim.', '',
             f'Primary checkpoint: maximum Val EMR, epoch {result["best_epoch"]}.', '',
             '| Metric | A1 | A2 | Change (percentage points) |', '|---|---:|---:|---:|']
    pairs = [('Overall', previous['emr'], current['emr'])]
    pairs += [(name, previous['by_legal_set'][name]['emr'], current['by_legal_set'][name]['emr'])
              for name in LEGAL_SET_NAMES]
    pairs += [(name, previous['by_sample_type'][name]['emr'], current['by_sample_type'][name]['emr'])
              for name in ('single','double')]
    lines += [f'| {name} | {a:.6f} | {b:.6f} | {(b-a)*100:+.4f} |' for name,a,b in pairs]
    minrow = min(history, key=lambda r:r['val_nll'])
    lines += ['', f'Lowest Val NLL epoch: {minrow["epoch"]}. Last Val NLL: {history[-1]["val_nll"]:.6f}.',
              'TrainEval is a fixed balanced in-sample subset, not OOF and not full-Train accuracy.',
              'Both models use EMA for TrainEval/Val; augmentation is off. Different class proportions still limit the aggregate gap interpretation.',
              'Primary result remains best EMR. best_nll_model.pt is diagnostic; do not choose whichever result looks better.',
              'Review single Cargo/Tanker, single total, double pairs and overall jointly. Do not declare improvement from overall EMR alone.',
              'No data generator change. SIR 10-15 dB retained. Test was not loaded.', '']
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description='V6-A2 original-Train s42 comparison')
    parser.add_argument('--data-dir', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--reference-dir')
    parser.add_argument('--seed', type=int, choices=[42], default=42)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--smoke-test', action='store_true')
    args = parser.parse_args()
    if args.num_workers < 0:
        parser.error('num-workers must be nonnegative')
    if not args.smoke_test and not args.reference_dir:
        parser.error('Formal comparison requires --reference-dir pointing to A1 s42')
    cfg = ConfigV6A2()
    cfg.data_combined = str(Path(args.data_dir).resolve())
    cfg.output_dir = str(Path(args.output_dir).resolve())
    cfg.num_workers = args.num_workers
    seed_everything(args.seed)
    run(cfg, args)


if __name__ == '__main__':
    main()
