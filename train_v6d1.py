"""Train unchanged V6-A2 on D1 recording-uniform Train and frozen Val."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
from pathlib import Path

# Import order is required by the user's Windows conda scientific runtime:
# NumPy/MKL must initialize before PyTorch's OpenMP runtime.
import numpy as np  # noqa: F401 - intentional runtime preflight
import torch

from config_v6d1 import ConfigV6D1, validate_v6d1_config
from d1_verdict import d1_diagnostic_verdict
from dataset import create_loaders, collate_multi
from experiment_snapshot import (build_dataset_manifest, build_experiment_snapshot,
                                 write_experiment_snapshot)
from model_v6a import LEGAL_SET_NAMES, multihot_to_set_targets, count_parameters
from model_v6a2 import MUARTV6A2
from train_v6a import ModelEMA, WarmupCosineScheduler, seed_everything
from train_v6a2 import (_atomic_torch_save, _move_batch, _prepare_empty_output,
                        evaluate, make_train_monitor, overfit_signal, preserve_rng,
                        train_epoch, write_csv, write_predictions)


FROZEN_CONFIG_KEYS = (
    "architecture_name", "batch_size", "num_epochs", "early_stop_patience",
    "lr", "lr_min", "weight_decay", "warmup_epochs", "ema_decay",
    "feature_type", "normalization_mode", "use_demon", "use_spec_gate",
    "dual_stream_fusion", "freq_shift_mode", "waveform_noise_mode",
    "use_speed_perturb", "presence_loss_weight", "pair_interaction_hidden",
    "pair_interaction_bound", "ortho_loss_weight",
)

def load_d1_audit(data_dir: Path) -> dict:
    path = data_dir / "d1_audit.json"
    if not path.is_file():
        raise FileNotFoundError(f"D1 audit is required before training: {path}")
    audit = json.loads(path.read_text(encoding="utf-8-sig"))
    if audit.get("status") != "PASS" or audit.get("failures"):
        raise ValueError("D1 dataset audit did not PASS")
    val = audit.get("val_content_hash", {})
    if val.get("reference", {}).get("sha256") != val.get("d1", {}).get("sha256"):
        raise ValueError("D1 audit does not prove byte-identical frozen Val")
    for name in ("Cargo", "Tanker", "Tug"):
        item = audit.get("recording_exposure", {}).get(name, {})
        old = item.get("template_uniform_reference", {})
        new = item.get("d1_accepted", {})
        if new.get("coverage") != 1.0 or not new.get("cv", math.inf) < old.get("cv", -math.inf):
            raise ValueError(f"D1 recording exposure did not improve for {name}")
    return audit


def d1_protocol_check(snapshot: dict, reference_dir: Path,
                      reference_data_dir: Path, d1_data_dir: Path) -> tuple[dict, dict]:
    experiment_path = reference_dir / "experiment_config.json"
    metrics_path = reference_dir / "val_metrics.json"
    if not experiment_path.is_file() or not metrics_path.is_file():
        raise FileNotFoundError("D1 reference must contain A2 experiment_config.json and val_metrics.json")
    reference = json.loads(experiment_path.read_text(encoding="utf-8-sig"))
    baseline = json.loads(metrics_path.read_text(encoding="utf-8-sig"))
    if baseline.get("stage") != "V6-A2" or baseline.get("seed") != 42:
        raise ValueError("D1 reference must be the formal V6-A2 s42 result")
    if baseline.get("test_loaded") is not False or baseline.get("test_evaluated") is not False:
        raise ValueError("D1 reference must be Train/Val-only")

    current_cfg = snapshot["resolved_config"]
    reference_cfg = reference["resolved_config"]
    for key in FROZEN_CONFIG_KEYS:
        if current_cfg.get(key) != reference_cfg.get(key):
            raise ValueError(f"D1 changed frozen model/training config: {key}")
    expected_counts = reference.get("loader_counts", {})
    if snapshot.get("loader_counts") != expected_counts:
        raise ValueError("D1 Train/Val sample counts differ from V6-A2")

    reference_val = build_dataset_manifest(reference_data_dir, include_splits={"Val"})
    d1_val = build_dataset_manifest(d1_data_dir, include_splits={"Val"})
    if (reference_val["path_size_sha256"] != d1_val["path_size_sha256"] or
            reference_val["wav_file_count"] != d1_val["wav_file_count"]):
        raise ValueError("D1 Val path/size manifest differs from frozen Val")
    reference_train = build_dataset_manifest(reference_data_dir, include_splits={"Train"})
    d1_train = build_dataset_manifest(d1_data_dir, include_splits={"Train"})
    # D1 deliberately preserves every WAV relative path, count, duration and
    # uncompressed byte size.  Therefore equality of the path/size manifest is
    # required; the changed source selection is proven separately by the D1
    # audit reconstructed from Train/all_info and source_sampling_info.tsv.
    if (reference_train["path_size_sha256"] != d1_train["path_size_sha256"] or
            reference_train["wav_file_count"] != d1_train["wav_file_count"]):
        raise ValueError("D1 Train path/size structure differs from the frozen A2 protocol")

    audit = load_d1_audit(d1_data_dir)
    if (Path(audit.get("reference_root", "")).resolve() != reference_data_dir.resolve() or
            Path(audit.get("d1_root", "")).resolve() != d1_data_dir.resolve()):
        raise ValueError("D1 audit roots do not match the formal Train comparison")
    return baseline, {"audit": audit, "reference_val_manifest": reference_val,
                      "d1_val_manifest": d1_val,
                      "reference_train_manifest": reference_train,
                      "d1_train_manifest": d1_train,
                      "train_change_evidence": audit["recording_exposure"]}


def comparison_report(result: dict, baseline: dict, history: list[dict]) -> str:
    current = result["reloaded_val_metrics"]
    previous = baseline["reloaded_val_metrics"]
    pairs = [("Overall", previous["emr"], current["emr"])]
    pairs += [(name, previous["by_legal_set"][name]["emr"],
               current["by_legal_set"][name]["emr"]) for name in LEGAL_SET_NAMES]
    pairs += [(name, previous["by_sample_type"][name]["emr"],
               current["by_sample_type"][name]["emr"]) for name in ("single", "double")]
    best_row = next(row for row in history if row["epoch"] == result["best_epoch"])
    nll_row = min(history, key=lambda row: row["val_nll"])
    verdict = result["d1_verdict"]
    lines = ["# V6-D1 recording-uniform Train versus V6-A2 original Train", "",
             "Same V6-A2 model, seed and physical protocol. Only Train source sampling changes; Val is byte-identical.", "",
             f"Primary checkpoint: maximum Val EMR, epoch {result['best_epoch']}.", "",
             "| Metric | A2 original Train | D1 Train | Change (percentage points) |",
             "|---|---:|---:|---:|"]
    lines += [f"| {name} | {old:.6f} | {new:.6f} | {(new-old)*100:+.4f} |"
              for name, old, new in pairs]
    lines += ["", f"Lowest Val NLL epoch: {nll_row['epoch']}.",
              f"Best-EMR TrainEval/Val gap: {(best_row['train_eval_emr']-best_row['val_emr'])*100:.4f} percentage points.",
              f"Best-EMR wrong confidence/Brier: {best_row['val_wrong_confidence']:.6f} / {best_row['val_brier']:.6f}.",
              "## Pre-registered D1 verdict", "",
              f"Verdict: **{verdict['label']}**.", "",
              "| Count diagnostic | V6-A2 | D1 | Delta | Strong target |",
              "|---|---:|---:|---:|---:|",
              f"| Cargo correct / 227 | {verdict['v6a2']['cargo_correct']} | {verdict['d1']['cargo_correct']} | {verdict['delta_d1_minus_v6a2']['cargo_correct']:+d} | >= {verdict['strong_recovery_targets']['cargo_correct']} |",
              f"| Single Cargo/Tanker swaps | {verdict['v6a2']['single_cargo_tanker_swaps']} | {verdict['d1']['single_cargo_tanker_swaps']} | {verdict['delta_d1_minus_v6a2']['single_cargo_tanker_swaps']:+d} | <= {verdict['strong_recovery_targets']['single_cargo_tanker_swaps']} |",
              f"| Tug-pair Cargo/Tanker swaps | {verdict['v6a2']['tug_pair_cargo_tanker_swaps']} | {verdict['d1']['tug_pair_cargo_tanker_swaps']} | {verdict['delta_d1_minus_v6a2']['tug_pair_cargo_tanker_swaps']:+d} | <= {verdict['strong_recovery_targets']['tug_pair_cargo_tanker_swaps']} |",
              f"| Overall correct | {verdict['v6a2']['overall_correct']} | {verdict['d1']['overall_correct']} | {verdict['delta_d1_minus_v6a2']['overall_correct']:+d} | no regression |",
              "",
              "D1 is successful only if Cargo and Cargo/Tanker identity errors improve together with generalization; overall EMR alone is insufficient.",
              "SIR 10-15 dB retained. No source gain adjustment. Test was not loaded.", ""]
    return "\n".join(lines)


def run(config, args):
    validate_v6d1_config(config)
    out = Path(config.output_dir).resolve()
    _prepare_empty_output(out)
    data_dir = Path(config.data_combined).resolve()
    audit = load_d1_audit(data_dir)
    for name in ("d1_audit.json", "d1_audit.md", "d1_protocol.txt"):
        source = data_dir / name
        if not source.is_file():
            raise FileNotFoundError(f"D1 provenance file is required: {source}")
        shutil.copy2(source, out / name)
    device = torch.device(config.device)
    train_loader, val_loader, forbidden_loader, info = create_loaders(config, include_test=False)
    if forbidden_loader is not None or info["n_test"] is not None:
        raise RuntimeError("Test must not be constructed")
    monitor = make_train_monitor(train_loader.dataset, config)
    snapshot = build_experiment_snapshot(config, info, vars(args))
    baseline = protocol = None
    if not args.smoke_test:
        baseline, protocol = d1_protocol_check(
            snapshot, Path(args.reference_dir).resolve(),
            Path(args.reference_data_dir).resolve(), data_dir)
    snapshot["d1_protocol"] = {
        "only_change": "Train source selection: recording-uniform then template-uniform",
        "model": "unchanged V6-A2; from scratch; no distillation",
        "val": "byte-identical frozen reference Val",
        "data_audit": audit,
        "formal_protocol_check": protocol,
        "monitor_ids": [Path(w).resolve().relative_to(data_dir).as_posix()
                        for w, _, _ in monitor.dataset.samples],
        "test_loaded": False,
    }
    code_dir = Path(__file__).resolve().parent
    names = ("config.py", "model.py", "dataset.py", "experiment_snapshot.py",
             "config_v6a.py", "model_v6a.py", "train_v6a.py", "config_v6a2.py",
             "model_v6a2.py", "train_v6a2.py", "config_v6d1.py", "d1_verdict.py",
             "train_v6d1.py")
    snapshot["code_sha256"] = {
        name: hashlib.sha256((code_dir / name).read_bytes()).hexdigest() for name in names}
    write_experiment_snapshot(snapshot, str(out / "experiment_config.json"))

    model = MUARTV6A2(config).to(device)
    print(f"D1 unchanged A2 parameters={count_parameters(model):,}; "
          f"Train={info['n_train']}; Val={info['n_val']}; Test NOT LOADED", flush=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr,
                                  weight_decay=config.weight_decay)
    for group in optimizer.param_groups:
        group["initial_lr"] = group["lr"]
    scheduler = WarmupCosineScheduler(optimizer, config.warmup_epochs, config.num_epochs,
                                     config.lr_min, len(train_loader))
    ema = ModelEMA(model, config.ema_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=config.use_amp and device.type == "cuda")
    scaler = scaler if scaler.is_enabled() else None

    if args.smoke_test:
        batch = collate_multi([monitor.dataset[next(
            i for i, (_, label, _) in enumerate(monitor.dataset.samples)
            if int(multihot_to_set_targets(torch.tensor([label])).item()) == k)]
            for k in range(7)])
        train_result = train_epoch(model, [batch], optimizer, scheduler, ema, scaler,
                                   device, config)
        with preserve_rng():
            val, _, _ = evaluate(ema.ema, [next(iter(val_loader))], device, config)
        _atomic_torch_save({"architecture": config.architecture_name,
                            "model_state_dict": ema.ema.state_dict()},
                           out / "smoke_checkpoint.pt")
        restored = MUARTV6A2(config).to(device).eval()
        restored.load_state_dict(torch.load(out / "smoke_checkpoint.pt", map_location=device,
                                           weights_only=False)["model_state_dict"])
        x, d, _, _ = _move_batch(batch, device, config.use_demon)
        with torch.no_grad():
            expected = ema.ema(x, d)["set_log_probs"]
            actual = restored(x, d)["set_log_probs"]
        if not torch.allclose(expected, actual, atol=1e-6, rtol=1e-6):
            raise RuntimeError("D1 checkpoint round-trip mismatch")
        result = {"status": "PASS", "purpose": "D1 code/data smoke only",
                  "train_step": train_result, "val_forward_n": val["n"],
                  "checkpoint_roundtrip": True, "test_loaded": False}
        write_experiment_snapshot(result, str(out / "smoke_result.json"))
        print(f"D1 SMOKE PASS: {out}", flush=True)
        return result

    history, epoch_metrics = [], []
    best_emr, best_nll, patience = -1.0, math.inf, 0
    best_result = None
    for epoch in range(1, config.num_epochs + 1):
        train = train_epoch(model, train_loader, optimizer, scheduler, ema, scaler,
                            device, config)
        with preserve_rng():
            val, probabilities, targets = evaluate(ema.ema, val_loader, device, config)
            probe, _, _ = evaluate(ema.ema, monitor, device, config)
        row = {"epoch": epoch, "train_loss": train["loss"], "train_nll": train["nll"],
               "train_presence_bce": train["presence_bce"], "train_aug_emr": train["emr"],
               "train_eval_emr": probe["emr"], "train_eval_nll": probe["nll"],
               "val_emr": val["emr"], "val_nll": val["nll"],
               "val_presence_bce": val["presence_bce"],
               "val_count_accuracy": val["count_accuracy"],
               "val_wrong_confidence": val["wrong_mean_confidence"],
               "val_brier": val["set_brier"], "lr": optimizer.param_groups[0]["lr"],
               "skipped_amp_steps": train["skipped_amp_steps"]}
        for name in LEGAL_SET_NAMES:
            row["val_set_" + name] = val["by_legal_set"][name]["emr"]
        for kind in ("single", "double"):
            row["val_" + kind + "_emr"] = val["by_sample_type"][kind]["emr"]
            row["train_eval_" + kind + "_emr"] = probe["by_sample_type"][kind]["emr"]
        history.append(row)
        alert = overfit_signal(history, config)
        epoch_metrics.append({"epoch": epoch, "val": val, "train_eval": probe,
                              "overfit": alert})
        write_csv(out / "history.csv", history)
        write_experiment_snapshot(epoch_metrics, str(out / "epoch_metrics.json"))
        write_experiment_snapshot(alert, str(out / "overfit_watch.json"))
        print(f"Epoch {epoch}: Val EMR={val['emr']:.4f} NLL={val['nll']:.4f}; "
              f"TrainEval EMR={probe['emr']:.4f} NLL={probe['nll']:.4f}", flush=True)
        print("  Single sets: " + ", ".join(
            f"{name}={val['by_legal_set'][name]['emr']:.4f}"
            for name in ("Tanker", "Cargo", "Tug")), flush=True)
        if alert["suspected_overfit"]:
            print("  OVERFIT WATCH: TrainEval NLL improves while Val NLL remains high.",
                  flush=True)
        improved, nll_improved = val["emr"] > best_emr, val["nll"] < best_nll
        if improved or nll_improved:
            payload = {"architecture": config.architecture_name, "epoch": epoch,
                       "legal_set_names": list(LEGAL_SET_NAMES),
                       "model_state_dict": ema.ema.state_dict(),
                       "ema_state_dict": ema.ema.state_dict(),
                       "raw_model_state_dict": model.state_dict(),
                       "optimizer_state_dict": optimizer.state_dict(),
                       "val_acc": val["emr"], "val_metrics": val,
                       "config": snapshot["resolved_config"],
                       "normalization_stats": snapshot["normalization_stats"],
                       "experiment_snapshot": snapshot, "test_loaded": False}
            if improved:
                best_emr = val["emr"]
                best_result = {"stage": "V6-D1", "seed": args.seed,
                               "best_epoch": epoch, "best_val_emr": best_emr,
                               "reloaded_val_metrics": val, "test_loaded": False,
                               "test_evaluated": False,
                               "selection": "maximum Val EMR; EMA weights"}
                _atomic_torch_save(payload, out / "best_model.pt")
                write_predictions(out / "val_predictions.csv", val_loader,
                                  probabilities, targets, data_dir)
                write_experiment_snapshot(best_result, str(out / "val_metrics.json"))
            if nll_improved:
                best_nll = val["nll"]
                _atomic_torch_save(payload, out / "best_nll_model.pt")
                write_experiment_snapshot({"epoch": epoch, "metrics": val,
                                           "purpose": "diagnostic only"},
                                          str(out / "best_nll_metrics.json"))
        patience = 0 if improved else patience + 1
        if patience >= config.early_stop_patience:
            print(f"Early stop at epoch {epoch}: EMR patience={patience}", flush=True)
            break

    saved = torch.load(out / "best_model.pt", map_location=device, weights_only=False)
    model.load_state_dict(saved["model_state_dict"])
    with preserve_rng():
        reloaded, _, _ = evaluate(model, val_loader, device, config)
    if reloaded["confusion_matrix"] != best_result["reloaded_val_metrics"]["confusion_matrix"]:
        raise RuntimeError("D1 best checkpoint evaluation changed after reload")
    best_result["reloaded_val_metrics"] = reloaded
    best_result["checkpoint_reload_verified"] = True
    best_result["d1_verdict"] = d1_diagnostic_verdict(
        reloaded, baseline["reloaded_val_metrics"])
    write_experiment_snapshot(best_result, str(out / "val_metrics.json"))
    (out / "comparison.md").write_text(
        comparison_report(best_result, baseline, history), encoding="utf-8")
    print(f"COMPLETE: D1 best Val={best_emr:.6f}. Test NOT LOADED.", flush=True)
    return best_result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--reference-data-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--reference-dir")
    parser.add_argument("--seed", type=int, choices=[42], default=42)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()
    if args.num_workers < 0:
        parser.error("num-workers must be nonnegative")
    if not args.smoke_test and not args.reference_dir:
        parser.error("formal D1 comparison requires the V6-A2 reference directory")
    config = ConfigV6D1()
    config.data_combined = str(Path(args.data_dir).resolve())
    config.output_dir = str(Path(args.output_dir).resolve())
    config.num_workers = args.num_workers
    seed_everything(args.seed)
    run(config, args)


if __name__ == "__main__":
    main()
