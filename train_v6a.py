"""Train MUART V6-A on Train/Val only.

This entry point has no Test mode, no threshold search, no teacher interface,
and no data-generation path.  It is intentionally separate from ``train.py``
so the V5 baseline remains untouched.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config_v6a import ConfigV6A, validate_v6a_config
from dataset import SAMPLE_TYPE_NAMES, create_loaders
from experiment_snapshot import build_experiment_snapshot, write_experiment_snapshot
from model_v6a import (
    LEGAL_SET_NAMES,
    MUARTV6A,
    count_parameters,
    multihot_to_set_targets,
    set_targets_to_multihot,
)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def query_orthogonal_loss(queries: torch.Tensor) -> torch.Tensor:
    normalized = F.normalize(queries, p=2, dim=1)
    similarity = normalized @ normalized.t()
    mask = ~torch.eye(
        queries.shape[0], dtype=torch.bool, device=queries.device
    )
    return (similarity[mask] ** 2).sum()


class WarmupCosineScheduler:
    def __init__(
        self,
        optimizer,
        warmup_epochs: int,
        total_epochs: int,
        lr_min: float,
        steps_per_epoch: int,
    ) -> None:
        self.optimizer = optimizer
        self.total_steps = max(1, total_epochs * steps_per_epoch)
        self.warmup_steps = max(0, warmup_epochs * steps_per_epoch)
        self.lr_min = lr_min
        self.current_step = 0

    def step(self) -> float:
        self.current_step += 1
        if self.warmup_steps and self.current_step <= self.warmup_steps:
            scale = self.current_step / self.warmup_steps
            for group in self.optimizer.param_groups:
                group["lr"] = group["initial_lr"] * scale
        else:
            denominator = max(1, self.total_steps - self.warmup_steps)
            progress = (self.current_step - self.warmup_steps) / denominator
            progress = min(max(progress, 0.0), 1.0)
            for group in self.optimizer.param_groups:
                initial = group["initial_lr"]
                group["lr"] = self.lr_min + 0.5 * (initial - self.lr_min) * (
                    1.0 + math.cos(math.pi * progress)
                )
        return float(self.optimizer.param_groups[0]["lr"])


class ModelEMA:
    def __init__(self, model: torch.nn.Module, decay: float) -> None:
        self.decay = decay
        self.ema = copy.deepcopy(model).eval()
        for parameter in self.ema.parameters():
            parameter.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        for ema_parameter, parameter in zip(
            self.ema.parameters(), model.parameters()
        ):
            ema_parameter.mul_(self.decay).add_(
                parameter.detach(), alpha=1.0 - self.decay
            )
        for ema_buffer, buffer in zip(self.ema.buffers(), model.buffers()):
            ema_buffer.copy_(buffer)


def conditional_set_loss(
    outputs: Mapping[str, torch.Tensor], labels: torch.Tensor
) -> Tuple[torch.Tensor, Dict[str, float], torch.Tensor]:
    """Exact seven-set NLL plus interpretable count/conditional components."""

    targets = multihot_to_set_targets(labels)
    joint_nll = F.nll_loss(outputs["set_log_probs"], targets)
    count_targets = labels.sum(dim=1).long()

    # Components are algebraic views of the same joint NLL, not extra losses.
    with torch.no_grad():
        count_ce = F.cross_entropy(outputs["tc_logits"], count_targets)
        single_mask = count_targets == 1
        double_mask = count_targets == 2
        single_ce = (
            F.cross_entropy(
                outputs["single_logits"][single_mask],
                targets[single_mask] - 1,
            )
            if bool(single_mask.any())
            else joint_nll.new_zeros(())
        )
        pair_ce = (
            F.cross_entropy(
                outputs["pair_logits"][double_mask],
                targets[double_mask] - 4,
            )
            if bool(double_mask.any())
            else joint_nll.new_zeros(())
        )
    parts = {
        "joint_nll": float(joint_nll.detach()),
        "count_ce": float(count_ce),
        "single_ce": float(single_ce),
        "pair_ce": float(pair_ce),
    }
    return joint_nll, parts, targets


def _safe_ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def compute_metrics(
    set_log_probs: torch.Tensor,
    labels: torch.Tensor,
    sample_types: torch.Tensor,
    class_names: Iterable[str],
) -> Dict[str, object]:
    """EMR, count accuracy, legal-set confusion, and per-class metrics."""

    true_sets = multihot_to_set_targets(labels).cpu()
    pred_sets = set_log_probs.argmax(dim=1).cpu()
    true_multihot = labels.float().cpu()
    pred_multihot = set_targets_to_multihot(pred_sets).cpu()
    sample_types = sample_types.cpu()
    n = int(labels.shape[0])

    correct = pred_sets == true_sets
    true_counts = true_multihot.sum(dim=1).long()
    pred_counts = pred_multihot.sum(dim=1).long()

    confusion = torch.zeros((7, 7), dtype=torch.long)
    for true_idx, pred_idx in zip(true_sets.tolist(), pred_sets.tolist()):
        confusion[true_idx, pred_idx] += 1

    by_legal_set: Dict[str, object] = {}
    for index, name in enumerate(LEGAL_SET_NAMES):
        mask = true_sets == index
        count = int(mask.sum())
        hits = int(correct[mask].sum())
        by_legal_set[name] = {
            "n": count,
            "correct": hits,
            "emr": _safe_ratio(hits, count),
        }

    by_sample_type: Dict[str, object] = {}
    for index, name in enumerate(SAMPLE_TYPE_NAMES):
        mask = sample_types == index
        count = int(mask.sum())
        hits = int(correct[mask].sum())
        by_sample_type[name] = {
            "n": count,
            "correct": hits,
            "emr": _safe_ratio(hits, count),
        }

    per_class: Dict[str, object] = {}
    for index, name in enumerate(class_names):
        predicted = pred_multihot[:, index].bool()
        actual = true_multihot[:, index].bool()
        tp = int((predicted & actual).sum())
        fp = int((predicted & ~actual).sum())
        fn = int((~predicted & actual).sum())
        precision = _safe_ratio(tp, tp + fp)
        recall = _safe_ratio(tp, tp + fn)
        f1 = _safe_ratio(2 * precision * recall, precision + recall)
        per_class[name] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "tp": tp,
            "fp": fp,
            "fn": fn,
        }

    return {
        "n": n,
        "emr": _safe_ratio(int(correct.sum()), n),
        "count_accuracy": _safe_ratio(int((pred_counts == true_counts).sum()), n),
        "by_legal_set": by_legal_set,
        "by_sample_type": by_sample_type,
        "per_class": per_class,
        "confusion_order": list(LEGAL_SET_NAMES),
        "confusion_matrix": confusion.tolist(),
    }


def _move_batch(batch, device: torch.device, use_demon: bool):
    lofar = batch[0].to(device, non_blocking=True)
    demon = (
        batch[1].to(device, non_blocking=True)
        if use_demon and batch[1] is not None
        else None
    )
    labels = batch[2].to(device)
    return lofar, demon, labels, batch[3]


def train_one_epoch(
    model: MUARTV6A,
    loader,
    optimizer,
    scheduler: WarmupCosineScheduler,
    device: torch.device,
    scaler,
    clip_norm: float,
    ortho_weight: float,
    ema: ModelEMA | None,
    epoch: int,
    total_epochs: int,
    use_demon: bool,
) -> Dict[str, float]:
    model.train()
    totals = {
        "samples": 0,
        "loss": 0.0,
        "nll": 0.0,
        "ortho": 0.0,
        "correct": 0,
        "count_correct": 0,
    }
    start = time.time()

    for batch_index, batch in enumerate(loader, start=1):
        lofar, demon, labels, _ = _move_batch(batch, device, use_demon)
        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=scaler is not None):
            outputs = model(lofar, demon)
            nll, _, targets = conditional_set_loss(outputs, labels)
            ortho = query_orthogonal_loss(
                model.classifier.query_attn.queries
            )
            loss = nll + ortho_weight * ortho

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
            optimizer.step()
        scheduler.step()
        if ema is not None:
            ema.update(model)

        batch_size = int(labels.shape[0])
        pred_sets = outputs["set_log_probs"].argmax(dim=1)
        count_targets = labels.sum(dim=1).long()
        totals["samples"] += batch_size
        totals["loss"] += float(loss.detach()) * batch_size
        totals["nll"] += float(nll.detach()) * batch_size
        totals["ortho"] += float(ortho.detach()) * batch_size
        totals["correct"] += int((pred_sets == targets).sum())
        totals["count_correct"] += int(
            (outputs["tc_logits"].argmax(dim=1) == count_targets).sum()
        )

        if batch_index % 100 == 0:
            seen = totals["samples"]
            print(
                f"    Epoch {epoch}/{total_epochs} "
                f"[{batch_index}/{len(loader)}] "
                f"loss={totals['loss']/seen:.4f} "
                f"EMR={totals['correct']/seen:.4f} "
                f"TC={totals['count_correct']/seen:.4f} "
                f"lr={optimizer.param_groups[0]['lr']:.2e}"
            )

    seen = totals["samples"]
    return {
        "loss": totals["loss"] / seen,
        "nll": totals["nll"] / seen,
        "ortho": totals["ortho"] / seen,
        "emr": totals["correct"] / seen,
        "count_accuracy": totals["count_correct"] / seen,
        "seconds": time.time() - start,
    }


@torch.no_grad()
def evaluate(
    model: MUARTV6A,
    loader,
    device: torch.device,
    class_names: Iterable[str],
    use_demon: bool,
) -> Tuple[float, Dict[str, object]]:
    model.eval()
    total_nll = 0.0
    total_samples = 0
    all_log_probs: List[torch.Tensor] = []
    all_labels: List[torch.Tensor] = []
    all_types: List[torch.Tensor] = []

    for batch in loader:
        lofar, demon, labels, sample_types = _move_batch(
            batch, device, use_demon
        )
        outputs = model(lofar, demon)
        nll, _, _ = conditional_set_loss(outputs, labels)
        batch_size = int(labels.shape[0])
        total_nll += float(nll) * batch_size
        total_samples += batch_size
        all_log_probs.append(outputs["set_log_probs"].cpu())
        all_labels.append(labels.cpu())
        all_types.append(sample_types.cpu())

    log_probs = torch.cat(all_log_probs)
    labels = torch.cat(all_labels)
    sample_types = torch.cat(all_types)
    metrics = compute_metrics(log_probs, labels, sample_types, class_names)
    return total_nll / total_samples, metrics


def _print_val_metrics(metrics: Mapping[str, object]) -> None:
    print(
        f"    Val: EMR={metrics['emr']:.4f}, "
        f"count_acc={metrics['count_accuracy']:.4f}"
    )
    print("    Legal-set EMR:")
    for name, row in metrics["by_legal_set"].items():
        print(f"      {name:>14s}: {row['emr']:.4f} (n={row['n']})")
    print("    Per-class P/R/F1:")
    for name, row in metrics["per_class"].items():
        print(
            f"      {name:>14s}: P={row['precision']:.4f} "
            f"R={row['recall']:.4f} F1={row['f1']:.4f}"
        )


def _atomic_torch_save(payload: object, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def _prepare_empty_output(path: Path) -> None:
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output directory: {path}")
    path.mkdir(parents=True, exist_ok=True)


def smoke_test(config: ConfigV6A) -> None:
    """One Train forward/backward and one Val forward; never constructs Test."""

    validate_v6a_config(config)
    device = torch.device(config.device)
    print(f"V6-A smoke test on {device}; Test will not be constructed")
    train_loader, val_loader, test_loader, info = create_loaders(
        config, include_test=False
    )
    if test_loader is not None or info.get("n_test") is not None:
        raise RuntimeError("Test loader was unexpectedly constructed")

    model = MUARTV6A(config).to(device)
    optimizer = AdamW(
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay
    )
    batch = next(iter(train_loader))
    lofar, demon, labels, _ = _move_batch(batch, device, config.use_demon)
    outputs = model(lofar, demon)
    nll, parts, _ = conditional_set_loss(outputs, labels)
    ortho = query_orthogonal_loss(model.classifier.query_attn.queries)
    loss = nll + config.ortho_loss_weight * ortho
    loss.backward()
    grad_ok = any(
        parameter.grad is not None
        and torch.isfinite(parameter.grad).all()
        and float(parameter.grad.abs().sum()) > 0
        for parameter in model.backbone.parameters()
    )
    if not grad_ok:
        raise RuntimeError("smoke test found no finite non-zero backbone gradient")
    optimizer.step()

    model.eval()
    with torch.no_grad():
        val_batch = next(iter(val_loader))
        v_lofar, v_demon, v_labels, _ = _move_batch(
            val_batch, device, config.use_demon
        )
        val_outputs = model(v_lofar, v_demon)
        probabilities = val_outputs["set_log_probs"].exp()
        if not torch.allclose(
            probabilities.sum(dim=1),
            torch.ones(probabilities.shape[0], device=device),
            atol=1e-5,
            rtol=1e-5,
        ):
            raise RuntimeError("seven legal-set probabilities do not sum to one")
        multihot_to_set_targets(v_labels)

    print(
        "PASS: one Train backward step and one Val forward step completed; "
        f"Train/Val={info['n_train']}/{info['n_val']}; "
        f"loss={float(loss.detach()):.4f}; components={parts}; Test NOT LOADED"
    )


def train(config: ConfigV6A, args: argparse.Namespace) -> Dict[str, object]:
    validate_v6a_config(config)
    device = torch.device(config.device)
    output_dir = Path(config.output_dir).resolve()
    _prepare_empty_output(output_dir)

    print(f"Device: {device}")
    print("Loading Train/Val only; Test will not be constructed")
    train_loader, val_loader, test_loader, info = create_loaders(
        config, include_test=False
    )
    if test_loader is not None or info.get("n_test") is not None:
        raise RuntimeError("Test loader was unexpectedly constructed")

    snapshot = build_experiment_snapshot(config, info, vars(args))
    snapshot["v6a_contract"] = {
        "architecture": "shared E8 encoder + TC/single/pair conditional heads",
        "legal_set_order": list(LEGAL_SET_NAMES),
        "objective": "negative log likelihood of normalized seven-set distribution",
        "distillation": False,
        "prototype_or_contrastive_loss": False,
        "test_loaded": False,
    }
    write_experiment_snapshot(
        snapshot, str(output_dir / "experiment_config.json")
    )

    model = MUARTV6A(config).to(device)
    parameter_count = count_parameters(model)
    print(f"Model: {config.architecture_name}; parameters={parameter_count:,}")
    print(f"Output: {output_dir}")

    optimizer = AdamW(
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay
    )
    for group in optimizer.param_groups:
        group["initial_lr"] = group["lr"]
    scheduler = WarmupCosineScheduler(
        optimizer,
        warmup_epochs=config.warmup_epochs,
        total_epochs=config.num_epochs,
        lr_min=config.lr_min,
        steps_per_epoch=len(train_loader),
    )
    scaler = torch.cuda.amp.GradScaler(
        enabled=bool(config.use_amp and device.type == "cuda")
    )
    scaler_or_none = scaler if scaler.is_enabled() else None
    ema = (
        ModelEMA(model, config.ema_decay)
        if float(getattr(config, "ema_decay", 0.0)) > 0
        else None
    )

    best_emr = -1.0
    best_epoch = 0
    patience = 0
    history: List[Dict[str, object]] = []
    checkpoint_path = output_dir / "best_model.pt"

    for epoch in range(1, config.num_epochs + 1):
        train_row = train_one_epoch(
            model,
            train_loader,
            optimizer,
            scheduler,
            device,
            scaler_or_none,
            config.clip_grad_norm,
            config.ortho_loss_weight,
            ema,
            epoch,
            config.num_epochs,
            config.use_demon,
        )
        eval_model = ema.ema if ema is not None else model
        val_nll, val_metrics = evaluate(
            eval_model,
            val_loader,
            device,
            config.class_names,
            config.use_demon,
        )
        print(
            f"\nEpoch {epoch}/{config.num_epochs} ({train_row['seconds']:.0f}s): "
            f"Train loss={train_row['loss']:.4f}, "
            f"EMR={train_row['emr']:.4f}, "
            f"TC={train_row['count_accuracy']:.4f}; Val NLL={val_nll:.4f}"
        )
        _print_val_metrics(val_metrics)

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_row["loss"],
                "train_nll": train_row["nll"],
                "train_ortho": train_row["ortho"],
                "train_emr": train_row["emr"],
                "train_count_accuracy": train_row["count_accuracy"],
                "val_nll": val_nll,
                "val_emr": val_metrics["emr"],
                "val_count_accuracy": val_metrics["count_accuracy"],
                "lr": optimizer.param_groups[0]["lr"],
            }
        )

        if float(val_metrics["emr"]) > best_emr:
            best_emr = float(val_metrics["emr"])
            best_epoch = epoch
            patience = 0
            selected_model = ema.ema if ema is not None else model
            payload = {
                "checkpoint_schema_version": config.checkpoint_schema_version,
                "architecture": config.architecture_name,
                "legal_set_names": list(LEGAL_SET_NAMES),
                "epoch": epoch,
                "model_state_dict": selected_model.state_dict(),
                "raw_model_state_dict": model.state_dict(),
                "ema_state_dict": ema.ema.state_dict() if ema is not None else None,
                "optimizer_state_dict": optimizer.state_dict(),
                "val_nll": val_nll,
                "val_acc": best_emr,
                "val_emr": best_emr,
                "val_metrics": val_metrics,
                "config": snapshot["resolved_config"],
                "normalization_stats": snapshot["normalization_stats"],
                "experiment_snapshot": snapshot,
                "test_loaded": False,
            }
            _atomic_torch_save(payload, checkpoint_path)
            print(f"    Saved best V6-A checkpoint: Val EMR={best_emr:.4f}")
        else:
            patience += 1
            if patience >= config.early_stop_patience:
                print(f"    Early stop: patience={config.early_stop_patience}")
                break

    with (output_dir / "history.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)

    checkpoint = torch.load(
        checkpoint_path, map_location=device, weights_only=False
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    final_nll, final_metrics = evaluate(
        model, val_loader, device, config.class_names, config.use_demon
    )
    result = {
        "stage": "V6-A",
        "seed": args.seed,
        "best_epoch": best_epoch,
        "best_val_emr": best_emr,
        "reloaded_val_nll": final_nll,
        "reloaded_val_metrics": final_metrics,
        "checkpoint": str(checkpoint_path),
        "test_loaded": False,
        "test_evaluated": False,
    }
    write_experiment_snapshot(result, str(output_dir / "val_metrics.json"))

    with (output_dir / "val_results.txt").open(
        "w", encoding="utf-8", newline="\n"
    ) as handle:
        handle.write("MUART V6-A conditional legal-set classification\n")
        handle.write(f"Seed: {args.seed}\n")
        handle.write(f"Best epoch: {best_epoch}\n")
        handle.write(f"Val EMR: {best_emr:.6f}\n")
        handle.write(f"Val NLL after reload: {final_nll:.6f}\n")
        handle.write(f"Val count accuracy: {final_metrics['count_accuracy']:.6f}\n")
        for name, row in final_metrics["by_legal_set"].items():
            handle.write(
                f"Set {name}: EMR={row['emr']:.6f}, N={row['n']}\n"
            )
        handle.write("Test: NOT LOADED / NOT EVALUATED\n")

    print("\nV6-A complete (Train/Val only). Test was not loaded or evaluated.")
    print(f"Best epoch={best_epoch}; Val EMR={best_emr:.6f}")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="MUART V6-A Train/Val-only conditional set classifier"
    )
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--tag", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="one Train backward and one Val forward; writes no checkpoint",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.seed < 0:
        parser.error("--seed must be non-negative")
    seed_everything(args.seed)

    config = ConfigV6A()
    config.data_combined = os.path.abspath(os.path.expanduser(args.data_dir))
    config.val_only = True
    if args.epochs is not None:
        if args.epochs <= 0:
            parser.error("--epochs must be positive")
        config.num_epochs = args.epochs
    if args.batch_size is not None:
        if args.batch_size <= 0:
            parser.error("--batch-size must be positive")
        config.batch_size = args.batch_size
    if args.patience is not None:
        if args.patience <= 0:
            parser.error("--patience must be positive")
        config.early_stop_patience = args.patience
    if args.num_workers is not None:
        if args.num_workers < 0:
            parser.error("--num-workers must be non-negative")
        config.num_workers = args.num_workers
    if args.lr is not None:
        if args.lr <= 0:
            parser.error("--lr must be positive")
        config.lr = args.lr

    validate_v6a_config(config)
    if args.smoke_test:
        smoke_test(config)
        return

    code_dir = Path(__file__).resolve().parent
    output_root = (
        Path(args.output_root).expanduser().resolve()
        if args.output_root
        else code_dir.parent / f"{code_dir.name}_runs"
    )
    tag = args.tag or f"v6a_condset_s{args.seed}"
    config.output_dir = str(output_root / f"checkpoints_{tag}")
    train(config, args)


if __name__ == "__main__":
    main()
