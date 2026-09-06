"""Audit D1 Train recording sampling and frozen-Val identity without using Test."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path


TYPE_ACTIVE = {
    "noise": (), "0": (0,), "1": (1,), "2": (2,),
    "0_1": (0, 1), "0_2": (0, 2), "1_2": (1, 2),
}
LABEL_NAME = {0: "Cargo", 1: "Tanker", 2: "Tug"}
LABEL_FOLDER = {0: "0", 1: "1", 2: "3"}
FILE_FIELD = {0: "fileA", 1: "fileB", 2: "fileC"}
EXPECTED_TRAIN = {
    "noise": 1512, "0": 1058, "1": 1058, "2": 1058,
    "0_1": 2268, "0_2": 2268, "1_2": 2268,
}
EXPECTED_VAL = {
    "noise": 324, "0": 227, "1": 227, "2": 227,
    "0_1": 486, "0_2": 486, "1_2": 486,
}
EXPECTED_SOURCE_SPLIT_COUNTS = {
    "Cargo": {"Train": 2860, "Val": 680},
    "Tanker": {"Train": 3053, "Val": 705},
    "Tug": {"Train": 4085, "Val": 708},
}
GENERATOR_FILES = (
    "build_source_sampling_pool.m", "sample_source_from_pool.m",
    "gen_sl_dataset.m", "d1_recording_uniform_config.m", "main_cargo_sl_d1.m",
    "audit_d1_dataset.py",
)


def atomic_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                         encoding="utf-8")
    temporary.replace(path)


def atomic_text(path: Path, value: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value, encoding="utf-8", newline="\n")
    temporary.replace(path)


def split_tree_content_hash(root: Path, split: str) -> dict:
    """Hash every file below {type}/{split}; never walks any other split."""
    records = []
    digest = hashlib.sha256()
    for type_name in TYPE_ACTIVE:
        split_dir = root / type_name / split
        if not split_dir.is_dir():
            raise FileNotFoundError(f"missing frozen split directory: {split_dir}")
        for path in sorted(item for item in split_dir.rglob("*") if item.is_file()):
            rel = path.relative_to(root).as_posix()
            content_hash = hashlib.sha256(path.read_bytes()).hexdigest()
            records.append((rel, path.stat().st_size, content_hash))
    for rel, size, content_hash in records:
        digest.update(f"{rel}\t{size}\t{content_hash}\n".encode("utf-8"))
    return {"file_count": len(records), "sha256": digest.hexdigest(),
            "semantics": "sha256(relative path, byte size, per-file content sha256)"}


def count_streams(root: Path, split: str) -> dict[str, dict[str, int]]:
    result = {}
    for type_name in TYPE_ACTIVE:
        current = {}
        for stream in ("mix", "s1", "s2", "s3"):
            directory = root / type_name / split / stream
            if not directory.is_dir():
                raise FileNotFoundError(f"missing stream directory: {directory}")
            current[stream] = sum(1 for path in directory.iterdir()
                                  if path.is_file() and path.suffix.lower() == ".wav")
        result[type_name] = current
    return result


def load_recording_maps(template_root: Path) -> dict[int, dict[str, str]]:
    output = {}
    for label, folder in LABEL_FOLDER.items():
        path = template_root / folder / "segment_mapping.txt"
        mapping = {}
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            next(handle, None)
            for line in handle:
                fields = [item.strip() for item in line.rstrip("\r\n").split("\t")]
                if len(fields) >= 3 and fields[0]:
                    mapping[fields[0]] = f"{fields[1].lower()}/{fields[2].lower()}"
        if not mapping:
            raise ValueError(f"empty recording mapping: {path}")
        output[label] = mapping
    return output


def collect_accepted_exposure(dataset_root: Path,
                              mappings: dict[int, dict[str, str]]) -> dict[str, Counter]:
    counts = {name: Counter() for name in LABEL_NAME.values()}
    for type_name, active in TYPE_ACTIVE.items():
        if not active:
            continue
        path = dataset_root / type_name / "Train" / "all_info.txt"
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            for row in reader:
                for label in active:
                    filename = row[FILE_FIELD[label]]
                    try:
                        recording = mappings[label][filename]
                    except KeyError as exc:
                        raise ValueError(f"unmapped Train template {label}/{filename}") from exc
                    counts[LABEL_NAME[label]][recording] += 1
    return counts


def audit_source_split_isolation(split_root: Path,
                                 mappings: dict[int, dict[str, str]]) -> tuple[dict, list[str]]:
    """Audit only Train/Val source pools; never constructs or reads another split."""
    result, failures = {}, []
    for label, name in LABEL_NAME.items():
        inventories = {}
        for split in ("Train", "Val"):
            directory = split_root / split / LABEL_FOLDER[label]
            if not directory.is_dir():
                raise FileNotFoundError(f"missing source split directory: {directory}")
            filenames = {path.name for path in directory.glob("*.wav") if path.is_file()}
            missing = sorted(filenames - set(mappings[label]))
            if missing:
                raise ValueError(f"source split contains unmapped {name} template: {missing[0]}")
            recordings = {mappings[label][filename] for filename in filenames}
            parent_groups = {recording.split("/", 1)[0] for recording in recordings}
            inventories[split] = {"templates": filenames, "recordings": recordings,
                                  "parent_groups": parent_groups}
            expected = EXPECTED_SOURCE_SPLIT_COUNTS[name][split]
            if len(filenames) != expected:
                failures.append(
                    f"{name} source {split} template count {len(filenames)} != {expected}")
        template_overlap = inventories["Train"]["templates"] & inventories["Val"]["templates"]
        recording_overlap = inventories["Train"]["recordings"] & inventories["Val"]["recordings"]
        parent_overlap = inventories["Train"]["parent_groups"] & inventories["Val"]["parent_groups"]
        result[name] = {
            "train_templates": len(inventories["Train"]["templates"]),
            "val_templates": len(inventories["Val"]["templates"]),
            "train_recordings": len(inventories["Train"]["recordings"]),
            "val_recordings": len(inventories["Val"]["recordings"]),
            "train_parent_groups": len(inventories["Train"]["parent_groups"]),
            "val_parent_groups": len(inventories["Val"]["parent_groups"]),
            "template_overlap": len(template_overlap),
            "recording_overlap": len(recording_overlap),
            "parent_group_overlap": len(parent_overlap),
            "vessel_id_available": False,
            "parent_group_semantics": "Parent_Folder proxy, not a verified physical vessel ID",
        }
        if template_overlap:
            failures.append(f"{name} Train/Val template overlap: {len(template_overlap)}")
        if recording_overlap:
            failures.append(f"{name} Train/Val recording overlap: {len(recording_overlap)}")
        if parent_overlap:
            failures.append(f"{name} Train/Val Parent_Folder overlap: {len(parent_overlap)}")
    return result, failures


def collect_sampling_audits(dataset_root: Path) -> tuple[dict[str, Counter],
                                                         dict[str, Counter],
                                                         dict[str, dict[str, int]]]:
    candidate = {name: Counter() for name in LABEL_NAME.values()}
    accepted = {name: Counter() for name in LABEL_NAME.values()}
    templates = {name: {} for name in LABEL_NAME.values()}
    for type_name, active in TYPE_ACTIVE.items():
        if not active:
            continue
        path = dataset_root / type_name / "Train" / "source_sampling_info.tsv"
        if not path.is_file():
            raise FileNotFoundError(f"missing source sampling audit: {path}")
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle, delimiter="\t"))
        observed_labels = set()
        for row in rows:
            if row["source_sampling_mode"] != "recording_uniform":
                raise ValueError(f"non-D1 sampling mode in {path}")
            label = int(row["label_idx"])
            if label not in active:
                raise ValueError(f"unexpected label {label} in {path}")
            observed_labels.add(label)
            name, recording = LABEL_NAME[label], row["recording_key"]
            candidate[name][recording] += int(row["candidate_uses"])
            accepted[name][recording] += int(row["accepted_uses"])
            count = int(row["templates_in_split"])
            old = templates[name].get(recording)
            if old is not None and old != count:
                raise ValueError(f"template-count mismatch for {name}/{recording}")
            templates[name][recording] = count
        if observed_labels != set(active):
            raise ValueError(f"incomplete labels in {path}: {observed_labels}")
    return candidate, accepted, templates


def dispersion(counter: Counter, universe: set[str]) -> dict:
    values = [int(counter.get(key, 0)) for key in sorted(universe)]
    total = sum(values)
    mean = total / len(values) if values else 0.0
    cv = statistics.pstdev(values) / mean if mean else math.inf
    ordered = sorted(values)
    gini = 0.0
    if total and ordered:
        n = len(ordered)
        gini = (2 * sum((i + 1) * value for i, value in enumerate(ordered)) /
                (n * total) - (n + 1) / n)
    return {
        "recordings": len(values), "total_uses": total,
        "coverage": sum(value > 0 for value in values) / len(values) if values else 0.0,
        "min": min(values) if values else 0, "max": max(values) if values else 0,
        "mean": mean, "cv": cv, "gini": gini,
        "max_share": max(values) / total if total else 0.0,
    }


def verify_sir_audits(dataset_root: Path) -> dict:
    output = {}
    for type_name in ("0_1", "0_2", "1_2"):
        path = dataset_root / type_name / "Train" / "sir_sampling_info.txt"
        values = {}
        for line in path.read_text(encoding="utf-8-sig").splitlines():
            if "\t" in line:
                key, value = line.split("\t", 1)
                values[key] = value
        if values.get("policy") != "stratified":
            raise ValueError(f"invalid SIR policy: {path}")
        if values.get("target_counts") != values.get("accepted_counts"):
            raise ValueError(f"SIR quota mismatch: {path}")
        if values.get("gain_adjustment_applied") != "0":
            raise ValueError(f"SIR gain adjustment detected: {path}")
        output[type_name] = values
    return output


def numeric_summary(dataset_root: Path) -> dict:
    wanted = ("snr_band_db", "overlap_start_s", "overlap_end_s",
              "sir_rx_overlap_band_db", "r1_km", "r2_km", "r3_km",
              "z_src1", "z_src2", "z_src3")
    values = {name: [] for name in wanted}
    for type_name in TYPE_ACTIVE:
        path = dataset_root / type_name / "Train" / "all_info.txt"
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle, delimiter="\t"):
                for name in wanted:
                    try:
                        value = float(row[name])
                    except (KeyError, ValueError):
                        continue
                    if math.isfinite(value) and (not name.startswith(("r", "z_src")) or value > 0):
                        values[name].append(value)
    result = {}
    for name, current in values.items():
        current.sort()
        if current:
            result[name] = {"n": len(current), "mean": statistics.fmean(current),
                            "median": statistics.median(current),
                            "min": current[0], "max": current[-1]}
    return result


def run_audit(reference_root: Path, d1_root: Path, template_root: Path,
              split_root: Path) -> dict:
    failures, warnings = [], []
    for root in (reference_root, d1_root, template_root, split_root):
        if not root.is_dir():
            failures.append(f"missing directory: {root}")
    if failures:
        return {"status": "FAIL", "failures": failures, "warnings": warnings}

    protocol = d1_root / "d1_protocol.txt"
    if not protocol.is_file() or "test_loaded\t0" not in protocol.read_text(encoding="utf-8-sig"):
        failures.append("missing or invalid d1_protocol.txt")

    try:
        reference_val = split_tree_content_hash(reference_root, "Val")
        d1_val = split_tree_content_hash(d1_root, "Val")
        if reference_val != d1_val:
            failures.append("D1 Val is not byte-identical to the frozen reference Val")
    except Exception as exc:
        failures.append(f"Val hash audit failed: {exc}")
        reference_val = d1_val = {}

    stream_counts = {}
    try:
        train_counts = count_streams(d1_root, "Train")
        val_counts = count_streams(d1_root, "Val")
        stream_counts = {"Train": train_counts, "Val": val_counts}
        for type_name in TYPE_ACTIVE:
            expected_train, expected_val = EXPECTED_TRAIN[type_name], EXPECTED_VAL[type_name]
            if any(value != expected_train for value in train_counts[type_name].values()):
                failures.append(f"D1 Train stream count mismatch: {type_name}")
            if any(value != expected_val for value in val_counts[type_name].values()):
                failures.append(f"D1 Val stream count mismatch: {type_name}")
    except Exception as exc:
        failures.append(f"stream-count audit failed: {exc}")

    exposure, split_isolation = {}, {}
    try:
        mappings = load_recording_maps(template_root)
        split_isolation, isolation_failures = audit_source_split_isolation(split_root, mappings)
        failures.extend(isolation_failures)
        warnings.append(
            "No explicit physical vessel ID is present in segment_mapping.txt; "
            "Parent_Folder is audited as a disjoint proxy, so true vessel-identity "
            "isolation cannot be proven from current metadata.")
        old_accepted = collect_accepted_exposure(reference_root, mappings)
        new_reconstructed = collect_accepted_exposure(d1_root, mappings)
        candidate, new_audited, template_counts = collect_sampling_audits(d1_root)
        for name in LABEL_NAME.values():
            universe = set(template_counts[name])
            if not universe:
                failures.append(f"empty D1 recording universe: {name}")
                continue
            if any(recording not in universe for recording in new_reconstructed[name]):
                failures.append(f"D1 all_info contains recording outside audit pool: {name}")
            if Counter(new_reconstructed[name]) != Counter(new_audited[name]):
                failures.append(f"accepted source audit does not match all_info: {name}")
            old_stats = dispersion(old_accepted[name], universe)
            candidate_stats = dispersion(candidate[name], universe)
            new_stats = dispersion(new_audited[name], universe)
            exposure[name] = {"template_uniform_reference": old_stats,
                              "d1_candidates": candidate_stats,
                              "d1_accepted": new_stats,
                              "templates_per_recording": {
                                  "min": min(template_counts[name].values()),
                                  "max": max(template_counts[name].values()),
                                  "median": statistics.median(template_counts[name].values())}}
            if new_stats["total_uses"] != 5594:
                failures.append(f"D1 positive exposure total is not 5594: {name}")
            if new_stats["coverage"] != 1.0:
                failures.append(f"D1 did not expose every Train recording: {name}")
            if not new_stats["cv"] < old_stats["cv"]:
                failures.append(f"D1 accepted recording CV did not improve: {name}")
            if candidate_stats["cv"] > 0.30:
                failures.append(f"D1 candidate recording CV exceeds 0.30: {name}")
    except Exception as exc:
        failures.append(f"recording-exposure audit failed: {exc}")

    try:
        sir = verify_sir_audits(d1_root)
    except Exception as exc:
        failures.append(f"SIR audit failed: {exc}")
        sir = {}

    try:
        physical = {"reference_train": numeric_summary(reference_root),
                    "d1_train": numeric_summary(d1_root)}
    except Exception as exc:
        warnings.append(f"physical summary unavailable: {exc}")
        physical = {}

    code_root = Path(__file__).resolve().parent
    generator_sha256 = {
        name: hashlib.sha256((code_root / name).read_bytes()).hexdigest()
        for name in GENERATOR_FILES
    }
    return {
        "status": "PASS" if not failures else "FAIL",
        "scope": "Train generation plus byte-identical frozen Val; Test not enumerated or evaluated",
        "failures": failures, "warnings": warnings,
        "reference_root": str(reference_root.resolve()),
        "d1_root": str(d1_root.resolve()),
        "val_content_hash": {"reference": reference_val, "d1": d1_val},
        "stream_counts": stream_counts, "recording_exposure": exposure,
        "source_split_isolation": split_isolation,
        "sir_sampling": sir, "physical_summary": physical,
        "generator_code_sha256": generator_sha256,
    }


def markdown_report(result: dict) -> str:
    lines = ["# D1 recording-uniform dataset audit", "",
             f"Status: **{result['status']}**", "",
             result.get("scope", ""), ""]
    if result.get("failures"):
        lines += ["## Failures", ""] + [f"- {item}" for item in result["failures"]] + [""]
    if result.get("warnings"):
        lines += ["## Warnings", ""] + [f"- {item}" for item in result["warnings"]] + [""]
    if result.get("recording_exposure"):
        lines += ["## Recording exposure", "",
                  "| Class | Reference CV | D1 candidate CV | D1 accepted CV | D1 coverage |",
                  "|---|---:|---:|---:|---:|"]
        for name, values in result["recording_exposure"].items():
            old = values["template_uniform_reference"]
            candidate = values["d1_candidates"]
            new = values["d1_accepted"]
            lines.append(f"| {name} | {old['cv']:.4f} | {candidate['cv']:.4f} | "
                         f"{new['cv']:.4f} | {new['coverage']:.4f} |")
        lines.append("")
    if result.get("source_split_isolation"):
        lines += ["## Frozen source split isolation", "",
                  "| Class | Train/Val template overlap | recording overlap | Parent_Folder overlap | true vessel ID |",
                  "|---|---:|---:|---:|---|"]
        for name, values in result["source_split_isolation"].items():
            vessel = "available" if values["vessel_id_available"] else "unavailable"
            lines.append(
                f"| {name} | {values['template_overlap']} | {values['recording_overlap']} | "
                f"{values['parent_group_overlap']} | {vessel} |")
        lines.append("")
    val = result.get("val_content_hash", {})
    if val:
        lines += ["## Frozen Val", "",
                  f"Reference SHA-256: `{val.get('reference', {}).get('sha256', '')}`  ",
                  f"D1 SHA-256: `{val.get('d1', {}).get('sha256', '')}`", ""]
    lines += ["D1 changes only Train source selection. SIR 10-15 dB is retained; "
              "no source gain is adjusted.", ""]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-data-dir", required=True)
    parser.add_argument("--d1-data-dir", required=True)
    parser.add_argument("--template-root", required=True)
    parser.add_argument("--split-root", required=True)
    args = parser.parse_args()
    result = run_audit(Path(args.reference_data_dir), Path(args.d1_data_dir),
                       Path(args.template_root), Path(args.split_root))
    output = Path(args.d1_data_dir)
    atomic_json(output / "d1_audit.json", result)
    atomic_text(output / "d1_audit.md", markdown_report(result))
    print(f"D1 AUDIT {result['status']}: {output / 'd1_audit.md'}")
    if result["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
