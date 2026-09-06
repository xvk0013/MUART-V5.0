"""MUART 实验配置与数据清单快照。

本模块只读取配置和数据集目录元数据，不读取 WAV 内容。训练入口用它把
完整解析配置、归一化统计和路径/大小清单指纹写入 checkpoint 与 JSON。
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import torch


SNAPSHOT_SCHEMA_VERSION = 1


def _json_safe(value: Any) -> Any:
    """把常见配置值转换为可稳定写入 JSON 的形式。"""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, os.PathLike):
        return os.fspath(value)
    if isinstance(value, Mapping):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(v) for v in value]
    return repr(value)


def collect_resolved_config(config: Any) -> dict[str, Any]:
    """收集实例属性、继承的类属性和 @property 的最终解析值。"""
    resolved: dict[str, Any] = {}
    for name in sorted(dir(config)):
        if name.startswith('_'):
            continue
        try:
            value = getattr(config, name)
        except Exception as exc:  # pragma: no cover - 仅保留无法解析的异常信息
            resolved[name] = f'<unavailable: {type(exc).__name__}: {exc}>'
            continue
        if callable(value):
            continue
        resolved[name] = _json_safe(value)
    return resolved


def build_dataset_manifest(
    data_dir: str | os.PathLike[str],
    include_splits: set[str] | None = None,
) -> dict[str, Any]:
    """根据 WAV 相对路径和字节数生成轻量、确定性的清单指纹。

    该 SHA-256 不是音频内容哈希；它用于检测文件集合、相对路径或文件大小
    是否变化，避免训练时读取全部 WAV 内容造成巨大开销。
    """
    root = Path(data_dir).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f'数据集目录不存在: {root}')

    records: list[tuple[str, int]] = []
    counts_by_split: dict[str, int] = {}
    counts_by_type_split_stream: dict[str, int] = {}

    allowed = {name.casefold() for name in include_splits} \
        if include_splits is not None else None

    # topdown=True 允许在 {type}/ 层直接剪掉未授权 split，Val-only 时不会
    # 进入 Test 子树，也不会枚举其中的 WAV 文件名或大小。
    for dirpath, dirnames, filenames in os.walk(root, topdown=True):
        relative_dir = Path(dirpath).relative_to(root)
        if allowed is not None and len(relative_dir.parts) == 1:
            dirnames[:] = [name for name in dirnames if name.casefold() in allowed]

        for filename in filenames:
            if Path(filename).suffix.lower() != '.wav':
                continue
            path = Path(dirpath) / filename
            rel = path.relative_to(root).as_posix()
            parts = rel.split('/')
            if len(parts) < 3:
                continue
            sample_type, split, stream = parts[0], parts[1], parts[2]
            if allowed is not None and split.casefold() not in allowed:
                continue

            records.append((rel, path.stat().st_size))
            counts_by_split[split] = counts_by_split.get(split, 0) + 1
            key = f'{sample_type}/{split}/{stream}'
            counts_by_type_split_stream[key] = \
                counts_by_type_split_stream.get(key, 0) + 1

    records.sort()
    digest = hashlib.sha256()
    for rel, size in records:
        digest.update(f'{rel}\t{size}\n'.encode('utf-8'))

    return {
        'root': str(root),
        'included_splits': sorted(include_splits) if include_splits else 'all',
        'wav_file_count': len(records),
        'path_size_sha256': digest.hexdigest(),
        'hash_semantics': 'SHA-256(relative POSIX path + tab + byte size + newline)',
        'counts_by_split_all_streams': dict(sorted(counts_by_split.items())),
        'counts_by_type_split_stream': dict(sorted(counts_by_type_split_stream.items())),
    }


def build_experiment_snapshot(
    config: Any,
    loader_info: Mapping[str, Any],
    cli_args: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """构建可同时保存到 JSON 和 PyTorch checkpoint 的完整快照。"""
    resolved = collect_resolved_config(config)
    data_dir = resolved.get('data_combined')
    if not isinstance(data_dir, str):
        raise ValueError('解析后的 config.data_combined 不是有效路径字符串')

    return {
        'snapshot_schema_version': SNAPSHOT_SCHEMA_VERSION,
        'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'resolved_config': resolved,
        'cli_args': _json_safe(dict(cli_args or {})),
        'normalization_stats': {
            'mode': loader_info.get('normalization_mode', 'global'),
            'semantics': (
                'Train-only scalar log-spectrum mean/std'
                if loader_info.get('normalization_mode', 'global') == 'global'
                else 'Train-only per-input-channel log-spectrum mean/std'
            ),
            # 保留旧键名兼容已有读取器；per_channel 模式下值为按通道列表。
            'global_mean': _json_safe(loader_info['global_mean']),
            'global_std': _json_safe(loader_info['global_std']),
        },
        'loader_counts': {
            'n_train': loader_info.get('n_train'),
            'n_val': loader_info.get('n_val'),
            'n_test': loader_info.get('n_test'),
        },
        'dataset_manifest': build_dataset_manifest(
            data_dir,
            include_splits={'Train', 'Val'} if resolved.get('val_only') else None,
        ),
        'runtime': {
            'python': sys.version,
            'platform': platform.platform(),
            'torch': torch.__version__,
            'torch_cuda_build': torch.version.cuda,
            'cuda_available': torch.cuda.is_available(),
            'cuda_device_name': (
                torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
            ),
        },
    }


def write_experiment_snapshot(snapshot: Mapping[str, Any], output_path: str) -> None:
    """原子写入 UTF-8 JSON，避免留下半写入的配置文件。"""
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temp = output.with_suffix(output.suffix + '.tmp')
    with temp.open('w', encoding='utf-8', newline='\n') as handle:
        json.dump(snapshot, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write('\n')
    os.replace(temp, output)
