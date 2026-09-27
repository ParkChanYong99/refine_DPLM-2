#!/usr/bin/env python3
"""Read-only audit of parquet-referenced, precomputed GVP features.

This script deliberately does not infer a feature root, manufacture residue masks,
or match structures by basename.  Those choices make absence of evidence visible
instead of turning it into an apparently successful latent-pipeline test.
"""

from __future__ import annotations

import argparse
import math
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import pyarrow.parquet as pq
import torch


REQUIRED_COLUMNS = ("processed_path", "gvp_feat_path", "struct_seq", "aa_seq")


@dataclass(frozen=True)
class FeatureRow:
    split: str
    parquet: Path
    row_index: int
    raw_path: str
    resolved_path: Path
    processed_path: Any
    struct_seq: Any
    aa_seq: Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_dir", default="data-bin/pdb_swissprot")
    parser.add_argument("--feature_root", default=None)
    parser.add_argument("--max_rows", type=int, default=None)
    parser.add_argument("--num_probe_files", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def section(name: str) -> None:
    print(f"\n[{name}]")


def split_name(path: Path) -> str:
    lowered = [part.lower() for part in path.parts]
    if "train" in lowered:
        return "train"
    if "valid" in lowered or "validation" in lowered:
        return "valid"
    return "other"


def nonempty_string(value: Any) -> str | None:
    if value is None:
        return None
    value = str(value).strip()
    return value or None


def first_component(raw: str) -> str:
    parts = Path(raw).parts
    if not parts:
        return "<EMPTY>"
    if Path(raw).is_absolute():
        return parts[1] if len(parts) > 1 else parts[0]
    return parts[0]


def resolve_feature(raw: str, feature_root: Path | None) -> tuple[str, Path | None]:
    path = Path(raw)
    if path.is_absolute():
        return "absolute-direct", path
    if feature_root is None:
        return "relative-unresolved", None
    return "feature-root", feature_root / path


def percentage(numerator: int, denominator: int) -> str:
    return "N/A" if denominator == 0 else f"{100.0 * numerator / denominator:.2f}%"


def tensor_from_loaded(value: Any) -> tuple[torch.Tensor | None, str]:
    if isinstance(value, torch.Tensor):
        return value, "torch.Tensor"
    # The repository loader ultimately calls torch.FloatTensor(gvp_feat), so an
    # array-like top-level object may be representable.  Dict keys are not guessed.
    if isinstance(value, dict):
        return None, "dict: no source-confirmed feature key"
    try:
        return torch.as_tensor(value), type(value).__name__
    except Exception as exc:  # audit output should record, not hide, incompatibility
        return None, f"{type(value).__name__}: conversion failed ({exc})"


def canonical_feature(tensor: torch.Tensor) -> tuple[torch.Tensor | None, str]:
    if tensor.ndim == 2:
        return tensor.unsqueeze(0), "accepted [L,D]; added batch dimension"
    if tensor.ndim == 3 and tensor.shape[0] == 1:
        return tensor, "accepted [1,L,D]"
    return None, f"incompatible shape {tuple(tensor.shape)}"


def sequence_lengths(row: FeatureRow) -> tuple[int | None, int | None]:
    aa = nonempty_string(row.aa_seq)
    struct = nonempty_string(row.struct_seq)
    aa_len = len(aa) if aa is not None else None
    struct_len = None
    if struct is not None:
        struct_len = len([token for token in struct.split(",") if token.strip()])
    return aa_len, struct_len


def derive_expected_dimension(tokenizer: Any) -> tuple[int, str]:
    pre_quant = tokenizer.pre_quant
    for module in pre_quant.modules():
        if isinstance(module, torch.nn.LayerNorm):
            shape = module.normalized_shape
            dimension = int(shape[-1] if isinstance(shape, (tuple, list)) else shape)
            return dimension, "pre_quant first LayerNorm.normalized_shape"
    dimension = int(tokenizer.encoder.embed_dim)
    return dimension, "encoder.embed_dim fallback"


def print_counts(label: str, counts: Counter[str]) -> None:
    rendered = ", ".join(f"{key}={value}" for key, value in sorted(counts.items()))
    print(f"{label}: {rendered or 'none'}")


def main() -> int:
    args = parse_args()
    data_dir = Path(args.data_dir)
    feature_root = Path(args.feature_root) if args.feature_root is not None else None
    parquet_files = sorted(data_dir.rglob("*.parquet")) if data_dir.is_dir() else []

    section("PARQUET")
    print(f"data_dir: {data_dir}")
    print(f"files discovered: {len(parquet_files)}")
    if not parquet_files:
        print("ERROR: no parquet files discovered")
        return 1

    schemas: dict[Path, list[str]] = {}
    for path in parquet_files:
        parquet = pq.ParquetFile(path)
        columns = parquet.schema_arrow.names
        schemas[path] = columns
        print(f"{path}: rows={parquet.metadata.num_rows}, columns={columns}")
        missing = [column for column in REQUIRED_COLUMNS if column not in columns]
        if missing:
            print(f"ERROR: required columns missing from {path}: {missing}")
            return 1

    stats: Counter[str] = Counter()
    by_split: dict[str, Counter[str]] = defaultdict(Counter)
    prefixes: Counter[str] = Counter()
    raw_counts: Counter[str] = Counter()
    examples: list[tuple[str, str, str | None]] = []
    existing_rows: list[FeatureRow] = []
    global_row = 0

    for parquet_path in parquet_files:
        split = split_name(parquet_path)
        parquet = pq.ParquetFile(parquet_path)
        stop = False
        local_index = 0
        for batch in parquet.iter_batches(columns=list(REQUIRED_COLUMNS), batch_size=4096):
            values = batch.to_pydict()
            for offset in range(batch.num_rows):
                if args.max_rows is not None and global_row >= args.max_rows:
                    stop = True
                    break
                stats["total"] += 1
                by_split[split]["total"] += 1
                raw = nonempty_string(values["gvp_feat_path"][offset])
                if raw is None:
                    stats["null_or_empty"] += 1
                    by_split[split]["null_or_empty"] += 1
                else:
                    stats["valid_path"] += 1
                    by_split[split]["valid_path"] += 1
                    raw_counts[raw] += 1
                    prefixes[first_component(raw)] += 1
                    path_kind = "absolute" if Path(raw).is_absolute() else "relative"
                    stats[path_kind] += 1
                    by_split[split][path_kind] += 1
                    rule, resolved = resolve_feature(raw, feature_root)
                    if resolved is None:
                        resolution = "unresolved"
                    elif resolved.is_file():
                        resolution = "existing"
                        existing_rows.append(
                            FeatureRow(
                                split=split,
                                parquet=parquet_path,
                                row_index=local_index,
                                raw_path=raw,
                                resolved_path=resolved,
                                processed_path=values["processed_path"][offset],
                                struct_seq=values["struct_seq"][offset],
                                aa_seq=values["aa_seq"][offset],
                            )
                        )
                    else:
                        resolution = "missing"
                    stats[resolution] += 1
                    by_split[split][resolution] += 1
                    if len(examples) < 10:
                        examples.append((raw, rule, str(resolved) if resolved is not None else None))
                global_row += 1
                local_index += 1
            if stop:
                break
        if stop:
            break

    duplicate_rows = sum(count - 1 for count in raw_counts.values() if count > 1)
    duplicate_paths = sum(1 for count in raw_counts.values() if count > 1)

    section("GVP PATH")
    print(f"rows audited: {stats['total']}")
    print(f"valid paths: {stats['valid_path']}; null/empty: {stats['null_or_empty']}")
    print(f"absolute: {stats['absolute']}; relative: {stats['relative']}")
    print(f"duplicate distinct paths: {duplicate_paths}; duplicate row occurrences: {duplicate_rows}")
    print_counts("first-component counts", prefixes)
    for raw, rule, resolved in examples:
        print(f"example: raw={raw!r}, rule={rule}, resolved={resolved or 'UNRESOLVED'}")

    section("RESOLUTION")
    print(f"feature_root: {feature_root if feature_root is not None else 'None'}")
    print("repository resolution source: PdbDataset uses csv_row['gvp_feat_path'].format(data_dir=dataset_cfg.data_dir)")
    print("audit rule: absolute paths are direct; relative paths require explicit --feature_root")
    for key in ("existing", "missing", "unresolved"):
        print(f"{key}: {stats[key]} ({percentage(stats[key], stats['valid_path'])})")
    for split in sorted(by_split):
        counts = by_split[split]
        print(
            f"split={split}: total={counts['total']}, valid={counts['valid_path']}, "
            f"existing={counts['existing']}, missing={counts['missing']}, unresolved={counts['unresolved']}"
        )

    rng = random.Random(args.seed)
    probe_rows = list(existing_rows)
    rng.shuffle(probe_rows)
    probe_rows = probe_rows[: max(args.num_probe_files, 0)]
    probe_results: list[tuple[FeatureRow, torch.Tensor | None, str]] = []

    section("GVP FILE PROBE")
    print("source-confirmed loader: torch.load(gvp_path, map_location='cpu')")
    if not probe_rows:
        print("NOT TESTED: no resolved existing feature files")
    for row in probe_rows:
        try:
            loaded = torch.load(row.resolved_path, map_location="cpu")
            keys = sorted(map(str, loaded.keys())) if isinstance(loaded, dict) else None
            tensor, interpretation = tensor_from_loaded(loaded)
            print(f"file: {row.resolved_path}")
            print(f"  loaded type: {type(loaded).__name__}; dict keys: {keys if keys is not None else 'N/A'}")
            if tensor is None:
                print(f"  interpretation: {interpretation}; shape/dtype/finite: NOT TESTED")
                probe_results.append((row, None, interpretation))
                continue
            finite = bool(torch.isfinite(tensor).all()) if torch.is_floating_point(tensor) else "NOT FLOAT"
            canonical, shape_note = canonical_feature(tensor)
            print(f"  shape={tuple(tensor.shape)}, dtype={tensor.dtype}, finite={finite}")
            print(f"  interpretation={interpretation}; compatibility={shape_note}")
            probe_results.append((row, canonical, shape_note))
        except Exception as exc:
            print(f"file: {row.resolved_path}; LOAD FAILED: {type(exc).__name__}: {exc}")
            probe_results.append((row, None, "load failed"))

    section("MASK / LENGTH")
    print("source-confirmed mask: StructTokenizer.encode receives coordinate-derived residue mask; pre_quant output is multiplied by it")
    print("parquet columns audited do not supply that coordinate-derived mask")
    if not probe_results:
        print("NOT TESTED: no feature probes")
    for row, feature, _ in probe_results:
        aa_len, struct_len = sequence_lengths(row)
        feature_len = int(feature.shape[1]) if feature is not None else None
        print(
            f"{row.resolved_path}: feature_L={feature_len}, aa_seq_L={aa_len}, "
            f"struct_seq_L={struct_len}, exact valid-residue mask=NOT CONFIRMED"
        )

    tokenizer = None
    expected_dim = None
    dimension_source = "NOT TESTED"
    model_error = None
    if any(feature is not None for _, feature, _ in probe_results):
        try:
            from byprot.models.utils import get_struct_tokenizer

            tokenizer = get_struct_tokenizer().to(args.device).eval()
            for parameter in tokenizer.parameters():
                parameter.requires_grad_(False)
            expected_dim, dimension_source = derive_expected_dimension(tokenizer)
        except Exception as exc:
            model_error = f"{type(exc).__name__}: {exc}"

    shape_compatible = 0
    for _, feature, _ in probe_results:
        if feature is not None and expected_dim is not None and feature.shape[-1] == expected_dim:
            shape_compatible += 1

    section("LATENT PIPELINE")
    print(f"runtime expected pre_quant input dimension: {expected_dim if expected_dim is not None else 'NOT TESTED'}")
    print(f"dimension source: {dimension_source}")
    if model_error:
        print(f"tokenizer load failed: {model_error}")
    print(f"shape-compatible probes: {shape_compatible}/{len(probe_results)}")
    print("stored GVP -> pre_quant -> z_cont: NOT TESTED unless an authoritative coordinate-derived mask is available")
    print("stored GVP -> LFQ -> z_quant -> residual: NOT TESTED unless the same authoritative mask is available")

    section("SEMANTIC VALIDATION")
    print("source-confirmed meaning: VQModel.encode uses gvp_feat directly in place of encoder output before pre_quant")
    print("stored-file generation provenance and numerical equivalence to the current frozen encoder: NOT CONFIRMED")
    print("valid CAMEO stored-vs-online comparison: NOT TESTED (no authoritative direct feature/coordinate pairing was established)")

    section("STRUCT TOKEN CROSS-CHECK")
    print("NOT TESTED: latent pipeline did not run with an authoritative residue mask")
    print("Whether parquet struct_seq is the exact raw LFQ-ID representation for these stored files: NOT CONFIRMED")

    all_probe_shapes_ok = bool(probe_results) and shape_compatible == len(probe_results)
    if stats["existing"] == 0:
        status = "NOT_AVAILABLE"
        reasons = [
            "No gvp_feat_path resolved to an existing file under the explicit audit rules.",
            "Stored feature shape, dtype, finiteness, and semantics could not be probed.",
            "The masked pre_quant/LFQ/residual pipeline was not run.",
        ]
    elif not all_probe_shapes_ok or expected_dim is None:
        status = "PARTIALLY_AVAILABLE"
        reasons = [
            "At least one referenced feature file exists, but probe/runtime compatibility is incomplete.",
            "Exact coordinate-derived residue masks were not established.",
            "Numerical equivalence to the current frozen encoder remains NOT CONFIRMED.",
        ]
    else:
        # Shape compatibility alone cannot establish semantic readiness.
        status = "PARTIALLY_AVAILABLE"
        reasons = [
            "Resolved probe files are shape-compatible with the runtime pre_quant input.",
            "Exact coordinate-derived residue masks were not established.",
            "Stored-file provenance and numerical encoder equivalence remain NOT CONFIRMED.",
            "Do not proceed to training until semantic equivalence is demonstrated.",
        ]

    section("TRAINING DATA STATUS")
    print(status)
    for reason in reasons:
        print(f"- {reason}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
