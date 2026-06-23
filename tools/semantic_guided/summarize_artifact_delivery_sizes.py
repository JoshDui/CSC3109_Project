#!/usr/bin/env python3
"""Summarize raw and gzip-compressed sizes for deployment artifacts."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import gzip
import json
from pathlib import Path
import shutil
from typing import Any


DEFAULT_RUN_ID = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Write CSV/JSON delivery-size summaries for selected model artifacts.",
        allow_abbrev=False,
    )
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID)
    parser.add_argument("--artifact", action="append", required=True, help="Artifact as NAME=PATH. May be repeated.")
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--gzip", action="store_true", help="Create/update PATH.gz for each artifact before summarizing.")
    parser.add_argument("--force-gzip", action="store_true", help="Overwrite existing PATH.gz files.")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    for _name, path in parse_artifacts(args.artifact):
        if not path.exists():
            raise FileNotFoundError(f"Artifact not found: {path}")


def main() -> None:
    args = parse_args()
    validate_args(args)
    rows: list[dict[str, Any]] = []
    for name, path in parse_artifacts(args.artifact):
        gzip_path = Path(f"{path}.gz")
        if args.gzip and (args.force_gzip or not gzip_path.exists()):
            write_gzip(path, gzip_path)
        row = build_size_row(args.run_id, name, path, gzip_path if gzip_path.exists() else None)
        rows.append(row)

    add_pairwise_comparisons(rows)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    write_csv(rows, args.output_csv)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(
            {
                "run_id": args.run_id,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "rows": rows,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    print(f"Wrote delivery-size summary: {args.output_csv}", flush=True)


def parse_artifacts(raw_values: list[str]) -> list[tuple[str, Path]]:
    specs: list[tuple[str, Path]] = []
    seen: set[str] = set()
    for raw in raw_values:
        if "=" not in raw:
            raise ValueError(f"--artifact must use NAME=PATH format, got {raw!r}")
        name, path_text = raw.split("=", 1)
        name = slugify(name)
        if not name:
            raise ValueError(f"Artifact name is empty in {raw!r}")
        if name in seen:
            raise ValueError(f"Duplicate artifact name {name!r}")
        seen.add(name)
        specs.append((name, Path(path_text).expanduser()))
    return specs


def write_gzip(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with source.open("rb") as src, gzip.open(destination, "wb", compresslevel=9) as dst:
        shutil.copyfileobj(src, dst)


def build_size_row(run_id: str, name: str, path: Path, gzip_path: Path | None) -> dict[str, Any]:
    raw_bytes = path.stat().st_size
    gzip_bytes = gzip_path.stat().st_size if gzip_path is not None else None
    raw_mib = raw_bytes / (1024.0 * 1024.0)
    gzip_mib = None if gzip_bytes is None else gzip_bytes / (1024.0 * 1024.0)
    return {
        "run_id": run_id,
        "artifact": name,
        "raw_path": str(path),
        "path": str(path),
        "raw_size_bytes": raw_bytes,
        "raw_size_mib": raw_mib,
        "raw_mib": raw_mib,
        "gzip_path": None if gzip_path is None else str(gzip_path),
        "gzip_size_bytes": gzip_bytes,
        "gzip_size_mib": gzip_mib,
        "gzip_mib": gzip_mib,
        "gzip_ratio": None if gzip_bytes is None else gzip_bytes / raw_bytes,
    }


def add_pairwise_comparisons(rows: list[dict[str, Any]]) -> None:
    by_name = {str(row["artifact"]): row for row in rows}
    fp32 = by_name.get("onnx_fp32")
    int8 = by_name.get("onnx_int8_qdq_fullcalib_minmax") or by_name.get("onnx_int8_qdq")
    if not fp32 or not int8:
        return
    fp32_gzip = fp32.get("gzip_size_bytes")
    int8_gzip = int8.get("gzip_size_bytes")
    fp32_raw = fp32.get("raw_size_bytes")
    int8_raw = int8.get("raw_size_bytes")
    for row in (fp32, int8):
        row["compressed_int8_vs_fp32_reduction_factor"] = None
        row["raw_int8_vs_fp32_reduction_factor"] = None
    if fp32_gzip and int8_gzip:
        ratio = float(fp32_gzip) / float(int8_gzip)
        fp32["compressed_int8_vs_fp32_reduction_factor"] = ratio
        int8["compressed_int8_vs_fp32_reduction_factor"] = ratio
    if fp32_raw and int8_raw:
        ratio = float(fp32_raw) / float(int8_raw)
        fp32["raw_int8_vs_fp32_reduction_factor"] = ratio
        int8["raw_int8_vs_fp32_reduction_factor"] = ratio


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def slugify(value: str) -> str:
    return "".join(character if character.isalnum() else "_" for character in value.lower()).strip("_") or "item"


if __name__ == "__main__":
    main()
