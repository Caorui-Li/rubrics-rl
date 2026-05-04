#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from run_olympiabench_compare import DATASET_SPECS, ensure_dir, load_suite_records, run_cmd, to_pipeline_record, write_jsonl


def count_jsonl_rows(path: Path) -> int:
    if not path.exists():
        return 0
    count = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                count += 1
    return count


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    rubrics_gen_dir = script_dir.parent

    parser = argparse.ArgumentParser(
        description="Compare pipeline rubrics generation across 5 benchmarks on sampled data"
    )
    parser.add_argument(
        "--suites",
        type=str,
        default="olympiadbench,mathvista,mathverse,mathvision,mmk12",
        help="Comma-separated suites from: olympiadbench,mathvista,mathverse,mathvision,mmk12",
    )
    parser.add_argument("--num_samples", type=int, default=20)
    parser.add_argument("--dataset_cache_root", type=Path, default=Path("dataset"))
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=script_dir / "outputs" / "multi_benchmark_pipeline_compare",
    )

    parser.add_argument("--olympiadbench_source", type=str, default=None)
    parser.add_argument("--mathvista_source", type=str, default=None)
    parser.add_argument("--mathverse_source", type=str, default=None)
    parser.add_argument("--mathvision_source", type=str, default=None)
    parser.add_argument("--mmk12_source", type=str, default=None)

    parser.add_argument("--pipeline_script", type=Path, default=rubrics_gen_dir / "pipeline.py")
    parser.add_argument("--pipeline_gpu_memory_utilization", type=float, default=0.6)
    parser.add_argument("--pipeline_max_model_len", type=int, default=None)
    parser.add_argument("--pipeline_max_num_seqs", type=int, default=64)

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.num_samples <= 0:
        raise ValueError("--num_samples must be > 0")
    if not args.pipeline_script.exists():
        raise FileNotFoundError(f"Pipeline script not found: {args.pipeline_script}")

    suite_list = [x.strip().lower() for x in args.suites.split(",") if x.strip()]
    for suite in suite_list:
        if suite not in DATASET_SPECS:
            raise ValueError(f"Unknown suite: {suite}")

    source_overrides = {
        "olympiadbench": args.olympiadbench_source or str((args.dataset_cache_root / "OlympiadBench").resolve()),
        "mathvista": args.mathvista_source or str((args.dataset_cache_root / "MathVista").resolve()),
        "mathverse": args.mathverse_source or str((args.dataset_cache_root / "MathVerse").resolve()),
        "mathvision": args.mathvision_source or str((args.dataset_cache_root / "MathVision").resolve()),
        "mmk12": args.mmk12_source or str((args.dataset_cache_root / "MMK12").resolve()),
    }

    ensure_dir(args.output_dir)
    summary: dict[str, Any] = {}

    for suite in suite_list:
        print(f"\n===== Running suite: {suite} =====", flush=True)
        suite_dir = args.output_dir / suite
        image_dir = suite_dir / "images"
        pipeline_dir = suite_dir / "pipeline"
        ensure_dir(suite_dir)
        ensure_dir(image_dir)

        raw_items = load_suite_records(
            suite_name=suite,
            cache_root=args.dataset_cache_root,
            num_samples=args.num_samples,
            source_override=source_overrides[suite],
        )

        rows: list[dict[str, Any]] = []
        for i, item in enumerate(raw_items, start=1):
            try:
                row = to_pipeline_record(suite, item, i, image_dir)
            except Exception as e:
                print(f"[warn] skip sample {i} in {suite}: {e}", flush=True)
                continue
            if not row.get("question"):
                continue
            rows.append(row)
            if len(rows) >= args.num_samples:
                break

        sample_file = suite_dir / f"first{args.num_samples}.jsonl"
        write_jsonl(sample_file, rows)
        print(f"[{suite}] prepared {len(rows)} samples -> {sample_file}", flush=True)

        pipeline_cmd = [
            sys.executable,
            str(args.pipeline_script),
            "--input_file",
            str(sample_file),
            "--cot_raw_out",
            str(pipeline_dir / "cot_raw.jsonl"),
            "--cot_filtered_out",
            str(pipeline_dir / "cot_filtered.jsonl"),
            "--filter_raw_out",
            str(pipeline_dir / "filter_raw.jsonl"),
            "--filter_filtered_out",
            str(pipeline_dir / "filter_filtered.jsonl"),
            "--rubrics_raw_out",
            str(pipeline_dir / "rubrics_raw.jsonl"),
            "--rubrics_filtered_out",
            str(pipeline_dir / "rubrics_filtered.jsonl"),
            "--gpu_memory_utilization",
            str(args.pipeline_gpu_memory_utilization),
            "--max_num_seqs",
            str(args.pipeline_max_num_seqs),
            "--image_base_path",
            str(image_dir),
        ]
        if args.pipeline_max_model_len is not None:
            pipeline_cmd.extend(["--max_model_len", str(args.pipeline_max_model_len)])

        run_cmd(pipeline_cmd)

        cot_raw = pipeline_dir / "cot_raw.jsonl"
        filter_raw = pipeline_dir / "filter_raw.jsonl"
        filter_filtered = pipeline_dir / "filter_filtered.jsonl"
        rubrics_raw = pipeline_dir / "rubrics_raw.jsonl"
        rubrics_filtered = pipeline_dir / "rubrics_filtered.jsonl"

        stats = {
            "prepared_samples": len(rows),
            "cot_raw": count_jsonl_rows(cot_raw),
            "filter_raw": count_jsonl_rows(filter_raw),
            "filter_filtered": count_jsonl_rows(filter_filtered),
            "rubrics_raw": count_jsonl_rows(rubrics_raw),
            "rubrics_filtered": count_jsonl_rows(rubrics_filtered),
        }

        summary[suite] = {
            "sample_file": str(sample_file),
            "pipeline_output_dir": str(pipeline_dir),
            "rubrics_filtered_file": str(rubrics_filtered),
            "stats": stats,
        }

    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n===== Compare Summary =====", flush=True)
    print(
        "suite\tprepared\tcot_raw\tfilter_in\tfilter_pass\trubrics_raw\trubrics_ok",
        flush=True,
    )
    for suite in suite_list:
        stats = summary[suite]["stats"]
        print(
            f"{suite}\t{stats['prepared_samples']}\t{stats['cot_raw']}\t{stats['filter_raw']}\t{stats['filter_filtered']}\t{stats['rubrics_raw']}\t{stats['rubrics_filtered']}",
            flush=True,
        )

    print(f"\n[done] summary -> {summary_path}", flush=True)


if __name__ == "__main__":
    main()
