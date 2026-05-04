#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from run_olympiabench_compare import ensure_dir, load_suite_records, run_cmd, to_pipeline_record, write_jsonl


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    rubrics_gen_dir = script_dir.parent

    parser = argparse.ArgumentParser(description="Run SFT-only rubrics generation on multiple benchmarks")
    parser.add_argument(
        "--suites",
        type=str,
        default="olympiadbench,mathvista,mathverse,mathvision,mmk12",
        help="Comma-separated suites",
    )
    parser.add_argument("--num_samples", type=int, default=20)
    parser.add_argument("--dataset_cache_root", type=Path, default=Path("dataset"))
    parser.add_argument("--output_dir", type=Path, default=script_dir / "outputs" / "multi_benchmark_sft_only")

    parser.add_argument("--olympiadbench_source", type=str, default=None)
    parser.add_argument("--mathvista_source", type=str, default=None)
    parser.add_argument("--mathverse_source", type=str, default=None)
    parser.add_argument("--mathvision_source", type=str, default=None)
    parser.add_argument("--mmk12_source", type=str, default=None)

    parser.add_argument("--sft_script", type=Path, default=script_dir / "eval_sft_lora_rubrics.py")
    parser.add_argument("--sft_base_model", type=str, required=True)
    parser.add_argument("--sft_adapter_path", type=str, required=True)
    parser.add_argument("--sft_system_prompt", type=Path, default=rubrics_gen_dir.parent / "sft_system_prompt.txt")
    parser.add_argument("--sft_torch_dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--sft_temperature", type=float, default=0.2)
    parser.add_argument("--sft_top_p", type=float, default=0.9)
    parser.add_argument("--sft_top_k", type=int, default=40)
    parser.add_argument("--sft_max_new_tokens", type=int, default=2048)
    parser.add_argument("--sft_seed", type=int, default=2026)
    parser.add_argument("--sft_batch_size", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.num_samples <= 0:
        raise ValueError("--num_samples must be > 0")
    if args.sft_batch_size <= 0:
        raise ValueError("--sft_batch_size must be > 0")
    if not args.sft_script.exists():
        raise FileNotFoundError(f"SFT script not found: {args.sft_script}")
    if not args.sft_system_prompt.exists():
        raise FileNotFoundError(f"System prompt not found: {args.sft_system_prompt}")

    suite_list = [x.strip().lower() for x in args.suites.split(",") if x.strip()]
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
        print(f"\n===== Running SFT suite: {suite} =====", flush=True)
        suite_dir = args.output_dir / suite
        image_dir = suite_dir / "images"
        ensure_dir(suite_dir)
        ensure_dir(image_dir)

        raw_items = load_suite_records(
            suite_name=suite,
            cache_root=args.dataset_cache_root,
            num_samples=args.num_samples,
            source_override=source_overrides[suite],
        )

        rows = []
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

        first_n_file = suite_dir / "first20.jsonl"
        write_jsonl(first_n_file, rows)
        print(f"[{suite}] prepared {len(rows)} samples -> {first_n_file}", flush=True)

        sft_out = suite_dir / "sft" / "rubrics_sft.jsonl"
        sft_cmd = [
            sys.executable,
            str(args.sft_script),
            "--input_file",
            str(first_n_file),
            "--adapter_path",
            args.sft_adapter_path,
            "--base_model",
            args.sft_base_model,
            "--system_prompt",
            str(args.sft_system_prompt),
            "--output_file",
            str(sft_out),
            "--num_samples",
            str(len(rows)),
            "--temperature",
            str(args.sft_temperature),
            "--top_p",
            str(args.sft_top_p),
            "--top_k",
            str(args.sft_top_k),
            "--max_new_tokens",
            str(args.sft_max_new_tokens),
            "--seed",
            str(args.sft_seed),
            "--torch_dtype",
            args.sft_torch_dtype,
            "--batch_size",
            str(args.sft_batch_size),
        ]
        run_cmd(sft_cmd)

        summary[suite] = {
            "prepared_samples": len(rows),
            "prepared_file": str(first_n_file),
            "sft_output": str(sft_out),
        }

    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[done] summary -> {summary_path}", flush=True)


if __name__ == "__main__":
    main()

