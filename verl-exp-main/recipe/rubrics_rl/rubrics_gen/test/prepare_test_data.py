#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
from pathlib import Path


def extract_first_n(input_path: Path, output_path: Path, num: int) -> int:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with input_path.open("r", encoding="utf-8") as fin, output_path.open("w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
            count += 1
            if count >= num:
                break
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract first N lines from rubrics_filtered.jsonl for model comparison")
    parser.add_argument("--input", type=Path, required=True, help="Input rubrics_filtered.jsonl")
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent / "test.jsonl")
    parser.add_argument("--num", type=int, default=10)
    args = parser.parse_args()

    if not args.input.exists():
        raise FileNotFoundError(f"Input not found: {args.input}")
    if args.num <= 0:
        raise ValueError("--num must be > 0")

    count = extract_first_n(args.input, args.output, args.num)
    print(f"Extracted {count} samples to {args.output}")


if __name__ == "__main__":
    main()
