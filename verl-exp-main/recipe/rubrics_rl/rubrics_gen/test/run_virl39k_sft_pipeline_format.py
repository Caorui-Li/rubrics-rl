#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import Any, Optional

import torch
from datasets import load_dataset
from transformers import AutoProcessor
from vllm import LLM, SamplingParams

SCRIPT_DIR = Path(__file__).resolve().parent
RUBRICS_GEN_DIR = SCRIPT_DIR.parent
if str(RUBRICS_GEN_DIR) not in sys.path:
    sys.path.append(str(RUBRICS_GEN_DIR))

from core import messages_to_vllm_input, unload_destroy


def load_data(input_file: Path) -> list[dict[str, Any]]:
    suffix = input_file.suffix.lower()
    if suffix == ".jsonl":
        rows: list[dict[str, Any]] = []
        with input_file.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows
    if suffix == ".json":
        data = json.loads(input_file.read_text(encoding="utf-8"))
        if isinstance(data, list):
            return [dict(x) for x in data]
        if isinstance(data, dict):
            return [dict(data)]
        raise ValueError(f"Unsupported JSON structure in {input_file}")
    if suffix == ".parquet":
        ds = load_dataset("parquet", data_files=str(input_file))["train"]
        return [dict(x) for x in ds]
    raise ValueError(f"Unsupported input format: {input_file}")


def dump_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def get_question(sample: dict[str, Any]) -> str:
    if sample.get("question"):
        return str(sample["question"])
    extra = sample.get("extra_info")
    if isinstance(extra, dict):
        return str(extra.get("prompt") or extra.get("question") or "")
    return ""


def get_answer(sample: dict[str, Any]) -> str:
    if sample.get("answer"):
        return str(sample["answer"])
    extra = sample.get("extra_info")
    if isinstance(extra, dict):
        return str(extra.get("answer") or "")
    return ""


def get_image_paths(sample: dict[str, Any], image_base_path: Optional[Path]) -> list[str]:
    image_value = sample.get("image")
    if image_value is None:
        image_value = sample.get("images")
    if image_value is None:
        return []

    if isinstance(image_value, str):
        raw_paths = [image_value]
    elif isinstance(image_value, list):
        raw_paths = [str(x) for x in image_value]
    else:
        return []

    out: list[str] = []
    for p in raw_paths:
        path = Path(p)
        if image_base_path is not None and not path.is_absolute():
            path = image_base_path / path
        out.append(str(path))
    return out


def build_messages(system_prompt: str, question: str, image_paths: list[str]) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = []
    if system_prompt.strip():
        messages.append({"role": "system", "content": system_prompt.strip()})

    content: list[dict[str, Any]] = []
    for img in image_paths:
        content.append({"type": "image", "image": img})
    content.append({"type": "text", "text": question})
    messages.append({"role": "user", "content": content})
    return messages


def extract_json_block(text: str) -> Optional[str]:
    if not text:
        return None
    s = text.strip()
    if "</think>" in s:
        s = s.split("</think>", 1)[1].strip()

    # Prefer fenced JSON block.
    if "```json" in s:
        return s.split("```json", 1)[1].split("```", 1)[0].strip()
    if "```" in s:
        return s.split("```", 1)[1].split("```", 1)[0].strip()

    # If there is extra prose, extract the first JSON array/object segment.
    for left, right in (("[", "]"), ("{", "}")):
        l = s.find(left)
        r = s.rfind(right)
        if l != -1 and r != -1 and r > l:
            return s[l : r + 1].strip()

    return s


def parse_rubrics(text: str) -> Optional[list[dict[str, Any]]]:
    block = extract_json_block(text)
    if not block:
        return None
    try:
        data = json.loads(block)
    except json.JSONDecodeError:
        return None

    if not isinstance(data, list):
        return None

    normalized: list[dict[str, Any]] = []
    for item in data:
        if not isinstance(item, dict):
            return None
        if "criterion" not in item or "weight" not in item:
            return None
        row = dict(item)
        normalized.append(row)

    for idx, row in enumerate(normalized, start=1):
        row["id"] = idx

    return normalized


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_llm(
    model_path: str,
    gpu_memory_utilization: float,
    max_model_len: Optional[int],
    max_num_seqs: int,
    tensor_parallel_size: Optional[int],
) -> LLM:
    tp = tensor_parallel_size or (torch.cuda.device_count() if torch.cuda.is_available() else 1)
    kwargs: dict[str, Any] = {
        "model": model_path,
        "tensor_parallel_size": tp,
        "dtype": "auto",
        "trust_remote_code": True,
        "gpu_memory_utilization": gpu_memory_utilization,
        "max_num_seqs": max_num_seqs,
        "seed": 0,
    }
    lower = model_path.lower()
    if "vl" in model_path or "vision" in lower:
        kwargs["mm_encoder_tp_mode"] = "data"
    if max_model_len is not None:
        kwargs["max_model_len"] = max_model_len
    return LLM(**kwargs)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run vLLM rubrics generation for full VIRL39K and export pipeline-style files")
    parser.add_argument("--input_file", type=Path, required=True, help="VIRL39K file: parquet/json/jsonl")
    parser.add_argument("--model_path", type=str, required=True, help="SFT merged model path or HF repo")
    parser.add_argument("--system_prompt", type=Path, default=Path("/Users/caroli/Documents/verl-exp/recipe/rubrics_rl/sft_system_prompt.txt"))
    parser.add_argument("--image_base_path", type=Path, default=None)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--num_samples", type=int, default=0, help="0 means full dataset")

    parser.add_argument("--max_new_tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--top_k", type=int, default=40)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--batch_size", type=int, default=16)

    parser.add_argument("--gpu_memory_utilization", type=float, default=0.6)
    parser.add_argument("--max_model_len", type=int, default=None)
    parser.add_argument("--max_num_seqs", type=int, default=64)
    parser.add_argument("--tensor_parallel_size", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.input_file.exists():
        raise FileNotFoundError(f"Input file not found: {args.input_file}")
    if not args.system_prompt.exists():
        raise FileNotFoundError(f"System prompt not found: {args.system_prompt}")
    if args.batch_size <= 0:
        raise ValueError("--batch_size must be > 0")
    if args.num_samples < 0:
        raise ValueError("--num_samples must be >= 0")

    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    if "vl" in args.model_path.lower() or "vision" in args.model_path.lower():
        os.environ.setdefault("VLLM_USE_V1", "1")

    set_seed(args.seed)
    rows = load_data(args.input_file)
    if args.num_samples > 0:
        rows = rows[: args.num_samples]

    system_prompt_text = args.system_prompt.read_text(encoding="utf-8").strip()
    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)

    raw_results: list[dict[str, Any]] = []
    filtered_results: list[dict[str, Any]] = []
    prepared: list[dict[str, Any]] = []

    for i, sample in enumerate(rows, start=1):
        item = dict(sample)
        qid = str(item.get("id", i))
        question = get_question(item)
        answer = get_answer(item)
        image_paths = get_image_paths(item, args.image_base_path)

        item["id"] = qid
        item["question"] = question
        if answer:
            item["answer"] = answer
        item["image"] = image_paths

        if not question:
            item["rubrics"] = None
            item["status"] = "failure"
            item["model_output"] = None
            raw_results.append(item)
            continue

        missing = [p for p in image_paths if not Path(p).exists()]
        if missing:
            item["rubrics"] = None
            item["status"] = "failure"
            item["missing_images"] = missing
            item["model_output"] = None
            raw_results.append(item)
            continue

        messages = build_messages(system_prompt_text, question, image_paths)
        vllm_input = messages_to_vllm_input(messages, processor)
        prepared.append({"base": item, "vllm_input": vllm_input})

    llm = build_llm(
        model_path=args.model_path,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
        tensor_parallel_size=args.tensor_parallel_size,
    )
    sampling = SamplingParams(
        temperature=args.temperature,
        max_tokens=args.max_new_tokens,
        top_p=args.top_p,
        top_k=args.top_k,
    )

    success = 0
    for start in range(0, len(prepared), args.batch_size):
        batch = prepared[start:start + args.batch_size]
        inputs = [x["vllm_input"] for x in batch]
        outputs = llm.generate(inputs, sampling_params=sampling)
        texts = [o.outputs[0].text for o in outputs]

        for item, text in zip(batch, texts):
            out = dict(item["base"])
            out["model_output"] = text
            rubrics = parse_rubrics(text)
            if rubrics is None:
                out["rubrics"] = None
                out["status"] = "failure"
            else:
                out["rubrics"] = json.dumps(rubrics, ensure_ascii=False)
                out["status"] = "success"
                success += 1
                filtered_results.append(dict(out))
            raw_results.append(out)

    unload_destroy(llm)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = args.output_dir / "rubrics_raw.jsonl"
    filtered_path = args.output_dir / "rubrics_filtered.jsonl"
    dump_jsonl(raw_path, raw_results)
    dump_jsonl(filtered_path, filtered_results)

    print(f"total={len(raw_results)}")
    print(f"success={success}")
    print(f"failed={len(raw_results) - success}")
    print(f"rubrics_raw={raw_path}")
    print(f"rubrics_filtered={filtered_path}")


if __name__ == "__main__":
    main()
