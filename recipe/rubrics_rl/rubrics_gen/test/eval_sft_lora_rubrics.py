#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import random
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
from PIL import Image
from peft import PeftModel
from transformers import AutoModelForImageTextToText, AutoProcessor


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def get_question(sample: dict[str, Any]) -> str:
    question = sample.get("question")
    if question:
        return str(question)
    extra = sample.get("extra_info")
    if isinstance(extra, dict):
        return str(extra.get("prompt") or extra.get("question") or "")
    return ""


def get_image_paths(sample: dict[str, Any], image_base_path: Optional[Path]) -> list[Path]:
    image_value = sample.get("images")
    if image_value is None:
        image_value = sample.get("image")

    if image_value is None:
        return []

    if isinstance(image_value, str):
        raw_paths = [image_value]
    elif isinstance(image_value, list):
        raw_paths = [str(x) for x in image_value]
    else:
        return []

    paths: list[Path] = []
    for p in raw_paths:
        path = Path(p)
        if image_base_path is not None and not path.is_absolute():
            path = image_base_path / path
        paths.append(path)
    return paths


def build_messages(system_prompt: str, question: str, num_images: int) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = []
    if system_prompt.strip():
        messages.append({"role": "system", "content": system_prompt.strip()})

    content: list[dict[str, Any]] = []
    for _ in range(num_images):
        content.append({"type": "image"})
    content.append({"type": "text", "text": question})
    messages.append({"role": "user", "content": content})
    return messages


def extract_json_block(text: str) -> Optional[str]:
    if not text:
        return None
    if "```json" in text:
        return text.split("```json", 1)[1].split("```", 1)[0].strip()
    if "```" in text:
        return text.split("```", 1)[1].split("```", 1)[0].strip()
    return text.strip()


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
    parsed: list[dict[str, Any]] = []
    for row in data:
        if not isinstance(row, dict):
            return None
        if "criterion" not in row or "weight" not in row:
            return None
        parsed.append({"criterion": row["criterion"], "weight": row["weight"]})
    return parsed


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_adapter_source(adapter_path: str) -> str:
    local_path = Path(adapter_path).expanduser()
    if local_path.exists():
        return str(local_path)
    return adapter_path


def load_model(base_model: str, adapter_path: str, torch_dtype: str):
    dtype_map = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }
    dtype = dtype_map[torch_dtype]

    processor = AutoProcessor.from_pretrained(base_model, trust_remote_code=True)
    base = AutoModelForImageTextToText.from_pretrained(
        base_model,
        torch_dtype=dtype,
        device_map="auto",
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    adapter_source = resolve_adapter_source(adapter_path)
    model = PeftModel.from_pretrained(base, adapter_source)
    model.eval()
    return model, processor, adapter_source


def generate_one(
    model,
    processor,
    question: str,
    image_paths: list[Path],
    system_prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
) -> str:
    images = [Image.open(p).convert("RGB") for p in image_paths]
    messages = build_messages(system_prompt=system_prompt, question=question, num_images=len(images))
    chat_text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    inputs = processor(text=[chat_text], images=images if images else None, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    do_sample = temperature > 0
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature if do_sample else None,
            top_p=top_p if do_sample else None,
            top_k=top_k if do_sample else None,
        )

    input_len = inputs["input_ids"].shape[1]
    gen_only = generated_ids[:, input_len:]
    output = processor.batch_decode(gen_only, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    return output.strip()


def generate_batch(
    model,
    processor,
    batch_items: list[dict[str, Any]],
    system_prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
) -> list[str]:
    chat_texts: list[str] = []
    images_batch: list[list[Image.Image]] = []
    for item in batch_items:
        messages = build_messages(
            system_prompt=system_prompt,
            question=item["question"],
            num_images=len(item["image_paths"]),
        )
        chat_texts.append(processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))
        images_batch.append([Image.open(p).convert("RGB") for p in item["image_paths"]])

    inputs = processor(
        text=chat_texts,
        images=images_batch if any(images_batch) else None,
        return_tensors="pt",
        padding=True,
    )
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    do_sample = temperature > 0
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature if do_sample else None,
            top_p=top_p if do_sample else None,
            top_k=top_k if do_sample else None,
        )

    input_lens = inputs["attention_mask"].sum(dim=1).tolist()
    outputs: list[str] = []
    for idx, input_len in enumerate(input_lens):
        gen_only = generated_ids[idx, int(input_len):]
        text = processor.decode(gen_only, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        outputs.append(text.strip())
    return outputs


def dump_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Qwen2.5-VL SFT LoRA adapter on rubrics test data")
    parser.add_argument("--input_file", type=Path, required=True, help="rubrics-test jsonl path")
    parser.add_argument(
        "--adapter_path",
        type=str,
        required=True,
        help="LoRA/PEFT adapter local directory or HuggingFace repo id",
    )
    parser.add_argument("--base_model", type=str, default="Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument(
        "--system_prompt",
        type=Path,
        default=Path("/Users/caroli/Documents/verl-exp/recipe/rubrics_rl/sft_system_prompt.txt"),
    )
    parser.add_argument("--image_base_path", type=Path, default=None)
    parser.add_argument("--output_file", type=Path, default=Path("test/outputs/sft_rubrics_eval.jsonl"))
    parser.add_argument("--num_samples", type=int, default=10)

    parser.add_argument("--max_new_tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--top_k", type=int, default=40)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--torch_dtype", type=str, choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--batch_size", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.input_file.exists():
        raise FileNotFoundError(f"Input file not found: {args.input_file}")
    if not args.system_prompt.exists():
        raise FileNotFoundError(f"System prompt file not found: {args.system_prompt}")
    if args.num_samples <= 0:
        raise ValueError("--num_samples must be > 0")
    if args.batch_size <= 0:
        raise ValueError("--batch_size must be > 0")

    set_seed(args.seed)
    system_prompt_text = args.system_prompt.read_text(encoding="utf-8").strip()

    rows = read_jsonl(args.input_file)
    rows = rows[: args.num_samples]

    model, processor, adapter_source = load_model(args.base_model, args.adapter_path, args.torch_dtype)

    prepared: list[dict[str, Any]] = []
    outputs: list[dict[str, Any]] = []

    for idx, sample in enumerate(rows, start=1):
        qid = str(sample.get("id", idx))
        question = get_question(sample)
        image_paths = get_image_paths(sample, args.image_base_path)
        missing = [str(p) for p in image_paths if not p.exists()]

        if not question:
            outputs.append(
                {
                    "id": qid,
                    "status": "failure",
                    "error": "empty question",
                }
            )
            continue

        if missing:
            outputs.append(
                {
                    "id": qid,
                    "question": question,
                    "status": "failure",
                    "error": "missing images",
                    "missing_images": missing,
                }
            )
            continue

        prepared.append(
            {
                "id": qid,
                "question": question,
                "image_paths": image_paths,
            }
        )

    success = 0
    for start in range(0, len(prepared), args.batch_size):
        batch_items = prepared[start:start + args.batch_size]
        texts = generate_batch(
            model=model,
            processor=processor,
            batch_items=batch_items,
            system_prompt=system_prompt_text,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
        )
        for item, text in zip(batch_items, texts):
            rubrics = parse_rubrics(text)
            status = "success" if rubrics is not None else "failure"
            if status == "success":
                success += 1
            outputs.append(
                {
                    "id": item["id"],
                    "question": item["question"],
                    "image": [str(p) for p in item["image_paths"]],
                    "status": status,
                    "rubrics": rubrics,
                    "model_output": text,
                    "base_model": args.base_model,
                    "adapter_path": adapter_source,
                }
            )

    dump_jsonl(args.output_file, outputs)
    print(f"total={len(outputs)}")
    print(f"success={success}")
    print(f"failed={len(outputs) - success}")
    print(f"output_file={args.output_file}")


if __name__ == "__main__":
    main()
