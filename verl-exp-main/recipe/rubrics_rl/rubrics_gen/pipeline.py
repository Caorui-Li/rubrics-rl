# -*- coding: utf-8 -*-
"""Rubrics generation pipeline using the run_vllm core."""

import argparse
import json
import os
import re
import sys
from functools import partial
from pathlib import Path
from typing import Any, Callable, Optional

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoProcessor, AutoTokenizer
from vllm import LLM, SamplingParams

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.append(str(SCRIPT_DIR))

from core import DistributeDataset, messages_to_vllm_input, run_vllm, unload_destroy

PROMPT_DIR = SCRIPT_DIR / "prompts"

COT_FIELD = "cot"
RUBRICS_FIELD = "rubrics"

COT_MODEL = "/mnt/data/liuchonghan/Qwen3-VL-32B-Thinking"
COT_PROCESSOR = "/mnt/data/liuchonghan/Qwen3-VL-32B-Thinking"
COT_BATCH_SIZE = 512  # Reduced for vision model to avoid OOM
COT_SAMPLING = {
    "temperature": 0.0,
    "max_tokens": 16384,
    "top_k": -1,
    "stop_token_ids": [],
}

FILTER_MODEL = "/mnt/data/liuchonghan/Qwen3-30B-A3B-Instruct-2507"
FILTER_PROCESSOR = "/mnt/data/liuchonghan/Qwen3-30B-A3B-Instruct-2507"
FILTER_BATCH_SIZE = 10240
FILTER_SAMPLING = {
    "temperature": 0.1,
    "max_tokens": 512,
    "top_k": 20,
    "top_p": 0.8,
    "stop_token_ids": [],
}

RUBRICS_MODEL = "/mnt/data/liuchonghan/Qwen3-30B-A3B-Instruct-2507"
RUBRICS_PROCESSOR = "/mnt/data/liuchonghan/Qwen3-30B-A3B-Instruct-2507"
RUBRICS_BATCH_SIZE = 10240
RUBRICS_SAMPLING = {
    "temperature": 0.7,
    "max_tokens": 16384,
    "top_k": 20,
    "top_p": 0.8,
    "stop_token_ids": [],
}


def load_prompt(name: str, default: str = "") -> str:
    path = PROMPT_DIR / name
    if not path.exists():
        return default
    return path.read_text(encoding="utf-8").strip()


def load_data(data_path: str | None, data_list: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    if data_list is not None:
        data = data_list
    elif data_path is not None and data_path.endswith(".jsonl"):
        with open(data_path, "r", encoding="utf-8") as handle:
            data = [json.loads(line) for line in handle]
    elif data_path is not None and data_path.endswith(".json"):
        with open(data_path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
    elif data_path is not None and data_path.endswith(".parquet"):
        dataset = load_dataset("parquet", data_files=data_path)["train"]
        data = [dict(item) for item in dataset]
    else:
        raise ValueError("Invalid data path and data list")

    return data


def get_question(sample: dict[str, Any]) -> str:
    if sample.get("question"):
        return sample["question"]
    extra = sample.get("extra_info")
    if isinstance(extra, dict):
        return extra.get("prompt") or extra.get("question") or ""
    return ""


def get_answer(sample: dict[str, Any]) -> str:
    if sample.get("answer"):
        return sample["answer"]
    extra = sample.get("extra_info")
    if isinstance(extra, dict):
        return extra.get("answer") or ""
    return ""


def resolve_image_paths(sample: dict[str, Any], image_base_path: str | None) -> list[str]:
    image_paths = sample.get("image") or []
    if isinstance(image_paths, str):
        image_paths = [image_paths]

    resolved = []
    for path in image_paths:
        if image_base_path and not os.path.isabs(path):
            resolved.append(os.path.join(image_base_path, path))
        else:
            resolved.append(path)
    return resolved


def vllm_collate_batch(batch: list[dict[str, Any]], processor: AutoProcessor) -> list[dict[str, Any]]:
    for item in batch:
        messages = item.get("messages")
        if messages is None:
            raise ValueError("Missing messages in batch item")
        item["vllm_inputs"] = messages_to_vllm_input(messages, processor)
    return batch


def build_vl_messages(system_prompt: str, user_text: str, image_paths: list[str] | None = None) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    content: list[dict[str, Any]] = []
    if image_paths:
        for img_path in image_paths:
            content.append({"type": "image", "image": img_path})
    content.append({"type": "text", "text": user_text})

    messages.append({"role": "user", "content": content})
    return messages

def build_text_messages(system_prompt: str, user_text: str) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_text})
    return messages


def extract_answer_from_cot(cot_text: str) -> str:
    if not cot_text or not cot_text.strip():
        return ""

    answer_match = re.search(r"<answer>(.*?)</answer>", cot_text, re.DOTALL)
    if answer_match:
        return answer_match.group(1).strip()

    answer_match = re.search(r"(?i)answer\s*:\s*(.+?)(?:\n|$)", cot_text, re.DOTALL)
    if answer_match:
        return answer_match.group(1).strip()

    if "</think>" in cot_text:
        remaining = cot_text.split("</think>")[-1].strip()
        if remaining:
            return remaining.strip()

    return cot_text.strip()


def parse_judgement(judgement_text: str) -> Optional[bool]:
    judgement = judgement_text.strip().upper()
    if "INCORRECT" in judgement:
        return False
    if "CORRECT" in judgement:
        return True
    return None


def extract_json_block(text: str) -> Optional[str]:
    if not text:
        return None
    if "```json" in text:
        return text.split("```json")[1].split("```")[0].strip()
    if "```" in text:
        return text.split("```")[1].split("```")[0].strip()
    return text.strip()


def parse_rubrics(text: str) -> Optional[list[dict[str, Any]]]:
    rubrics_json = extract_json_block(text)
    if not rubrics_json:
        return None
    try:
        rubrics = json.loads(rubrics_json)
    except json.JSONDecodeError:
        return None

    if not isinstance(rubrics, list):
        return None

    normalized: list[dict[str, Any]] = []
    for item in rubrics:
        if not isinstance(item, dict):
            return None
        if "criterion" not in item or "weight" not in item:
            return None
        normalized.append(dict(item))

    for idx, rubric in enumerate(normalized, start=1):
        rubric["id"] = idx

    return normalized


class StageDataset(DistributeDataset):
    def __init__(
        self,
        data_path: str | None,
        data_list: list[dict[str, Any]] | None,
        rank: int,
        world_size: int,
        message_fn: Callable[[dict[str, Any]], Any],
    ) -> None:
        self.message_fn = message_fn
        super().__init__(rank=rank, world_size=world_size, data_path=data_path, data_list=data_list)

    def _load_data(self, data_path, data_list):
        return load_data(data_path, data_list)

    def filter_data(self, item: dict[str, Any]) -> bool:
        return self.message_fn(item) is not None

    def __getitem__(self, idx):
        item = super().__getitem__(idx)
        result = self.message_fn(item)
        if result is None:
            raise ValueError("message_fn returned None for filtered item")
        if isinstance(result, tuple) and len(result) == 2:
            messages, context = result
            if isinstance(context, dict):
                item.update(context)
        else:
            messages = result
        item["messages"] = messages
        return item


def build_llm(
    model_name: str,
    enable_expert_parallel: bool,
    gpu_memory_utilization: float = 0.6,
    max_model_len: int | None = None,
    max_num_seqs: int = 256,
) -> LLM:
    """
    Build vLLM LLM instance with memory optimizations for vision models.
    
    Args:
        model_name: Model name or path
        enable_expert_parallel: Whether to enable expert parallel
        gpu_memory_utilization: GPU memory utilization ratio (default 0.6 for vision models)
        max_model_len: Maximum model length (None for auto, lower values reduce memory)
        max_num_seqs: Maximum number of sequences to process in parallel
    """
    tensor_parallel_size = torch.cuda.device_count() if torch.cuda.is_available() else 1
    
    llm_kwargs = {
        "model": model_name,
        "tensor_parallel_size": tensor_parallel_size,
        "enable_expert_parallel": enable_expert_parallel,
        "dtype": "auto",
        "gpu_memory_utilization": gpu_memory_utilization,
        "trust_remote_code": True,
        "seed": 0,
        "max_num_seqs": max_num_seqs,
    }
    
    # Add mm_encoder_tp_mode for vision models
    if "VL" in model_name or "vision" in model_name.lower():
        llm_kwargs["mm_encoder_tp_mode"] = "data"
    
    # Add max_model_len if specified (helps reduce memory for vision models)
    if max_model_len is not None:
        llm_kwargs["max_model_len"] = max_model_len
    
    return LLM(**llm_kwargs)


def run_pipeline(args: argparse.Namespace) -> None:
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    
    # Enable V1 engine for better memory management with vision models (vLLM 0.12+)
    if "VL" in COT_MODEL or "vision" in COT_MODEL.lower():
        os.environ.setdefault("VLLM_USE_V1", "1")

    cot_system = load_prompt("cot_system.txt")
    cot_user = load_prompt("cot_user.txt", "{question}")
    answer_filter_system = load_prompt("answer_filter_system.txt")
    answer_filter_user = load_prompt("answer_filter_user.txt")
    rubrics_system = load_prompt("rubrics_system.txt")
    rubrics_user = load_prompt("rubrics_user.txt")

    cot_processor = AutoProcessor.from_pretrained(COT_PROCESSOR)
    filter_processor = AutoTokenizer.from_pretrained(FILTER_PROCESSOR)
    rubrics_processor = AutoTokenizer.from_pretrained(RUBRICS_PROCESSOR)

    def cot_message_fn(sample: dict[str, Any]) -> Optional[list[dict[str, Any]]]:
        question = get_question(sample)
        if not question:
            return None
        image_paths = resolve_image_paths(sample, args.image_base_path)
        user_text = cot_user.format(question=question)
        return build_vl_messages(cot_system, user_text, image_paths or None)

    cot_dataset = StageDataset(
        data_path=args.input_file,
        data_list=None,
        rank=args.rank,
        world_size=args.world_size,
        message_fn=cot_message_fn,
    )
    cot_dataloader = DataLoader(
        cot_dataset,
        batch_size=COT_BATCH_SIZE,
        shuffle=False,
        collate_fn=partial(vllm_collate_batch, processor=cot_processor),
    )

    cot_llm = build_llm(
        COT_MODEL,
        enable_expert_parallel=False,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
    )
    cot_sampling = SamplingParams(**COT_SAMPLING)

    def cot_post_process(item: dict[str, Any], output: str) -> dict[str, Any]:
        item[COT_FIELD] = output
        return item

    run_vllm(
        dataloader=cot_dataloader,
        llm=cot_llm,
        sampling_params=cot_sampling,
        save_path=args.cot_raw_out,
        filtered_save_path=args.cot_filtered_out,
        run_name="cot_gen",
        post_process_output=cot_post_process,
        post_filter_func=None,
        keep_raw=args.keep_raw,
    )
    unload_destroy(cot_llm)
    cot_llm = None

    def filter_message_fn(sample: dict[str, Any]) -> Optional[tuple[list[dict[str, Any]], dict[str, Any]]]:
        question = get_question(sample)
        ground_truth = get_answer(sample)
        cot_text = sample.get(COT_FIELD, "")
        if not question or not ground_truth or not cot_text:
            return None
        extracted_answer = extract_answer_from_cot(cot_text)
        if not extracted_answer:
            return None
        user_text = answer_filter_user.format(
            question=question,
            ground_truth=ground_truth,
            model_answer=extracted_answer,
        )
        messages = build_text_messages(answer_filter_system, user_text)
        return messages, {"extracted_answer": extracted_answer}

    filter_dataset = StageDataset(
        data_path=args.cot_filtered_out,
        data_list=None,
        rank=args.rank,
        world_size=args.world_size,
        message_fn=filter_message_fn,
    )
    filter_dataloader = DataLoader(
        filter_dataset,
        batch_size=FILTER_BATCH_SIZE,
        shuffle=False,
        collate_fn=partial(vllm_collate_batch, processor=filter_processor),
    )

    filter_llm = build_llm(
        FILTER_MODEL,
        enable_expert_parallel=True,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
    )
    filter_sampling = SamplingParams(**FILTER_SAMPLING)

    def filter_post_process(item: dict[str, Any], output: str) -> dict[str, Any]:
        judgement = parse_judgement(output)
        if judgement is None:
            item["is_correct"] = False
            item["status"] = "failure"
        else:
            item["is_correct"] = judgement
            item["status"] = "success"
        item["judege_result"] = output
        return item

    def filter_post_filter(item: dict[str, Any], _output: str) -> bool:
        return item.get("is_correct") is True and item.get("status") == "success"

    run_vllm(
        dataloader=filter_dataloader,
        llm=filter_llm,
        sampling_params=filter_sampling,
        save_path=args.filter_raw_out,
        filtered_save_path=args.filter_filtered_out,
        run_name="answer_filter",
        post_process_output=filter_post_process,
        post_filter_func=filter_post_filter,
        keep_raw=args.keep_raw,
    )
    unload_destroy(filter_llm)
    filter_llm = None

    def rubrics_message_fn(sample: dict[str, Any]) -> Optional[list[dict[str, Any]]]:
        question = get_question(sample)
        cot_text = sample.get(COT_FIELD, "")
        if not question or not cot_text:
            return None
        user_text = rubrics_user.format(
            query=question,
            reference_cot="<think>\n" + cot_text,
        )
        return build_text_messages(rubrics_system, user_text)

    rubrics_dataset = StageDataset(
        data_path=args.filter_filtered_out,
        data_list=None,
        rank=args.rank,
        world_size=args.world_size,
        message_fn=rubrics_message_fn,
    )
    rubrics_dataloader = DataLoader(
        rubrics_dataset,
        batch_size=RUBRICS_BATCH_SIZE,
        shuffle=False,
        collate_fn=partial(vllm_collate_batch, processor=rubrics_processor),
    )

    rubrics_llm = build_llm(
        RUBRICS_MODEL,
        enable_expert_parallel=True,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
    )
    rubrics_sampling = SamplingParams(**RUBRICS_SAMPLING)

    def rubrics_post_process(item: dict[str, Any], output: str) -> dict[str, Any]:
        rubrics = parse_rubrics(output)
        if rubrics is None:
            item[RUBRICS_FIELD] = None
            item["status"] = "failure"
        else:
            item[RUBRICS_FIELD] = json.dumps(rubrics, ensure_ascii=False)
            item["status"] = "success"
        return item

    def rubrics_post_filter(item: dict[str, Any], _output: str) -> bool:
        return item.get("status") == "success"

    run_vllm(
        dataloader=rubrics_dataloader,
        llm=rubrics_llm,
        sampling_params=rubrics_sampling,
        save_path=args.rubrics_raw_out,
        filtered_save_path=args.rubrics_filtered_out,
        run_name="rubrics_gen",
        post_process_output=rubrics_post_process,
        post_filter_func=rubrics_post_filter,
        keep_raw=args.keep_raw,
    )
    unload_destroy(rubrics_llm)
    rubrics_llm = None


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run rubrics generation pipeline")
    parser.add_argument("--input_file", type=str, required=True, help="Input dataset (parquet/json/jsonl)")
    parser.add_argument("--image_base_path", type=str, default=None, help="Base path for image files")
    parser.add_argument("--rank", type=int, default=0, help="Rank for distributed split")
    parser.add_argument("--world_size", type=int, default=1, help="World size for distributed split")
    parser.add_argument("--keep_raw", action="store_true", help="Save raw outputs for each stage")
    
    # Memory optimization arguments for vLLM
    parser.add_argument(
        "--gpu_memory_utilization",
        type=float,
        default=0.6,
        help="GPU memory utilization ratio (default 0.6 for vision models, lower if OOM)",
    )
    parser.add_argument(
        "--max_model_len",
        type=int,
        default=None,
        help="Maximum model length (None for auto, lower values reduce memory for vision models)",
    )
    parser.add_argument(
        "--max_num_seqs",
        type=int,
        default=256,
        help="Maximum number of sequences to process in parallel (default 256)",
    )

    parser.add_argument("--cot_raw_out", type=str, required=True, help="Raw output JSONL for CoT stage")
    parser.add_argument("--cot_filtered_out", type=str, required=True, help="Filtered output JSONL for CoT stage")

    parser.add_argument("--filter_raw_out", type=str, required=True, help="Raw output JSONL for filter stage")
    parser.add_argument("--filter_filtered_out", type=str, required=True, help="Filtered output JSONL for filter stage")

    parser.add_argument("--rubrics_raw_out", type=str, required=True, help="Raw output JSONL for rubrics stage")
    parser.add_argument(
        "--rubrics_filtered_out", type=str, required=True, help="Filtered output JSONL for rubrics stage"
    )
    return parser


if __name__ == "__main__":
    args = build_arg_parser().parse_args()
    run_pipeline(args)
