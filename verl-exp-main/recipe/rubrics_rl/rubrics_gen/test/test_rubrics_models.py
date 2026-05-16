#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import copy
import json
import os
import re
import sys
from functools import partial
from pathlib import Path
from typing import Any, Optional

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

import pipeline as p
from core import unload_destroy


def sanitize_model_name(name: str) -> str:
    value = re.sub(r"[^a-zA-Z0-9._-]+", "--", name.strip("/"))
    return value or "model"


def maybe_download_model(model_name: str, models_dir: Path, download_models: bool) -> str:
    local_input = Path(model_name).expanduser()
    if local_input.exists():
        return str(local_input)

    target_dir = models_dir / sanitize_model_name(model_name)
    if target_dir.exists() and any(target_dir.iterdir()):
        return str(target_dir)

    if not download_models:
        return model_name

    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise RuntimeError("huggingface_hub is required for --download-models") from exc

    target_dir.mkdir(parents=True, exist_ok=True)
    print(f"[download] {model_name} -> {target_dir}")
    snapshot_download(
        repo_id=model_name,
        local_dir=str(target_dir),
        local_dir_use_symlinks=False,
        resume_download=True,
    )
    return str(target_dir)


def build_cot_message(sample: dict[str, Any], cot_system: str, cot_user: str, image_base_path: Optional[str]) -> Optional[list[dict[str, Any]]]:
    question = p.get_question(sample)
    if not question:
        return None
    image_paths = p.resolve_image_paths(sample, image_base_path)
    user_text = cot_user.format(question=question)
    return p.build_vl_messages(cot_system, user_text, image_paths or None)


def build_rubrics_message(sample: dict[str, Any], rubrics_system: str, rubrics_user: str) -> Optional[list[dict[str, Any]]]:
    question = p.get_question(sample)
    cot_text = sample.get(p.COT_FIELD, "")
    if not question or not cot_text:
        return None
    user_text = rubrics_user.format(
        query=question,
        reference_cot="<think>\n" + cot_text,
    )
    return p.build_text_messages(rubrics_system, user_text)


def run_one_model(
    cot_model_name: str,
    cot_model_path: str,
    rubrics_model_name: str,
    rubrics_model_path: str,
    input_file: Path,
    output_root: Path,
    image_base_path: Optional[str],
    batch_size: int,
    gpu_memory_utilization: float,
    max_model_len: Optional[int],
    max_num_seqs: int,
    enable_expert_parallel: bool,
) -> tuple[int, int]:
    from torch.utils.data import DataLoader
    from transformers import AutoProcessor, AutoTokenizer
    from vllm import SamplingParams

    model_tag = sanitize_model_name(cot_model_name)
    model_out_dir = output_root / model_tag
    model_out_dir.mkdir(parents=True, exist_ok=True)

    cot_raw_out = model_out_dir / "cot_raw.jsonl"
    cot_results_out = model_out_dir / "cot_results.jsonl"
    rubrics_raw_out = model_out_dir / "rubrics_raw.jsonl"
    rubrics_results_out = model_out_dir / "rubrics_results.jsonl"
    for path in (cot_raw_out, cot_results_out, rubrics_raw_out, rubrics_results_out):
        if path.exists():
            path.unlink()

    cot_system = p.load_prompt("cot_system.txt")
    cot_user = p.load_prompt("cot_user.txt", "{question}")
    rubrics_system = p.load_prompt("rubrics_system.txt")
    rubrics_user = p.load_prompt("rubrics_user.txt")

    cot_message_fn = partial(
        build_cot_message,
        cot_system=cot_system,
        cot_user=cot_user,
        image_base_path=image_base_path,
    )
    cot_dataset = p.StageDataset(
        data_path=str(input_file),
        data_list=None,
        rank=0,
        world_size=1,
        message_fn=cot_message_fn,
    )

    cot_processor = AutoProcessor.from_pretrained(cot_model_path, trust_remote_code=True)
    cot_dataloader = DataLoader(
        cot_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=partial(p.vllm_collate_batch, processor=cot_processor),
    )

    if "VL" in cot_model_path or "vision" in cot_model_path.lower():
        os.environ.setdefault("VLLM_USE_V1", "1")

    cot_sampling_cfg = copy.deepcopy(p.COT_SAMPLING)
    cot_sampling = SamplingParams(**cot_sampling_cfg)
    cot_llm = p.build_llm(
        model_name=cot_model_path,
        enable_expert_parallel=enable_expert_parallel,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_model_len,
        max_num_seqs=max_num_seqs,
    )

    def cot_post_process(item: dict[str, Any], output: str) -> dict[str, Any]:
        item[p.COT_FIELD] = output
        item["cot_model_name"] = cot_model_name
        item["cot_model_path"] = cot_model_path
        item["cot_status"] = "success" if output and output.strip() else "failure"
        return item

    p.run_vllm(
        dataloader=cot_dataloader,
        llm=cot_llm,
        sampling_params=cot_sampling,
        save_path=str(cot_raw_out),
        filtered_save_path=str(cot_results_out),
        run_name=f"cot_gen[{cot_model_name}]",
        post_process_output=cot_post_process,
        post_filter_func=None,
        keep_raw=True,
    )
    unload_destroy(cot_llm)

    rubrics_message_fn = partial(build_rubrics_message, rubrics_system=rubrics_system, rubrics_user=rubrics_user)
    rubrics_dataset = p.StageDataset(
        data_path=str(cot_results_out),
        data_list=None,
        rank=0,
        world_size=1,
        message_fn=rubrics_message_fn,
    )

    rubrics_processor = AutoTokenizer.from_pretrained(rubrics_model_path, trust_remote_code=True)
    rubrics_dataloader = DataLoader(
        rubrics_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=partial(p.vllm_collate_batch, processor=rubrics_processor),
    )

    rubrics_sampling_cfg = copy.deepcopy(p.RUBRICS_SAMPLING)
    rubrics_sampling = SamplingParams(**rubrics_sampling_cfg)
    rubrics_llm = p.build_llm(
        model_name=rubrics_model_path,
        enable_expert_parallel=enable_expert_parallel,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_model_len,
        max_num_seqs=max_num_seqs,
    )

    def rubrics_post_process(item: dict[str, Any], output: str) -> dict[str, Any]:
        parsed = p.parse_rubrics(output)
        item["rubrics_model_name"] = rubrics_model_name
        item["rubrics_model_path"] = rubrics_model_path
        item["rubrics_raw_text"] = output
        if parsed is None:
            item[p.RUBRICS_FIELD] = None
            item["rubrics_status"] = "failure"
        else:
            item[p.RUBRICS_FIELD] = json.dumps(parsed, ensure_ascii=False)
            item["rubrics_status"] = "success"
        return item

    all_results, _ = p.run_vllm(
        dataloader=rubrics_dataloader,
        llm=rubrics_llm,
        sampling_params=rubrics_sampling,
        save_path=str(rubrics_raw_out),
        filtered_save_path=str(rubrics_results_out),
        run_name=f"rubrics_gen[{cot_model_name}]",
        post_process_output=rubrics_post_process,
        post_filter_func=None,
        keep_raw=True,
    )
    unload_destroy(rubrics_llm)

    parsed_success = sum(1 for item in all_results if item.get("rubrics_status") == "success")
    return len(all_results), parsed_success


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare QwenVL CoT outputs and downstream rubrics with fixed rubrics model"
    )
    parser.add_argument(
        "--input_file",
        type=Path,
        default=ROOT_DIR / "test" / "test.jsonl",
        help="Path to test JSONL (10 samples)",
    )
    parser.add_argument("--models", nargs="+", required=True, help="QwenVL model names or local paths for CoT stage")
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=ROOT_DIR / "test" / "outputs",
        help="Root directory for per-model outputs",
    )
    parser.add_argument(
        "--models_dir",
        type=Path,
        default=ROOT_DIR / "models",
        help="Local model cache dir used with --download-models",
    )
    parser.add_argument("--download-models", action="store_true", help="Download remote models into --models_dir")
    parser.add_argument("--image_base_path", type=str, default=None, help="Base path for relative image paths")

    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.6)
    parser.add_argument("--max_model_len", type=int, default=None)
    parser.add_argument("--max_num_seqs", type=int, default=32)
    parser.add_argument("--enable_expert_parallel", action="store_true")

    args = parser.parse_args()

    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

    if not args.input_file.exists():
        raise FileNotFoundError(f"Input test file not found: {args.input_file}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.models_dir.mkdir(parents=True, exist_ok=True)

    rubrics_model_name = p.RUBRICS_MODEL
    rubrics_model_path = maybe_download_model(rubrics_model_name, args.models_dir, args.download_models)

    summary: list[tuple[str, int, int]] = []
    for cot_model_name in args.models:
        cot_model_path = maybe_download_model(cot_model_name, args.models_dir, args.download_models)
        print(f"\n[run] cot_model={cot_model_name}")
        print(f"[run] cot_model_path={cot_model_path}")
        print(f"[run] fixed_rubrics_model={rubrics_model_name}")
        print(f"[run] fixed_rubrics_model_path={rubrics_model_path}")

        total, parsed_success = run_one_model(
            cot_model_name=cot_model_name,
            cot_model_path=cot_model_path,
            rubrics_model_name=rubrics_model_name,
            rubrics_model_path=rubrics_model_path,
            input_file=args.input_file,
            output_root=args.output_dir,
            image_base_path=args.image_base_path,
            batch_size=args.batch_size,
            gpu_memory_utilization=args.gpu_memory_utilization,
            max_model_len=args.max_model_len,
            max_num_seqs=args.max_num_seqs,
            enable_expert_parallel=args.enable_expert_parallel,
        )
        summary.append((cot_model_name, total, parsed_success))

    print("\n=== Summary ===")
    for model_name, total, parsed_success in summary:
        rate = (parsed_success / total * 100.0) if total else 0.0
        print(f"{model_name}: rubrics_parsed={parsed_success}/{total} ({rate:.2f}%)")


if __name__ == "__main__":
    main()
