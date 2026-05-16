#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import copy
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

    from huggingface_hub import snapshot_download

    target_dir.mkdir(parents=True, exist_ok=True)
    print(f"[download] {model_name} -> {target_dir}")
    snapshot_download(
        repo_id=model_name,
        local_dir=str(target_dir),
        local_dir_use_symlinks=False,
        resume_download=True,
    )
    return str(target_dir)


def build_cot_message(
    sample: dict[str, Any],
    cot_system: str,
    cot_user: str,
    image_base_path: Optional[str],
) -> Optional[list[dict[str, Any]]]:
    question = p.get_question(sample)
    if not question:
        return None
    image_paths = p.resolve_image_paths(sample, image_base_path)
    user_text = cot_user.format(question=question)
    return p.build_vl_messages(cot_system, user_text, image_paths or None)


def run_one_model(
    model_name: str,
    model_path: str,
    input_file: Path,
    output_root: Path,
    image_base_path: Optional[str],
    batch_size: int,
    gpu_memory_utilization: float,
    max_model_len: Optional[int],
    max_num_seqs: int,
    enable_expert_parallel: bool,
    cot_max_tokens: int,
) -> int:
    from torch.utils.data import DataLoader
    from transformers import AutoProcessor
    from vllm import SamplingParams

    model_tag = sanitize_model_name(model_name)
    model_out_dir = output_root / model_tag
    model_out_dir.mkdir(parents=True, exist_ok=True)

    cot_raw_out = model_out_dir / "cot_raw.jsonl"
    cot_results_out = model_out_dir / "cot_results.jsonl"
    for path in (cot_raw_out, cot_results_out):
        if path.exists():
            path.unlink()

    cot_system = p.load_prompt("cot_system.txt")
    cot_user = p.load_prompt("cot_user.txt", "{question}")
    cot_message_fn = partial(
        build_cot_message,
        cot_system=cot_system,
        cot_user=cot_user,
        image_base_path=image_base_path,
    )

    dataset = p.StageDataset(
        data_path=str(input_file),
        data_list=None,
        rank=0,
        world_size=1,
        message_fn=cot_message_fn,
    )

    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=partial(p.vllm_collate_batch, processor=processor),
    )

    # Qwen2-VL is more stable with legacy engine in some vLLM versions.
    if "Qwen2-VL" in model_path:
        os.environ["VLLM_USE_V1"] = "0"
    elif "VL" in model_path or "vision" in model_path.lower():
        os.environ.setdefault("VLLM_USE_V1", "1")

    sampling_cfg = copy.deepcopy(p.COT_SAMPLING)
    sampling_cfg["max_tokens"] = cot_max_tokens
    sampling = SamplingParams(**sampling_cfg)

    llm = p.build_llm(
        model_name=model_path,
        enable_expert_parallel=enable_expert_parallel,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_model_len,
        max_num_seqs=max_num_seqs,
    )

    def cot_post_process(item: dict[str, Any], output: str) -> dict[str, Any]:
        # Keep CoT outputs minimal for manual comparison and to avoid carrying source rubrics.
        record: dict[str, Any] = {
            "id": str(item.get("id", "")),
            "question": p.get_question(item),
            "image": item.get("image", []),
            p.COT_FIELD: output,
            "cot_model_name": model_name,
            "cot_model_path": model_path,
            "cot_status": "success" if output and output.strip() else "failure",
        }
        return record

    results, _ = p.run_vllm(
        dataloader=dataloader,
        llm=llm,
        sampling_params=sampling,
        save_path=str(cot_raw_out),
        filtered_save_path=str(cot_results_out),
        run_name=f"cot_gen[{model_name}]",
        post_process_output=cot_post_process,
        post_filter_func=None,
        keep_raw=True,
    )
    unload_destroy(llm)
    return len(results)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run CoT stage only for multiple models")
    parser.add_argument("--input_file", type=Path, required=True)
    parser.add_argument("--models", nargs="+", required=True, help="CoT model names or local paths")
    parser.add_argument("--output_dir", type=Path, default=ROOT_DIR / "test" / "outputs")
    parser.add_argument("--models_dir", type=Path, default=ROOT_DIR / "models")
    parser.add_argument("--download-models", action="store_true")
    parser.add_argument("--image_base_path", type=str, default=None)

    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.35)
    parser.add_argument("--max_model_len", type=int, default=4096)
    parser.add_argument("--max_num_seqs", type=int, default=1)
    parser.add_argument("--enable_expert_parallel", action="store_true")
    parser.add_argument("--cot_max_tokens", type=int, default=4096)

    args = parser.parse_args()

    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

    if not args.input_file.exists():
        raise FileNotFoundError(f"Input not found: {args.input_file}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.models_dir.mkdir(parents=True, exist_ok=True)

    for model_name in args.models:
        if "Qwen2-VL" in model_name and "Instruct" not in model_name:
            print(f"[warn] {model_name} is not an Instruct checkpoint; multimodal chat may fail.")
            print(f"[warn] Prefer: {model_name}-Instruct")
        model_path = maybe_download_model(model_name, args.models_dir, args.download_models)
        print(f"\n[run] cot_model={model_name}")
        print(f"[run] cot_model_path={model_path}")
        total = run_one_model(
            model_name=model_name,
            model_path=model_path,
            input_file=args.input_file,
            output_root=args.output_dir,
            image_base_path=args.image_base_path,
            batch_size=args.batch_size,
            gpu_memory_utilization=args.gpu_memory_utilization,
            max_model_len=args.max_model_len,
            max_num_seqs=args.max_num_seqs,
            enable_expert_parallel=args.enable_expert_parallel,
            cot_max_tokens=args.cot_max_tokens,
        )
        print(f"[done] {model_name}: cot_results={total}")


if __name__ == "__main__":
    main()
