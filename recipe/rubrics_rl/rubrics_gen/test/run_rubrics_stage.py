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
    rubrics_model_name: str,
    rubrics_model_path: str,
    output_root: Path,
    batch_size: int,
    gpu_memory_utilization: float,
    max_model_len: Optional[int],
    max_num_seqs: int,
    enable_expert_parallel: bool,
    rubrics_max_tokens: int,
) -> tuple[int, int]:
    from torch.utils.data import DataLoader
    from transformers import AutoTokenizer
    from vllm import SamplingParams

    model_tag = sanitize_model_name(cot_model_name)
    model_out_dir = output_root / model_tag
    cot_results_out = model_out_dir / "cot_results.jsonl"
    rubrics_raw_out = model_out_dir / "rubrics_raw.jsonl"
    rubrics_results_out = model_out_dir / "rubrics_results.jsonl"

    if not cot_results_out.exists():
        raise FileNotFoundError(f"Missing CoT results for {cot_model_name}: {cot_results_out}")

    for path in (rubrics_raw_out, rubrics_results_out):
        if path.exists():
            path.unlink()

    rubrics_system = p.load_prompt("rubrics_system.txt")
    rubrics_user = p.load_prompt("rubrics_user.txt")
    message_fn = partial(build_rubrics_message, rubrics_system=rubrics_system, rubrics_user=rubrics_user)

    dataset = p.StageDataset(
        data_path=str(cot_results_out),
        data_list=None,
        rank=0,
        world_size=1,
        message_fn=message_fn,
    )

    processor = AutoTokenizer.from_pretrained(rubrics_model_path, trust_remote_code=True)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=partial(p.vllm_collate_batch, processor=processor),
    )

    sampling_cfg = copy.deepcopy(p.RUBRICS_SAMPLING)
    sampling_cfg["max_tokens"] = rubrics_max_tokens
    sampling = SamplingParams(**sampling_cfg)

    llm = p.build_llm(
        model_name=rubrics_model_path,
        enable_expert_parallel=enable_expert_parallel,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_model_len,
        max_num_seqs=max_num_seqs,
    )

    def rubrics_post_process(item: dict[str, Any], output: str) -> dict[str, Any]:
        parsed = p.parse_rubrics(output)
        # Keep rubrics outputs minimal and drop CoT text from output files.
        record: dict[str, Any] = {
            "id": str(item.get("id", "")),
            "question": p.get_question(item),
            "image": item.get("image", []),
            "cot_model_name": item.get("cot_model_name"),
            "cot_model_path": item.get("cot_model_path"),
            "rubrics_model_name": rubrics_model_name,
            "rubrics_model_path": rubrics_model_path,
            "rubrics_raw_text": output,
        }
        if parsed is None:
            record[p.RUBRICS_FIELD] = None
            record["rubrics_status"] = "failure"
        else:
            record[p.RUBRICS_FIELD] = json.dumps(parsed, ensure_ascii=False)
            record["rubrics_status"] = "success"
        return record

    all_results, _ = p.run_vllm(
        dataloader=dataloader,
        llm=llm,
        sampling_params=sampling,
        save_path=str(rubrics_raw_out),
        filtered_save_path=str(rubrics_results_out),
        run_name=f"rubrics_gen[{cot_model_name}]",
        post_process_output=rubrics_post_process,
        post_filter_func=None,
        keep_raw=True,
    )
    unload_destroy(llm)

    parsed_success = sum(1 for item in all_results if item.get("rubrics_status") == "success")
    return len(all_results), parsed_success


def main() -> None:
    parser = argparse.ArgumentParser(description="Run rubrics stage only using fixed rubrics model")
    parser.add_argument("--models", nargs="+", required=True, help="CoT model names used in CoT stage")
    parser.add_argument("--output_dir", type=Path, default=ROOT_DIR / "test" / "outputs")
    parser.add_argument("--models_dir", type=Path, default=ROOT_DIR / "models")
    parser.add_argument("--download-models", action="store_true")
    parser.add_argument("--rubrics_model", type=str, default=p.RUBRICS_MODEL)

    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.35)
    parser.add_argument("--max_model_len", type=int, default=4096)
    parser.add_argument("--max_num_seqs", type=int, default=1)
    parser.add_argument("--enable_expert_parallel", action="store_true")
    parser.add_argument("--rubrics_max_tokens", type=int, default=4096)

    args = parser.parse_args()

    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.models_dir.mkdir(parents=True, exist_ok=True)

    rubrics_model_path = maybe_download_model(args.rubrics_model, args.models_dir, args.download_models)
    print(f"[run] fixed_rubrics_model={args.rubrics_model}")
    print(f"[run] fixed_rubrics_model_path={rubrics_model_path}")

    for cot_model_name in args.models:
        print(f"\n[run] cot_model={cot_model_name}")
        total, parsed_success = run_one_model(
            cot_model_name=cot_model_name,
            rubrics_model_name=args.rubrics_model,
            rubrics_model_path=rubrics_model_path,
            output_root=args.output_dir,
            batch_size=args.batch_size,
            gpu_memory_utilization=args.gpu_memory_utilization,
            max_model_len=args.max_model_len,
            max_num_seqs=args.max_num_seqs,
            enable_expert_parallel=args.enable_expert_parallel,
            rubrics_max_tokens=args.rubrics_max_tokens,
        )
        rate = (parsed_success / total * 100.0) if total else 0.0
        print(f"[done] {cot_model_name}: rubrics_parsed={parsed_success}/{total} ({rate:.2f}%)")


if __name__ == "__main__":
    main()
