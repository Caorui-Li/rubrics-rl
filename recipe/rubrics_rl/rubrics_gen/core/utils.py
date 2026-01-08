# -*- coding: utf-8 -*-
"""Utility helpers for the rubrics pipeline."""

from __future__ import annotations

import gc
from typing import Any

import torch
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor


def messages_to_vllm_input(messages: list[dict[str, Any]], processor: AutoProcessor) -> dict[str, Any]:
    prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    vllm_input: dict[str, Any] = {"prompt": prompt}

    image_inputs, video_inputs, video_kwargs = process_vision_info(
        messages,
        image_patch_size=processor.image_processor.patch_size,
        return_video_kwargs=True,
        return_video_metadata=True,
    )

    mm_data: dict[str, Any] = {}
    if image_inputs is not None:
        mm_data["image"] = image_inputs
    if video_inputs is not None:
        mm_data["video"] = video_inputs
    if mm_data:
        vllm_input["multi_modal_data"] = mm_data
    if video_kwargs is not None:
        vllm_input["mm_processor_kwargs"] = video_kwargs

    return vllm_input


def unload_with_sleep(llm, level: int = 2) -> None:
    llm.sleep(level=level)
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()


def unload_destroy(llm) -> None:
    llm.llm_engine.model_executor.shutdown()
    del llm
    gc.collect()

    from vllm.distributed.parallel_state import destroy_distributed_environment, destroy_model_parallel

    destroy_model_parallel()
    destroy_distributed_environment()

    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
