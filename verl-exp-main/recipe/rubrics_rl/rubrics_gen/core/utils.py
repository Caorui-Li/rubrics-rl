# -*- coding: utf-8 -*-
"""Utility helpers for the rubrics pipeline."""

from __future__ import annotations

import gc
from typing import Any

import torch
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, AutoTokenizer


def messages_to_vllm_input(
    messages: list[dict[str, Any]], processor: AutoProcessor | AutoTokenizer
) -> dict[str, Any]:
    prompt = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    vllm_input: dict[str, Any] = {"prompt": prompt}

    # Check if processor is multimodal by checking for image_processor
    if hasattr(processor, "image_processor"):
        # Multimodal model: process vision info
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
    """Safely shutdown and destroy vLLM LLM instance (vLLM 0.12+).

    For vLLM 0.12, the shutdown process involves:
    1. Calling shutdown() if available
    2. Deleting the LLM object
    3. Cleaning up distributed environment
    4. Clearing CUDA cache
    """
    # Try to shutdown using vLLM 0.12 API
    try:
        if hasattr(llm, "shutdown"):
            llm.shutdown()
        elif hasattr(llm, "llm_engine") and hasattr(llm.llm_engine, "shutdown"):
            llm.llm_engine.shutdown()
    except Exception as e:
        print(f"Warning: Error during vLLM engine shutdown: {e}")

    # Delete the LLM object
    try:
        del llm
    except Exception as e:
        print(f"Warning: Error deleting LLM object: {e}")

    # Force garbage collection
    gc.collect()

    # Clean up distributed environment
    try:
        from vllm.distributed.parallel_state import destroy_distributed_environment, destroy_model_parallel

        destroy_model_parallel()
        destroy_distributed_environment()
    except Exception as e:
        print(f"Warning: Error destroying distributed environment: {e}")

    # Clear CUDA cache
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
