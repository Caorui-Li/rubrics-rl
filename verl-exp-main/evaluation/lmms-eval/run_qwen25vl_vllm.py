#!/usr/bin/env python3
"""
Qwen2.5-VL Evaluation Script using vLLM

This script evaluates Qwen2.5-VL models on Math benchmarks using flash attention.
Supports both base models and locally trained models.

# Usage with accelerate launch (recommended for multi-GPU):
# accelerate launch --num_processes 1 -m run_qwen25vl_vllm --model_path Qwen/Qwen2.5-VL-7B-Instruct --tasks mathvista --batch_size 64

# Or directly with Python (single GPU):
# python -m run_qwen25vl_vllm --model_path Qwen/Qwen2.5-VL-7B-Instruct --tasks mathvista --batch_size 64

# locally trained model
python -m run_qwen25vl_vllm --model_path /path/to/trained/model --tasks mathvista

# multiple tasks (with interleaved visuals for MMMU)
python -m run_qwen25vl_vllm --model_path /path/to/model --tasks mathvista,mmmu --interleave_visuals --log_samples

# limit number of samples for quick testing
python -m run_qwen25vl_vllm --model_path /path/to/model --tasks mathvista --limit 5 --log_samples

# specify HuggingFace cache directory
python -m run_qwen25vl_vllm --model_path /path/to/model --hf_home /path/to/hf_cache

# custom system prompt from file
echo "You are a helpful assistant." > system_prompt.txt
python -m run_qwen25vl_vllm --model_path /path/to/model --system_prompt system_prompt.txt --tasks mathvista

"""

# Force IPv4 for distributed communication (must be set before any imports)
import os
os.environ['MASTER_ADDR'] = '127.0.0.1'
os.environ['MASTER_PORT'] = '29500'
os.environ['RANK'] = '0'
os.environ['WORLD_SIZE'] = '1'
os.environ['LOCAL_RANK'] = '0'
os.environ['NCCL_SOCKET_FAMILY'] = 'AF_INET'
os.environ['NCCL_IB_DISABLE'] = '1'
os.environ['NCCL_SOCKET_IFNAME'] = 'eth0'
os.environ['GLOO_SOCKET_IFNAME'] = 'eth0'
os.environ['NCCL_P2P_DISABLE'] = '1'
os.environ['NCCL_DEBUG'] = 'WARN'
os.environ['TORCH_DISTRIBUTED_DEBUG'] = 'OFF'
os.environ['TORCH_NCCL_BLOCKING_WAIT'] = '0'

import sys
import socket

# Monkey patch socket.getaddrinfo to force IPv4 for localhost
_original_getaddrinfo = socket.getaddrinfo
def _patched_getaddrinfo(host, port, family=0, type=0, proto=0, flags=0):
    # Force IPv4 (AF_INET) for localhost to avoid IPv6 issues
    if host in ('localhost', '::1'):
        host = '127.0.0.1'
        family = socket.AF_INET
    return _original_getaddrinfo(host, port, family, type, proto, flags)
socket.getaddrinfo = _patched_getaddrinfo

import re
import multiprocessing

from datetime import datetime
from typing import List, Optional
from copy import deepcopy

# Add the lmms-eval path to sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from loguru import logger as eval_logger
from tqdm import tqdm
from PIL import Image
from qwen_vl_utils import process_vision_info

from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model

# Pre-check and handle torch.distributed to avoid IPv6 issues
try:
    import torch
    import torch.distributed as dist
    # Check if already initialized
    if dist.is_available() and not dist.is_initialized():
        # For single-process mode, we can skip distributed init entirely
        # by ensuring WORLD_SIZE=1 is set (already done above)
        eval_logger.info(f"Torch distributed available but not initialized. WORLD_SIZE={os.environ.get('WORLD_SIZE')}")
except Exception as e:
    print(f"Note: torch.distributed check: {e}")

try:
    from vllm import LLM, SamplingParams
except ImportError:
    vllm = None
    print("Warning: vllm not installed. Please install vllm to use this script.")


def get_visual_content(visuals):
    """Process visual inputs and return formatted visual content."""
    visual_content = []
    for visual in visuals:
        if isinstance(visual, str) and (
            ".mp4" in visual or ".avi" in visual or ".mov" in visual or 
            ".flv" in visual or ".wmv" in visual or ".mkv" in visual
        ):
            visual_content.append({"type": "video", "video": visual})
        elif isinstance(visual, str) and (
            ".wav" in visual or ".mp3" in visual or ".flac" in visual or ".m4a" in visual
        ):
            visual_content.append({"type": "audio", "audio": visual})
        elif isinstance(visual, Image.Image):
            visual_content.append({"type": "image", "image": visual})
    return visual_content


@register_model("qwen2_5_vl_vllm")
class Qwen2_5_VL_VLLM(lmms):
    """Qwen2.5-VL VLLM model for evaluation."""
    
    def __init__(
        self,
        model_version: str = "Qwen/Qwen2.5-VL-7B-Instruct",
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.8,
        batch_size: int = 1,
        timeout: int = 60,
        max_images: int = 32,
        max_videos: int = 8,
        max_audios: int = 8,
        max_frame_num: int = 32,
        threads: int = 16,
        trust_remote_code: Optional[bool] = True,
        system_prompt: Optional[str] = "You are a helpful assistant.",
        extract_answer: Optional[bool] = False,
        place_visual_first: Optional[bool] = False,
        min_pixels: Optional[int] = 4 * 28 * 28,
        max_pixels: Optional[int] = 12845056,
        interleave_visuals: Optional[bool] = False,
        **kwargs,
    ) -> None:
        super().__init__()
        self.model_version = model_version
        self.max_images = max_images
        self.max_frame_num = max_frame_num
        self.threads = threads
        self.place_visual_first = place_visual_first
        self.interleave_visuals = interleave_visuals

        init_params = [
            "model_version", "tensor_parallel_size", "gpu_memory_utilization", 
            "batch_size", "timeout", "max_images", "max_videos", "max_audios", 
            "max_frame_num", "threads", "trust_remote_code", "place_visual_first",
            "interleave_visuals"
        ]

        filtered_kwargs = {k: v for k, v in kwargs.items() if k not in init_params}

        self.client = LLM(
            model=self.model_version,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            limit_mm_per_prompt={"image": max_images, "video": max_videos, "audio": max_audios},
            trust_remote_code=trust_remote_code,
            **filtered_kwargs,
        )
        
        self._rank = 0
        self._world_size = 1

        self.device = "cuda"
        self.batch_size_per_gpu = int(batch_size)
        self.max_pixels = max_pixels
        self.min_pixels = min_pixels
        self.system_prompt = system_prompt
        self.enable_extract_answer = extract_answer
        
        if os.path.exists(system_prompt):
            with open(system_prompt, "r") as f:
                self.system_prompt = f.read()
        else:
            self.system_prompt = system_prompt

        from transformers import AutoProcessor
        self.processor = AutoProcessor.from_pretrained(self.model_version)
        self.tokenizer = self.processor.tokenizer

    def extract_answer(self, response: str) -> str:
        """Extract the answer from the response with r1 format."""
        if self.enable_extract_answer:
            answer_pattern = r"<answer>(.*?)</answer>"
            match = re.search(answer_pattern, response)
            if match:
                return match.group(1).strip()
        return response

    def flatten(self, input_list):
        """Flatten a nested list."""
        new_list = []
        for i in input_list:
            for j in i:
                new_list.append(j)
        return new_list

    def generate_until(self, requests, task_dict=None) -> List[str]:
        """Generate responses for all requests."""
        res = []
        pbar = tqdm(total=len(requests), disable=(self.rank != 0), desc="Model Responding")

        # Use provided task_dict or fall back to instance attribute
        task_dict = task_dict if task_dict is not None else getattr(self, 'task_dict', None)

        batch_size = self.batch_size_per_gpu
        batched_requests = [requests[i : i + batch_size] for i in range(0, len(requests), batch_size)]

        for batch_requests in batched_requests:
            inputs = []
            for idx in tqdm(range(len(batch_requests)), disable=(self.rank != 0), desc="Processing batch"):
                contexts, gen_kwargs, doc_to_visual, doc_id, task, split = batch_requests[idx].arguments

                if task_dict is not None:
                    visuals = [doc_to_visual(task_dict[task][split][doc_id])]
                else:
                    visuals = []
                visuals = [] if None in visuals else self.flatten(visuals)

                messages = [{"role": "system", "content": self.system_prompt}] if self.system_prompt else []
                
                # Initialize user message content as a list
                user_content = []
                
                visual_content = get_visual_content(visuals)
                
                if self.place_visual_first:
                    user_content.extend(visual_content)
                    if contexts:
                        user_content.append({"type": "text", "text": contexts})
                else:
                    if contexts:
                        user_content.append({"type": "text", "text": contexts})
                    user_content.extend(visual_content)
                
                messages.append({"role": "user", "content": user_content})

                prompt = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                
                image_inputs, video_inputs, video_kwargs = process_vision_info(messages, return_video_kwargs=True)
                mm_data = {}
                if image_inputs is not None:
                    mm_data["image"] = image_inputs
                if video_inputs is not None:
                    mm_data["video"] = video_inputs
                mm_processor_kwargs = {
                    **video_kwargs,
                    "min_pixels": self.min_pixels,
                    "max_pixels": self.max_pixels,
                }
                inputs.append(
                    {
                        "prompt": prompt,
                        "multi_modal_data": mm_data if len(mm_data) > 0 else None,
                        "mm_processor_kwargs": mm_processor_kwargs if len(mm_data) > 0 else None,
                    }
                )

            if self.rank == 0 and len(inputs) > 0:
                demo_input = deepcopy(inputs[0])
                print(f"Demo input: {demo_input}")

            sample_params = {}
            if "max_new_tokens" in gen_kwargs:
                sample_params["max_tokens"] = gen_kwargs.pop("max_new_tokens")
            if "temperature" in gen_kwargs:
                sample_params["temperature"] = gen_kwargs.pop("temperature")
            if "top_p" in gen_kwargs:
                sample_params["top_p"] = int(gen_kwargs.pop("top_p"))
            if "top_k" in gen_kwargs:
                sample_params["top_k"] = int(gen_kwargs.pop("top_k"))
            if "repetition_penalty" in gen_kwargs:
                sample_params["repetition_penalty"] = gen_kwargs.pop("repetition_penalty")

            sampling_params = SamplingParams(**sample_params)
            if self.rank == 0:
                print(f"Sampling params: {sampling_params}")

            model_outputs = self.client.generate(inputs, sampling_params=sampling_params)

            response_text = [output.outputs[0].text for output in model_outputs]
            response_text = [self.extract_answer(text) for text in response_text]

            assert len(response_text) == len(batch_requests)
            res.extend(response_text)
            pbar.update(len(batch_requests))

        pbar.close()
        return res

    def loglikelihood(self, requests: List[Instance]) -> List[tuple]:
        """Loglikelihood not supported for VLLM."""
        raise NotImplementedError("loglikelihood not supported for VLLM")

    def generate_until_multi_round(self, requests) -> List[str]:
        """Multi-round generation not implemented."""
        raise NotImplementedError("TODO: Implement multi-round generation")


def run_evaluation(
    model_args: dict,
    tasks: List[str],
    batch_size: int = 1,
    output_path: str = None,
    log_file: str = None,
    limit: int = None,
    log_samples: bool = False,
    system_prompt: str = None,
):
    """Run evaluation on specified tasks."""
    from lmms_eval import evaluator
    from lmms_eval.api.registry import get_model
    
    if log_file:
        eval_logger.add(log_file)
    
    model = get_model("qwen2_5_vl_vllm")(**model_args)
    
    # Use simple_evaluate instead of evaluate (which requires task_dict)
    # Pass system_instruction for custom system prompt
    results = evaluator.simple_evaluate(
        model=model,
        model_args={"batch_size": batch_size},
        tasks=tasks,
        limit=limit,
        log_samples=log_samples,
        system_instruction=system_prompt if system_prompt else "You are a helpful assistant.",
    )
    
    if output_path:
        import json
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        # Custom JSON serializer to handle non-serializable objects
        def json_serializer(obj):
            """Custom serializer for objects that aren't serializable by default json code"""
            if callable(obj):
                return f"<function {obj.__name__}>"
            elif hasattr(obj, '__dict__'):
                return str(obj)
            else:
                return str(obj)
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2, ensure_ascii=False, default=json_serializer)
        print(f"Results saved to: {output_path}")
    
    return results


def main():
    """Main function to run evaluations."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Qwen2.5-VL Evaluation using vLLM")
    parser.add_argument("--model_path", type=str, 
                        default="Qwen/Qwen2.5-VL-7B-Instruct",
                        help="Path to model (HuggingFace ID or local path)")
    parser.add_argument("--max_pixels", type=int, default=12845056,
                        help="Maximum number of pixels for image processing")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Batch size for evaluation")
    parser.add_argument("--tensor_parallel_size", type=int, default=1,
                        help="Tensor parallel size for vLLM")
    parser.add_argument("--num_processes", type=int, default=1,
                        help="Number of processes for distributed evaluation")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.8,
                        help="GPU memory utilization ratio")
    parser.add_argument("--tasks", type=str, default="mathvista",
                        help="Tasks to evaluate (comma-separated)")
    parser.add_argument("--output_dir", type=str, 
                        default="eval_results/qwen25vl_vllm",
                        help="Output directory for results")
    parser.add_argument("--interleave_visuals", action="store_true", default=False,
                        help="Whether to interleave visuals (required for MMMU)")
    parser.add_argument("--system_prompt", type=str, default="You are a helpful assistant.",
                        help="System prompt (or path to file containing the prompt)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit the number of samples to evaluate (for testing)")
    parser.add_argument("--log_samples", action="store_true", default=False,
                        help="Whether to log model outputs to file")
    parser.add_argument("--hf_home", type=str, default=None,
                        help="HuggingFace cache directory (default: ~/.cache/huggingface)")
    
    args = parser.parse_args()
    
    # Single GPU mode - always main process
    is_main_process = True
    
    # Setup HuggingFace cache directory
    if args.hf_home:
        os.environ["HF_HOME"] = args.hf_home
        os.environ["HUGGINGFACE_HUB_CACHE"] = args.hf_home
        os.environ["HF_DATASETS_CACHE"] = args.hf_home
    
    # Create output directory with model_path and task suffix for differentiation
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # Replace / with - in model path to create a safe directory name, strip leading -
    model_name = args.model_path.rstrip('/').replace('/', '-').replace('\\', '-').lstrip('-')
    # Use first task as suffix (or 'all' if multiple)
    task_suffix = args.tasks.split(',')[0] if ',' not in args.tasks else 'multi'
    
    # Check if output_dir already contains a full path (has path separator and doesn't end with _)
    # If shell script provides a complete path like "eval_results/..._timestamp", use it as-is
    # Otherwise, construct the full path with timestamp
    if '/' in args.output_dir or '\\' in args.output_dir:
        # It's a full path, check if it ends with underscore (indicating incomplete construction)
        if args.output_dir.endswith('_'):
            output_dir = f"{args.output_dir}{model_name}_{task_suffix}_{timestamp}"
        else:
            # Use the provided path directly, shell script should have created the full path
            output_dir = args.output_dir
    else:
        # Just a directory name, construct full path
        output_dir = f"{args.output_dir}_{model_name}_{task_suffix}_{timestamp}"
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Setup log file
    log_file = os.path.join(output_dir, "eval.log")
    
    # Model arguments
    model_args = {
        "model_version": args.model_path,
        "max_pixels": args.max_pixels,
        "batch_size": args.batch_size,
        "tensor_parallel_size": args.tensor_parallel_size,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "interleave_visuals": args.interleave_visuals,
        "system_prompt": args.system_prompt,
    }
    
    # Parse tasks (can be single or comma-separated multiple)
    task_list = [t.strip() for t in args.tasks.split(",")]
    
    # Only print from main process
    print("=" * 50)
    print("Qwen2.5-VL Evaluation (vLLM)")
    print("=" * 50)
    print(f"Model: {args.model_path}")
    print(f"HF Home: {args.hf_home or '~/.cache/huggingface'}")
    print(f"Tasks: {task_list}")
    print(f"System Prompt: {args.system_prompt[:50]}..." if len(args.system_prompt) > 50 else f"System Prompt: {args.system_prompt}")
    print(f"Output: {output_dir}")
    print(f"Log: {log_file}")
    print("=" * 50)
    
    # Run evaluation
    results = run_evaluation(
        model_args=model_args,
        tasks=task_list,
        batch_size=args.batch_size,
        output_path=os.path.join(output_dir, "results.json"),
        log_file=log_file,
        limit=args.limit,
        log_samples=args.log_samples,
        system_prompt=args.system_prompt,
    )
    
    print("\n" + "=" * 50)
    print("Evaluation completed!")
    print("=" * 50)
    
    # Print summary results
    if results and 'results' in results:
        print("\nResults Summary:")
        for task_name, task_results in results['results'].items():
            print(f"\n{task_name}:")
            if isinstance(task_results, dict):
                for metric, value in task_results.items():
                    if isinstance(value, (int, float)):
                        print(f"  {metric}: {value:.4f}" if isinstance(value, float) else f"  {metric}: {value}")
    print("")


if __name__ == "__main__":
    main()
