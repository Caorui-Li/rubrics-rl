# -*- coding: utf-8 -*-
"""Core dataset base and vLLM runner for rubrics generation."""
from __future__ import annotations

import json
import os
import time
from copy import deepcopy
from typing import Any, Callable, Optional

import datasets
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from vllm import LLM, SamplingParams


def save_batch_into_jsonl_file(results: list[dict[str, Any]], save_path: str) -> None:
    """Save batch results to JSONL file."""
    save_results = deepcopy(results)
    for result in save_results:
        for key in list(result.keys()):
            if key in {"vllm_inputs", "messages"}:
                result.pop(key, None)
                continue
            try:
                json.dumps(result[key])
            except TypeError:
                result.pop(key, None)

    # Ensure the directory exists before writing
    dir_path = os.path.dirname(save_path)
    if dir_path:
        os.makedirs(dir_path, exist_ok=True)

    with open(save_path, "a", encoding="utf-8") as handle:
        for result in save_results:
            handle.write(json.dumps(result, ensure_ascii=False) + "\n")


def time_cost(func):
    """Decorator to measure execution time."""

    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"Time cost: {end_time - start_time:.2f}s for {func.__name__}")
        return result

    return wrapper


@time_cost
def run_vllm(
    dataloader: DataLoader,
    llm: LLM,
    sampling_params: SamplingParams,
    save_path: str,
    run_name: str = "Generation",
    post_process_output: Callable[[dict[str, Any], str], dict[str, Any]] = lambda x, y: x,
    post_filter_func: Optional[Callable[[dict[str, Any], str], bool]] = None,
    pre_filter_func: Optional[Callable[[dict[str, Any]], bool]] = None,
    filtered_save_path: Optional[str] = None,
    keep_raw: bool = True,
):
    """
    1. Each item must have an 'id' field.
    2. Each item must include 'vllm_inputs' for vLLM generate.
    """
    generation_results: list[dict[str, Any]] = []
    filter_results: list[dict[str, Any]] = []
    total_samples = 0
    successful_generation = 0
    pre_filter_kept = 0

    if filtered_save_path is None:
        filtered_save_path = save_path.replace(".jsonl", "_filter.jsonl")

    if not keep_raw:
        for path in (save_path, filtered_save_path):
            if path and os.path.exists(path):
                os.remove(path)

    if keep_raw and os.path.exists(save_path):
        with open(save_path, "r", encoding="utf-8") as handle:
            generation_results = [json.loads(line) for line in handle]
        print(f"Resume from {save_path}, total {len(generation_results)} results")
        if post_filter_func is not None and os.path.exists(filtered_save_path):
            with open(filtered_save_path, "r", encoding="utf-8") as handle:
                filter_results = [json.loads(line) for line in handle]

    id_set = {str(item["id"]) for item in generation_results if "id" in item}
    if hasattr(dataloader.dataset, "data"):
        dataloader.dataset.data = [
            item for item in dataloader.dataset.data if str(item.get("id")) not in id_set
        ]
        if hasattr(dataloader.dataset, "get_rank_split"):
            dataloader.dataset.get_rank_split()

    if not getattr(dataloader.dataset, "data", None):
        print(f"Run VLLM {run_name} completed, skip...")
        return generation_results, filter_results

    bar = tqdm(
        dataloader,
        desc=f"Running VLLM: {run_name}",
        total=len(dataloader),
        ncols=100,
    )

    for _, batch in enumerate(bar):
        if pre_filter_func is not None:
            batch = list(filter(pre_filter_func, batch))
        pre_filter_kept += len(batch)
        vllm_inputs = [item.pop("vllm_inputs") for item in batch]
        outputs = llm.generate(vllm_inputs, sampling_params=sampling_params)
        outputs = [output.outputs[0].text for output in outputs]

        filter_batch: list[dict[str, Any]] = []
        for idx, (item, output) in enumerate(zip(batch, outputs)):
            updated = post_process_output(item, output)
            if updated is not item:
                batch[idx] = updated
            generation_results.append(updated)
            if post_filter_func is not None and not post_filter_func(updated, output):
                continue
            filter_batch.append(updated)

        total_samples += len(batch)
        successful_generation += len(filter_batch)
        filter_results.extend(filter_batch)

        if keep_raw:
            save_batch_into_jsonl_file(batch, save_path)
        save_batch_into_jsonl_file(filter_batch, filtered_save_path)

        bar.set_postfix(
            {
                "post_filter": (
                    f"{successful_generation/total_samples:.2%}" if total_samples > 0 else "0.00%"
                ),
                "successful": successful_generation,
                "total": total_samples,
                "pre_filter_kept": pre_filter_kept,
            }
        )

    ratio = len(filter_results) / len(generation_results) if generation_results else 0.0
    print(
        f"Processed {len(filter_results)} valid samples, "
        f"Total samples {len(generation_results)}, "
        f"Ratio: {ratio:.2%}"
    )
    return generation_results, filter_results


class DistributeDataset(Dataset):
    def __init__(self, rank: int, world_size: int, *args, **kwargs):
        self.rank = rank
        self.world_size = world_size

        self.data: list[dict[str, Any]] | datasets.Dataset = []
        self.data = self._load_data(*args, **kwargs)
        self._ensure_id()
        print("raw data length:", len(self.data))

        print("Filtering data...")
        if isinstance(self.data, list):
            self.data = list(filter(self.filter_data, self.data))
        elif isinstance(self.data, datasets.Dataset):
            self.data = self.data.filter(self.filter_data, num_proc=1)
        else:
            raise ValueError("data error")

        assert self.check_data(), "data has duplicate IDs"
        print("filtered data length:", len(self.data))

        self.get_rank_split()
        print(f"Rank {self.rank} split data length:", len(self.cur_rank_idx_list))

    def _ensure_id(self) -> None:
        if isinstance(self.data, list):
            for idx, item in enumerate(self.data):
                if "id" not in item:
                    if "qid" in item and item["qid"] is not None:
                        item["id"] = str(item["qid"])
                    else:
                        item["id"] = str(idx)
                else:
                    item["id"] = str(item["id"])
        elif isinstance(self.data, datasets.Dataset):
            if "id" not in self.data.column_names:
                self.data = self.data.map(lambda item, idx: {"id": str(idx)}, with_indices=True)
            else:
                self.data = self.data.map(lambda item: {"id": str(item["id"])})
        else:
            raise ValueError("data type error")

    def _load_data(self, *args, **kwargs) -> list[dict[str, Any]]:
        raise NotImplementedError("This method should be implemented in subclasses")

    def __len__(self):
        return len(self.cur_rank_idx_list)

    def __getitem__(self, idx):
        cidx = self.cur_rank_idx_list[idx]
        try:
            item = deepcopy(self.data[cidx])
            return item
        except Exception as exc:
            print(f"{exc}, {cidx}, {len(self.data)}")
            raise

    def filter_data(self, item: dict[str, Any]) -> bool:
        return True

    def get_rank_split(self):
        self.cur_rank_idx_list = [i for i in range(len(self.data)) if i % self.world_size == self.rank]

    def _create_message(self, *args, **kwargs):
        raise NotImplementedError("This method should be implemented in subclasses")

    def check_data(self) -> bool:
        if isinstance(self.data, list):
            id_set = {str(item["id"]) for item in self.data}
        elif isinstance(self.data, datasets.Dataset):
            id_set = set(self.data["id"])
        else:
            raise ValueError("data type error")

        if len(id_set) != len(self.data):
            print("Duplicate IDs found in the dataset")
            return False
        return True
