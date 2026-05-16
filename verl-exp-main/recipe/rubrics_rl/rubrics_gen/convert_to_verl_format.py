# -*- coding: utf-8 -*-
"""Convert rubrics generation pipeline output to verl format.

This script converts the JSONL output from pipeline.py to the parquet format
required by verl for RLHF training.
"""

import argparse
import json
import os
from typing import Any

import numpy as np
import pandas as pd


def load_image_as_bytes(image_path: str) -> dict[str, bytes]:
    """Load image from file path and convert to bytes format.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        Dictionary with "bytes" key containing image bytes
    """
    try:
        with open(image_path, "rb") as f:
            image_bytes = f.read()
        return {"bytes": image_bytes}
    except Exception as e:
        print(f"Warning: Failed to load image {image_path}: {e}")
        return None


def convert_sample_to_verl_format(
    sample: dict[str, Any],
    data_source: str = "rubrics_rl",
    image_base_path: str | None = None,
) -> dict[str, Any] | None:
    """Convert a single sample from pipeline output to verl format.
    
    Args:
        sample: Sample dictionary from pipeline output JSONL
        data_source: Data source identifier for verl
        image_base_path: Base path for resolving relative image paths
        
    Returns:
        Dictionary in verl format, or None if sample should be filtered out
    """
    # Extract fields from pipeline output
    question = sample.get("question", "")
    answer = sample.get("answer", "")
    image_paths = sample.get("image", [])
    rubrics_str = sample.get("rubrics", "")
    
    # Normalize image_paths to list
    if isinstance(image_paths, str):
        image_paths = [image_paths]
    elif image_paths is None:
        image_paths = []
    
    # Filter: Only keep samples with exactly 1 image_path (cannot be remedied)
    if len(image_paths) != 1:
        return None
    
    # Count <image> tags in question
    image_tag_count = question.count("<image>")
    
    # Filter: Only keep samples with 0 or 1 <image> tags
    # If 0, add <image>\n at the beginning (remediation allowed)
    # If > 1, filter out (cannot be remedied)
    if image_tag_count == 0:
        # Remediation: Add <image>\n at the beginning
        question = "<image>\n" + question
    elif image_tag_count > 1:
        # Cannot be remedied, filter out
        return None
    # If image_tag_count == 1, keep as is
    
    # Parse rubrics if it's a JSON string
    rubrics = None
    if rubrics_str:
        try:
            if isinstance(rubrics_str, str):
                rubrics = json.loads(rubrics_str)
            else:
                rubrics = rubrics_str
        except json.JSONDecodeError as e:
            print(f"Warning: Failed to parse rubrics for sample {sample.get('id', 'unknown')}: {e}")
            rubrics = None
    
    # Build prompt in chat format
    # For verl, we need to format the prompt as a list of messages
    # The question should be in user role
    prompt = [{"role": "user", "content": question}]
    
    # Process images (we know image_paths has exactly 1 element)
    images = []
    img_path = image_paths[0]
    
    # Resolve image path
    if image_base_path and not os.path.isabs(img_path):
        full_img_path = os.path.join(image_base_path, img_path)
    else:
        full_img_path = img_path
    
    # Load image as bytes
    img_bytes_dict = load_image_as_bytes(full_img_path)
    if img_bytes_dict is not None:
        images.append(img_bytes_dict)
    else:
        # If image cannot be loaded, filter out the sample
        return None
    
    # Build extra_info with only necessary fields
    # RubricsRLHFDataset requires: prompt and rubrics
    # Note: question may have been modified (e.g., <image> tag added)
    extra_info = {
        "prompt": question,  # Question text (may include <image> tag, required by compute_score)
        "rubrics": rubrics,  # Rubrics list (required by compute_score)
    }
    
    # Build verl format data
    verl_sample = {
        "data_source": data_source,
        "prompt": prompt,
        "reward_model": {
            "style": "rule",
            "ground_truth": answer,
        },
        "extra_info": extra_info,
        "images": images,
    }
    
    # Add rubrics at top level as JSON string for RubricsRLHFDataset compatibility
    # RubricsRLHFDataset expects item["rubrics"] to be a JSON string
    if rubrics_str:
        if isinstance(rubrics_str, str):
            verl_sample["rubrics"] = rubrics_str
        else:
            verl_sample["rubrics"] = json.dumps(rubrics_str, ensure_ascii=False)
    
    # Add ability field if available
    if "ability" in sample:
        verl_sample["ability"] = sample["ability"]
    
    return verl_sample


def convert_jsonl_to_verl_parquet(
    input_jsonl: str,
    output_parquet: str,
    data_source: str = "rubrics_rl",
    image_base_path: str | None = None,
    train_val_split: float | None = None,
    seed: int = 42,
) -> None:
    """Convert JSONL file from pipeline output to verl parquet format.
    
    Args:
        input_jsonl: Path to input JSONL file from pipeline
        output_parquet: Path to output parquet file (or base path if split)
        data_source: Data source identifier for verl
        image_base_path: Base path for resolving relative image paths
        train_val_split: Ratio for train/val split (e.g., 0.9 means 90% train, 10% val).
                        If None, output single file.
        seed: Random seed for train/val split
    """
    print(f"Reading JSONL file: {input_jsonl}")
    
    # Read JSONL file
    samples = []
    with open(input_jsonl, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                sample = json.loads(line)
                samples.append(sample)
            except json.JSONDecodeError as e:
                print(f"Warning: Failed to parse line {line_num}: {e}")
                continue
    
    print(f"Loaded {len(samples)} samples from JSONL file")
    
    # Convert samples to verl format
    verl_samples = []
    filtered_count = 0
    for idx, sample in enumerate(samples):
        try:
            verl_sample = convert_sample_to_verl_format(
                sample,
                data_source=data_source,
                image_base_path=image_base_path,
            )
            if verl_sample is None:
                # Sample was filtered out (e.g., wrong number of images)
                filtered_count += 1
                continue
            verl_samples.append(verl_sample)
        except Exception as e:
            print(f"Warning: Failed to convert sample {idx} (id: {sample.get('id', 'unknown')}): {e}")
            filtered_count += 1
            continue
    
    if filtered_count > 0:
        print(f"Filtered out {filtered_count} samples (image_paths != 1 or <image> tags != 0 or 1)")
    
    print(f"Converted {len(verl_samples)} samples to verl format")
    
    # Print a sample (without images) for inspection
    if verl_samples:
        sample_for_display = verl_samples[0].copy()
        # Replace images with placeholder to avoid printing large byte data
        if "images" in sample_for_display:
            image_count = len(sample_for_display["images"])
            sample_for_display["images"] = f"<{image_count} image(s) with bytes data>"
        print("\n" + "=" * 80)
        print("Sample output (first sample, images replaced with placeholder):")
        print("=" * 80)
        print(json.dumps(sample_for_display, ensure_ascii=False, indent=2))
        print("=" * 80 + "\n")
    
    # Ensure output directory exists
    output_dir = os.path.dirname(output_parquet)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Use default split ratio if not specified
    if train_val_split is None:
        train_val_split = 0.9
    
    # Always save full dataset
    full_df = pd.DataFrame(verl_samples)
    print(f"\nSaving full dataset to: {output_parquet}")
    full_df.to_parquet(output_parquet, index=False)
    print(f"Successfully saved {len(verl_samples)} samples (full dataset)")
    
    # Split into train and val
    # Shuffle data with fixed seed for reproducibility
    rng = np.random.default_rng(seed)
    indices = np.arange(len(verl_samples))
    rng.shuffle(indices)
    
    # Split indices
    split_idx = int(len(verl_samples) * train_val_split)
    train_indices = indices[:split_idx]
    val_indices = indices[split_idx:]
    
    # Split samples
    train_samples = [verl_samples[i] for i in train_indices]
    val_samples = [verl_samples[i] for i in val_indices]
    
    print(f"\nSplitting data: {len(train_samples)} train, {len(val_samples)} val")
    
    # Create DataFrames
    train_df = pd.DataFrame(train_samples)
    val_df = pd.DataFrame(val_samples)
    
    # Determine output paths for train/val
    base_path = output_parquet.replace(".parquet", "")
    train_path = f"{base_path}_train.parquet"
    val_path = f"{base_path}_val.parquet"
    
    # Save train and val files
    print(f"Saving train set to: {train_path}")
    train_df.to_parquet(train_path, index=False)
    print(f"Successfully saved {len(train_samples)} train samples")
    
    print(f"Saving val set to: {val_path}")
    val_df.to_parquet(val_path, index=False)
    print(f"Successfully saved {len(val_samples)} val samples")


def build_arg_parser() -> argparse.ArgumentParser:
    """Build argument parser."""
    parser = argparse.ArgumentParser(
        description="Convert rubrics generation pipeline output to verl format"
    )
    parser.add_argument(
        "--input_jsonl",
        type=str,
        required=True,
        help="Input JSONL file from pipeline (e.g., rubrics_filtered_out)",
    )
    parser.add_argument(
        "--output_parquet",
        type=str,
        required=True,
        help="Output parquet file path for verl",
    )
    parser.add_argument(
        "--data_source",
        type=str,
        default="rubrics_rl",
        help="Data source identifier (default: rubrics_rl)",
    )
    parser.add_argument(
        "--image_base_path",
        type=str,
        default=None,
        help="Base path for resolving relative image paths",
    )
    parser.add_argument(
        "--train_val_split",
        type=float,
        default=0.9,
        help="Ratio for train/val split (e.g., 0.9 means 90%% train, 10%% val). "
             "Default: 0.9. Always outputs three files: "
             "<output_parquet> (full), <output_parquet>_train.parquet, and <output_parquet>_val.parquet",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for train/val split (default: 42)",
    )
    return parser


if __name__ == "__main__":
    args = build_arg_parser().parse_args()
    convert_jsonl_to_verl_parquet(
        input_jsonl=args.input_jsonl,
        output_parquet=args.output_parquet,
        data_source=args.data_source,
        image_base_path=args.image_base_path,
        train_val_split=args.train_val_split,
        seed=args.seed,
    )
