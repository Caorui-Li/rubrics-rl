"""
Download evaluation datasets from HuggingFace into dataset/ using HF mirror when enabled.
Run with main conda env after enabling mirror: source scripts/hf_mirror_on.sh
"""
import os
import sys

from datasets import load_dataset

# Full datasets only (no testmini/subset). load_kwargs: name=, split= for load_dataset.
# cache_subdir is relative to cwd, e.g. dataset/MathVista
DATASETS = [
    ("AI4Math/MathVista", "dataset/MathVista", {"split": "test"}),  # full test ~5.14k
    ("AI4Math/MathVerse", "dataset/MathVerse", {"name": "testmini"}),  # main eval set ~3.94k (no separate "test" on HF)
    ("MathLLMs/MathVision", "dataset/MathVision", {"split": "test"}),
    ("FanqingM/MMK12", "dataset/MMK12", {"split": "test"}),
    ("Hothan/OlympiadBench", "dataset/OlympiadBench", {"name": "OE_MM_maths_en_COMP"}),
    ("Hothan/OlympiadBench", "dataset/OlympiadBench", {"name": "OE_MM_physics_en_COMP"}),
    ("Hothan/OlympiadBench", "dataset/OlympiadBench", {"name": "OE_TO_maths_en_COMP"}),
    ("Hothan/OlympiadBench", "dataset/OlympiadBench", {"name": "OE_TO_physics_en_COMP"}),
]


def main(dataset_filter=None):
    cwd = os.getcwd()
    if os.path.basename(cwd) != "verl-exp":
        print("Warning: run from project root (verl-exp) so dataset/ is created there.", file=sys.stderr)

    hf_endpoint = os.environ.get("HF_ENDPOINT", "")
    if hf_endpoint:
        print(f"Using HF_ENDPOINT={hf_endpoint} for downloads.", flush=True)
    to_download = DATASETS
    if dataset_filter:
        want = dataset_filter.lower()
        to_download = [(r, c, k) for r, c, k in DATASETS if want in c.lower()]
        if not to_download:
            print(f"No dataset matching '{dataset_filter}'", file=sys.stderr)
            sys.exit(1)

    for repo_id, cache_subdir, kwargs in to_download:
        cache_dir = os.path.join(cwd, cache_subdir)
        os.makedirs(cache_dir, exist_ok=True)
        label = kwargs.get("name") or kwargs.get("split", "")
        name = f"{repo_id} ({label})"
        print(f"Downloading {name} -> {cache_dir} ...", flush=True)
        try:
            load_dataset(repo_id, cache_dir=cache_dir, **kwargs)
            print(f"  Done: {name}", flush=True)
        except Exception as e:
            print(f"  Failed: {name}: {e}", file=sys.stderr, flush=True)
            raise

    print("All datasets downloaded to dataset/.", flush=True)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Download evaluation datasets to dataset/ (use HF mirror via hf_mirror_on)")
    parser.add_argument("--dataset", type=str, default=None, help="Optional: download only one (e.g. MathVista, MathVerse, MathVision, MMK12, OlympiadBench)")
    args = parser.parse_args()
    main(dataset_filter=args.dataset)
