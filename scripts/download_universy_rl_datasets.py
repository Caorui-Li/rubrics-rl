#!/usr/bin/env python3
"""Download public datasets and related repos for the universy RL survey."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

from huggingface_hub import snapshot_download
import gdown


ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "datasets" / "universy_rl_dataset"

HF_DATASETS = [
    "osunlp/TravelPlanner",
    "rubricreward/R3-full-dataset",
    "inclusionAI/ASearcher-train-data",
    "openai/healthbench",
    "OpenRubrics/OpenRubrics",
    "OpenRubrics/OpenRubric-v2",
    "Gen-Verse/Open-AgentRL-SFT-3K",
    "Gen-Verse/Open-AgentRL-30K",
    "Gen-Verse/Open-AgentRL-Eval",
    "rico2512/RubricHub_v1",
    "yikeee/rubrichub-sft-judgment-gen",
    "MiniByte-666/Dr.SCI",
]

GIT_REPOS = {
    "TravelPlanner": "https://github.com/OSU-NLP-Group/TravelPlanner.git",
    "ASearcher": "https://github.com/inclusionAI/ASearcher.git",
    "Open-AgentRL": "https://github.com/Gen-Verse/Open-AgentRL.git",
}

EXTERNAL_FILES = [
    {
        "name": "travelplanner_database",
        "url": "https://drive.google.com/uc?id=1pF1Sw6pBmq2sFkJvm-LzJOqrmfWoQgxE",
        "output": "travelplanner_database.zip",
    }
]


def run(cmd: list[str], cwd: Path | None = None) -> None:
    subprocess.run(cmd, cwd=str(cwd) if cwd else None, check=True)


def clone_repo(name: str, url: str, repos_dir: Path) -> str:
    target = repos_dir / name
    if target.exists():
        run(["git", "-C", str(target), "pull", "--ff-only"])
        return "updated"
    run(["git", "clone", "--depth", "1", url, str(target)])
    return "cloned"


def download_hf_dataset(repo_id: str, hf_dir: Path) -> str:
    local_dir = hf_dir / repo_id.replace("/", "__")
    if local_dir.exists() and any(local_dir.iterdir()):
        return "exists"
    snapshot_download(repo_id=repo_id, repo_type="dataset", local_dir=str(local_dir))
    return "downloaded"


def download_external(entry: dict[str, str], ext_dir: Path) -> str:
    output_path = ext_dir / entry["output"]
    if output_path.exists():
        return "exists"
    gdown.download(entry["url"], str(output_path), quiet=False, fuzzy=True)
    return "downloaded"


def main() -> None:
    hf_dir = TARGET / "huggingface"
    repos_dir = TARGET / "repos"
    ext_dir = TARGET / "external"
    TARGET.mkdir(parents=True, exist_ok=True)
    hf_dir.mkdir(exist_ok=True)
    repos_dir.mkdir(exist_ok=True)
    ext_dir.mkdir(exist_ok=True)

    manifest: dict[str, dict[str, str]] = {
        "huggingface": {},
        "repos": {},
        "external": {},
    }

    for repo_id in HF_DATASETS:
        manifest["huggingface"][repo_id] = download_hf_dataset(repo_id, hf_dir)

    for name, url in GIT_REPOS.items():
        manifest["repos"][name] = clone_repo(name, url, repos_dir)

    for entry in EXTERNAL_FILES:
        manifest["external"][entry["name"]] = download_external(entry, ext_dir)

    (TARGET / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
