"""
Simple reward + dataset adapter for ViRL39K without rubrics.
Uses format_reward + accuracy_reward, and maps question/answer/image fields.
"""

import logging
import os
import re

from latex2sympy2_extended import NormalizationConfig
from math_verify import LatexExtractionConfig, parse, verify

from verl.utils.dataset.rl_dataset import RLHFDataset

logger = logging.getLogger(__name__)


DEFAULT_SYSTEM_PROMPT = (
    "Solve the question. The user asks a question, and you solves it. "
    "You first thinks about the reasoning process in the mind and then provides the user with the answer. "
    "The answer is in latex format and wrapped in $...$. The final answer must be wrapped using the "
    "\\\\boxed{} command. The reasoning process and answer are enclosed within <think> </think> and "
    "<answer> </answer> tags, respectively, i.e., <think> Since $1+1=2$, so the answer is $2$. "
    "<answer> The answer is $\\\\boxed{2}$ </answer>, which means assistant's output should start with "
    "<think> and end with </answer>."
)


def format_reward(response: str) -> float:
    format_pattern = r"^<think>.*?</think>\s*<answer>.*?</answer>\s*$"
    think_count = response.count("<think>")
    answer_count = response.count("<answer>")
    is_valid = bool(re.match(format_pattern, response, re.DOTALL)) and think_count == 1 and answer_count == 1
    return 0.5 if is_valid else 0.0


def accuracy_reward(response: str, answer: str) -> float:
    if answer is None:
        return 0.0
    if not answer.startswith("$"):
        answer = "$" + answer + "$"

    gold_parsed = parse(
        answer,
        extraction_mode="first_match",
        extraction_config=[LatexExtractionConfig()],
        parsing_timeout=None,
    )
    if len(gold_parsed) == 0:
        print("Failed to parse gold solution: ", answer)
        return 1.0

    answer_parsed = parse(
        response,
        extraction_config=[
            LatexExtractionConfig(
                normalization_config=NormalizationConfig(
                    nits=False,
                    malformed_operators=False,
                    basic_latex=True,
                    boxed="all",
                    units=True,
                ),
                boxed_match_priority=0,
                try_extract_without_anchor=False,
            )
        ],
        extraction_mode="first_match",
        parsing_timeout=None,
    )
    try:
        reward = float(verify(answer_parsed, gold_parsed, timeout_seconds=None))
    except Exception as e:
        print("Failed to verify: ", e)
        reward = 0.0
    return reward


def compute_score(data_source: str, solution_str: str, ground_truth: str, extra_info=None) -> dict:
    format_score = format_reward(solution_str)
    if isinstance(ground_truth, list):
        ground_truth = ground_truth[0]
    accuracy_score = accuracy_reward(solution_str, ground_truth)
    total_score = format_score + accuracy_score
    return {
        "score": total_score,
        "format_reward": format_score,
        "accuracy_reward": accuracy_score,
    }


class ViRL39KSimpleDataset(RLHFDataset):
    """Dataset adapter for ViRL39K parquet with question/answer/image fields."""

    def _build_messages(self, example: dict):
        image_root = self.config.get("image_root") or os.environ.get("VIRL39K_IMAGE_ROOT")
        # Normalize image field to list[dict] for fetch_image.
        if self.image_key in example and example[self.image_key] is not None:
            images = example[self.image_key]
            if hasattr(images, "tolist"):
                images = images.tolist()
            if isinstance(images, str):
                images = [images]
            if isinstance(images, list):
                normalized = []
                for img in images:
                    if isinstance(img, str):
                        if image_root and not re.match(r"^(?:[a-zA-Z]+:|/)", img):
                            img = os.path.join(image_root, img)
                        normalized.append({"image": img})
                    else:
                        normalized.append(img)
                example[self.image_key] = normalized

        raw_prompt = example.pop(self.prompt_key)
        if isinstance(raw_prompt, list):
            messages = raw_prompt
        else:
            # Normalize multimodal placeholders to vLLM-expected "<image>" / "<video>"
            if isinstance(raw_prompt, str):
                # Convert <image1>, <image2>, ... to <image>
                raw_prompt = re.sub(r"<image\\d+>", "<image>", raw_prompt)
                # Convert <image1> variants like <image_1> if present
                raw_prompt = re.sub(r"<image[_\\-]?\\d+>", "<image>", raw_prompt)
                # If images exist but no placeholder, prepend one
                if (
                    self.image_key in example
                    and example[self.image_key]
                    and "<image>" not in raw_prompt
                ):
                    raw_prompt = "<image>\n" + raw_prompt
            messages = [{"role": "user", "content": raw_prompt}]

        if not any(msg.get("role") == "system" for msg in messages):
            messages.insert(0, {"role": "system", "content": DEFAULT_SYSTEM_PROMPT})

        # Convert <image>/<video> placeholders to structured content for processor.
        if self.image_key in example or self.video_key in example:
            for message in messages:
                content = message.get("content")
                if not isinstance(content, str):
                    continue
                content_list = []
                segments = re.split("(<image>|<video>)", content)
                segments = [item for item in segments if item != ""]
                for segment in segments:
                    if segment == "<image>":
                        content_list.append({"type": "image"})
                    elif segment == "<video>":
                        content_list.append({"type": "video"})
                    else:
                        content_list.append({"type": "text", "text": segment})
                message["content"] = content_list

        return messages

    def __getitem__(self, item):
        row_dict = super().__getitem__(item)

        if "data_source" not in row_dict or row_dict["data_source"] is None:
            row_dict["data_source"] = "virl39k"

        answer = row_dict.get("answer")
        reward_model = row_dict.get("reward_model") or {}
        reward_model.setdefault("style", "rule")
        if reward_model.get("ground_truth") is None:
            reward_model["ground_truth"] = answer
        row_dict["reward_model"] = reward_model

        if "extra_info" not in row_dict or row_dict["extra_info"] is None:
            row_dict["extra_info"] = {}
        if "prompt" not in row_dict["extra_info"]:
            row_dict["extra_info"]["prompt"] = row_dict.get("full_prompts") or row_dict.get("raw_prompt")

        return row_dict
