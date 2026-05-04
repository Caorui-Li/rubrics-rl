import json
import logging
import os
import re
import time

import requests
from latex2sympy2_extended import NormalizationConfig
from math_verify import LatexExtractionConfig, parse, verify
from openai import APITimeoutError, OpenAI

from verl.utils.dataset.rl_dataset import RLHFDataset

# Default system prompt (same as in rubrics_rl.py)
DEFAULT_SYSTEM_PROMPT = (
    "Solve the question. The user asks a question, and you solves it. "
    "You first thinks about the reasoning process in the mind and then provides the user with the answer. "
    "The answer is in latex format and wrapped in $...$. The final answer must be wrapped using the "
    "\\\\boxed{} command. The reasoning process and answer are enclosed within <think> </think> and "
    "<answer> </answer> tags, respectively, i.e., <think> Since $1+1=2$, so the answer is $2$. "
    "<answer> The answer is $\\\\boxed{2}$ </answer>, which means assistant's output should start with "
    "<think> and end with </answer>."
)

"""
A example of rubrics:
```json
[
  {{
    "criterion": "The response explicitly identifies that the triangle in the image is a right-angled triangle.",
    "weight": 3,
    "id": 1
  }},
  {{
    "criterion": "The response uses the Pythagorean theorem to calculate the hypotenuse.",
    "weight": 2,
    "id": 2
  }},
  {{
    "criterion": "The final answer is stated to be 5.",
    "weight": 3,
    "id": 3
  }}
]
```
"""
logger = logging.getLogger(__name__)

openai_api_key = os.environ.get("JUDGE_MODEL_API_KEY", "EMPTY")
openai_api_base = os.environ.get("LLM_AS_A_JUDGE_BASE", "http://28.12.131.189:8000/v1")

# Lazily initialize client/model to avoid pickling SSLContext in multiprocessing.
_client = None
_model_name = None


def _get_client_and_model():
    global _client, _model_name
    if _client is None:
        _client = OpenAI(
            api_key=openai_api_key,
            base_url=openai_api_base,
            timeout=300.0,
        )
    if _model_name is None:
        # Prefer explicit JUDGE_MODEL env var; fall back to auto-discovery via /models
        explicit = os.environ.get("JUDGE_MODEL", "").strip()
        if explicit:
            _model_name = explicit
        elif openai_api_base:
            try:
                response = requests.get(f"{openai_api_base}/models")
                response.raise_for_status()
                models = response.json()
                if models.get("data"):
                    _model_name = models["data"][0]["id"]
                else:
                    logger.warning("No models found at the specified API base for reward scoring.")
                    _model_name = ""
            except (requests.exceptions.RequestException, KeyError, IndexError) as e:
                logger.warning(f"Failed to get model from {openai_api_base}: {e}. Reward scoring will be disabled.")
                _model_name = ""
        else:
            _model_name = ""
    return _client, _model_name


def check_is_valid_verify_response(response_text: str) -> tuple[bool, list]:
    """
    Check if the response is a valid JSON array and parse it.
    Returns (is_valid, parsed_list).
    A valid response:
        - can be parsed as JSON
        - is a list
        - each entry has "id" and "is_satisfied" fields
        - "is_satisfied" is either "yes" or "no"
    """
    try:
        # Try to extract JSON from markdown code blocks if present
        if "```json" in response_text:
            response_text = response_text.split("```json")[1].split("```")[0].strip()
        elif "```" in response_text:
            response_text = response_text.split("```")[1].split("```")[0].strip()

        parsed = json.loads(response_text)

        if not isinstance(parsed, list):
            print(f"Invalid response format: not a list for {response_text}")
            return False, []

        for item in parsed:
            if not isinstance(item, dict):
                print(f"Invalid response format: not a dict for {item}")
                return False, []
            if "id" not in item or "is_satisfied" not in item:
                print(f"Invalid response format: missing id or is_satisfied for {item}")
                return False, []
            if item["is_satisfied"] not in ["yes", "no"]:
                print(f"Invalid response format: invalid is_satisfied for {item}")
                return False, []

            # Normalize id to string for stable comparison
            item["id"] = str(item["id"])

        return True, parsed
    except (json.JSONDecodeError, ValueError, AttributeError):
        print(f"Invalid response format: JSON decode error for {response_text}")
        return False, []


def merge_rubrics_with_verification(original_rubrics: list, verified_list: list) -> list:
    """
    Merge original rubrics with verification results.

    Args:
        original_rubrics: List of original rubric dicts with "id", "criterion", "weight"
        verified_list: List of verification results with "id" and "is_satisfied"

    Returns:
        List of merged rubric dicts with "criterion", "weight", "id", "is_satisfied"
    """
    rubric_dict = {str(r["id"]): r for r in original_rubrics}
    result = []

    for verified_item in verified_list:
        rubric_id = str(verified_item["id"])
        if rubric_id not in rubric_dict:
            raise ValueError(f"Rubric id {rubric_id} not found in original rubrics")

        original_rubric = rubric_dict[rubric_id]
        result.append(
            {
                "criterion": original_rubric["criterion"],
                "weight": original_rubric["weight"],
                "id": rubric_id,
                "is_satisfied": verified_item["is_satisfied"].lower() == "yes",
            }
        )

    return result


def verify_rubric(rubrics: list, solution_str: str, prompt: str, max_retries: int = 5) -> list:
    """
    Use LLM to verify if the solution satisfies these rubrics.
    For each rubric, return a dictionary with the following keys:
    - "criterion": The criterion of the rubric.
    - "weight": The weight of the rubric.
    - "id": The id of the rubric.
    - "is_satisfied": Whether the solution satisfies the rubric (bool).
    
    Returns:
        list: List of verified rubrics, or None if all retries failed (fallback mechanism)
    """
    rubric_ids = {str(r["id"]) for r in rubrics}
    rubric_json = json.dumps(rubrics, indent=2)
    formatted_prompt = verify_prompt.format(prompt=prompt, response=solution_str, rubric=rubric_json)

    for attempt in range(max_retries):
        try:
            client, model_name = _get_client_and_model()
            response = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": formatted_prompt}],
                temperature=0.1,
            )

            response_text = response.choices[0].message.content.strip()
            is_valid, verified_list = check_is_valid_verify_response(response_text)

            if is_valid:
                # Check if all rubric ids are present
                verified_ids = {item["id"] for item in verified_list}
                if verified_ids != rubric_ids:
                    print(f"Rubric id mismatch (attempt {attempt + 1}/{max_retries}). "
                        f"Expected: {rubric_ids}, Got: {verified_ids}. Retrying...")
                    if attempt < max_retries - 1:
                        # Exponential backoff: 1s, 2s, 4s, ...
                        time.sleep(min(2 ** attempt, 10))
                        continue
                else:
                    return merge_rubrics_with_verification(rubrics, verified_list)
            else:
                print(f"Invalid response format (attempt {attempt + 1}/{max_retries}). Retrying...")
                if attempt < max_retries - 1:
                    # Exponential backoff: 1s, 2s, 4s, ...
                    time.sleep(min(2 ** attempt, 10))
                    continue
        except APITimeoutError as e:
            print(
                f"API timeout error (attempt {attempt + 1}/{max_retries}): {e}. Retrying with exponential backoff..."
            )
            if attempt < max_retries - 1:
                # Exponential backoff: 2s, 4s, 8s, ... (max 30s)
                backoff_time = min(2 ** (attempt + 1), 30)
                print(f"Waiting {backoff_time} seconds before retry...")
                time.sleep(backoff_time)
                continue
            else:
                logger.error(f"API timeout after {max_retries} retries. Using fallback mechanism.")
                # Fall through to fallback mechanism
        except (KeyError, AttributeError) as e:
            print(f"Failed to parse response (attempt {attempt + 1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                time.sleep(min(2 ** attempt, 10))
                continue
        except Exception as e:
            # Catch other API errors (rate limit, connection errors, etc.)
            print(f"API error (attempt {attempt + 1}/{max_retries}): {type(e).__name__}: {e}")
            if attempt < max_retries - 1:
                # Exponential backoff: 2s, 4s, 8s, ... (max 30s)
                backoff_time = min(2 ** (attempt + 1), 30)
                print(f"Waiting {backoff_time} seconds before retry...")
                time.sleep(backoff_time)
                continue
            else:
                print(f"API error after {max_retries} retries: {e}. Using fallback mechanism.")
                # Fall through to fallback mechanism

    # Fallback mechanism: If all retries failed, return None to indicate failure
    # The caller (compute_score) will handle this by returning a default score
    logger.error(
        f"Failed to get valid verification after {max_retries} retries. "
        f"Using fallback: returning None (will be treated as score 0.0)"
    )
    return None


def format_reward(response: str) -> float:
    """
    Verify if the response meets the format requirements and return reward.
    
    Args:
        response: The response string to verify
        format_pattern: Regex pattern to match the required format. 
                       Default pattern requires <think>...</think><answer>...</answer>
    
    Returns:
        float: 0.5 if format is correct, 0.0 otherwise
    """
    format_pattern = r"^<think>.*?</think>\s*<answer>.*?</answer>\s*$"
    
    think_count = response.count("<think>")
    answer_count = response.count("<answer>")
    
    is_valid = bool(re.match(format_pattern, response, re.DOTALL)) and think_count == 1 and answer_count == 1
    
    return 0.5 if is_valid else 0.0


def accuracy_reward(response: str, answer: str) -> float:
    """
    Verify if the response contains the correct mathematical answer.
    
    Args:
        response: The response string containing the answer
        answer: The ground truth answer (should be in LaTeX format, e.g., "$42$" or "42")
    
    Returns:
        float: 1.0 if answer is correct, 0.0 otherwise
    """
    # Ensure answer is wrapped in $ if not already
    if not answer.startswith("$"):
        answer = "$" + answer + "$"
    
    # Parse ground truth answer
    gold_parsed = parse(
        answer,
        extraction_mode="first_match",
        extraction_config=[LatexExtractionConfig()],
        parsing_timeout=None,  # Disable timeout to support multithreaded environment
    )
    
    if len(gold_parsed) == 0:
        # If the gold solution is not parseable, we reward 1.0 to skip this example
        print("Failed to parse gold solution: ", answer)
        return 1.0
    
    # Parse response answer with strict normalization
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
                # Ensures that boxed is tried first
                boxed_match_priority=0,
                try_extract_without_anchor=False,
            )
        ],
        extraction_mode="first_match",
        parsing_timeout=None,  # Disable timeout to support multithreaded environment
    )
    
    # Verify if the parsed answers match
    # Disable timeout to support multithreaded environment
    try:
        reward = float(verify(answer_parsed, gold_parsed, timeout_seconds=None))
    except Exception as e:
        print("Failed to verify: ", e)
        reward = 0.0
    
    return reward


def compute_score(data_source: str, solution_str: str, ground_truth: str, extra_info=None) -> dict:
    """
    Compute reward score for model solutions with robust handling of various formats.
    
    Returns:
        dict: Dictionary containing "score", "rubric_reward", "format_reward", and "accuracy_reward".
              All values are floats between 0.0 and 1.0.
    """
    rubrics = extra_info["rubrics"]
    if isinstance(rubrics, str):
        rubrics = json.loads(rubrics)

    prompt = extra_info["prompt"]
    
    # Compute format_reward and accuracy_reward first (used in all cases)
    format_score = format_reward(solution_str)
    if isinstance(ground_truth, list):
        ground_truth = ground_truth[0]
    accuracy_score = accuracy_reward(solution_str, ground_truth)
    
    verified_rubrics = verify_rubric(rubrics, solution_str, prompt)
    
    # Fallback mechanism: If verification failed (returned None), set rubric_reward to 0.0
    # but still compute format_reward and accuracy_reward normally
    if verified_rubrics is None:
        print(
            f"Rubric verification failed for solution. "
            f"Setting rubric_reward to 0.0. Solution preview: {solution_str[:100]}..."
        )
        rubric_score = 0.0
        total_score = rubric_score + format_score + accuracy_score
        return {
            "score": total_score,
            "rubric_reward": rubric_score,
            "format_reward": format_score,
            "accuracy_reward": accuracy_score,
        }

    rubric_reward = 0.0
    total_weight = 0.0
    for verified_rubric in verified_rubrics:
        weight = verified_rubric["weight"]
        is_satisfied = verified_rubric["is_satisfied"]
        rubric_reward += weight * (1 if is_satisfied else 0)
        total_weight += weight

    if total_weight == 0:
        logger.warning("Total weight is 0, returning 0.0 as score")
        rubric_score = 0.0
        total_score = rubric_score + format_score + accuracy_score
        return {
            "score": total_score,
            "rubric_reward": rubric_score,
            "format_reward": format_score,
            "accuracy_reward": accuracy_score,
        }
    
    rubric_score = rubric_reward / total_weight
    total_score = rubric_score + format_score + accuracy_score
    return {
        "score": total_score,
        "rubric_reward": rubric_score,
        "format_reward": format_score,
        "accuracy_reward": accuracy_score,
    }


class RubricsRLHFDataset(RLHFDataset):
    
    def _build_messages(self, example: dict):
        # Directly extract messages from example (same logic as RLHFDataset._build_messages)
        import re
        messages: list = example.pop(self.prompt_key)

        # Handle image/video content if present (same logic as RLHFDataset._build_messages)
        if self.image_key in example or self.video_key in example:
            for message in messages:
                content = message["content"]
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

        # Add custom system prompt directly to dataset
        SYSTEM_PROMPT = DEFAULT_SYSTEM_PROMPT

        # Check if system prompt already exists
        has_system = any(msg.get("role") == "system" for msg in messages)
        if not has_system:
            messages.insert(0, {"role": "system", "content": SYSTEM_PROMPT})
            
        # # Print final message for debugging
        # print("=" * 80)
        # print("[_BUILD_MESSAGES] Final Messages:")
        # for i, msg in enumerate(messages):
        #     print(f"  [{i}] role={msg.get('role')}, content={msg.get('content')}")
        # print("=" * 80)

        return messages


    def __getitem__(self, item):
        row_dict = super().__getitem__(item)

        rubrics_raw = row_dict.get("rubrics")
        if rubrics_raw is None and isinstance(row_dict.get("extra_info"), dict):
            rubrics_raw = row_dict["extra_info"].get("rubrics")

        rubrics = json.loads(rubrics_raw) if isinstance(rubrics_raw, str) else rubrics_raw

        if "extra_info" not in row_dict or row_dict["extra_info"] is None:
            row_dict["extra_info"] = {}
        if rubrics is not None:
            row_dict["extra_info"]["rubrics"] = rubrics

        # assert the necessary fields are present
        assert "rubrics" in row_dict["extra_info"], "rubrics not found in extra_info"
        assert "prompt" in row_dict["extra_info"], "prompt not found in extra_info"
        return row_dict


verify_prompt = """

### Prompt for Scoring Responses

You are a skilled judge who will be assessing the quality of LLM responses to a user prompt.

Given a user prompt, LLM response, and a rubric, your task is evaluating the performance of the model response by seeing whether or not it meets the rubric dimension.

Answer the each of the given rubric dimension in either "yes" or "no". Do not output any response other than "yes" or "no".

Keep in mind that you will be grading industry-leading LLMs. Make sure to have high expectation for grading the responses.

Make sure your evaluation is as objective and consistent as it could be. By consistent we mean that a different evaluator's assessment of the task should agree with yours.

Think carefully before you make the decision. After you make the decision, explicitly output which dimension receives "yes" and which dimension receives "no".

**Input:**

* **PROMPT:** {prompt}

* **RESPONSE:** {response}

* **RUBRIC:** {rubric}

Return ONLY the JSON array, no other text. For example:

```json
[
  {{
    "id": 1,
    "is_satisfied": "yes"
  }},
  {{
    "id": 2,
    "is_satisfied": "no"
  }},
  {{
    "id": 3,
    "is_satisfied": "yes"
  }}
  ......
]
```
"""
