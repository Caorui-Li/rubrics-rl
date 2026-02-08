"""
Shared LLM judge for evaluation: supports OpenAI (gpt-4o) and DeepSeek (OpenAI-compatible API).
Set OPENAI_API_KEY for OpenAI, or DEEPSEEK_API_KEY (and optionally DEEPSEEK_BASE_URL, DEEPSEEK_MODEL) for DeepSeek.
"""
import os
import logging

import openai

# DeepSeek judge (OpenAI-compatible API)
DEEPSEEK_MODEL = os.environ.get("DEEPSEEK_MODEL", "deepseek-chat")
DEEPSEEK_BASE_URL = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")

_openai_client = None
_deepseek_client = None


def _get_openai_client():
    global _openai_client
    if _openai_client is None and os.environ.get("OPENAI_API_KEY"):
        _openai_client = openai.OpenAI()
    return _openai_client


def _get_deepseek_client():
    global _deepseek_client
    if _deepseek_client is None and DEEPSEEK_API_KEY:
        _deepseek_client = openai.OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)
    return _deepseek_client


def get_judge_client_and_model(judge_model=None):
    """
    Return (client, model) for judge. Prefer DeepSeek if DEEPSEEK_API_KEY is set.
    judge_model: optional override (e.g. 'gpt-4o', 'deepseek-chat'). If None, use env.
    """
    model = judge_model or os.environ.get("JUDGE_MODEL", "")
    if model and ("deepseek" in model.lower() or "DeepSeek" in model):
        client = _get_deepseek_client()
        if client:
            return client, (model or DEEPSEEK_MODEL)
    if model and model != "deepseek":
        client = _get_openai_client()
        if client:
            return client, model
    # Default: DeepSeek if key set, else OpenAI
    client = _get_deepseek_client()
    if client:
        return client, DEEPSEEK_MODEL
    client = _get_openai_client()
    if client:
        return client, (model or "gpt-4o")
    return None, None


def get_chat_response(
    prompt,
    model=None,
    max_token=256,
    retry=5,
    temperature=None,
    judge_model=None,
):
    """
    Call LLM for judge (extract/score). Uses OpenAI or DeepSeek based on env/args.
    """
    client, effective_model = get_judge_client_and_model(judge_model or model)
    if client is None:
        logging.error(
            "No judge API key set. Set OPENAI_API_KEY or DEEPSEEK_API_KEY (and optionally DEEPSEEK_BASE_URL)."
        )
        return ""
    messages = [{"role": "user", "content": prompt}]
    for i in range(retry):
        try:
            temp = temperature if temperature is not None else 0.5 * i
            completion = client.chat.completions.create(
                model=effective_model,
                messages=messages,
                temperature=temp,
                max_tokens=max_token,
            )
            prediction = completion.choices[0].message.content.strip()
            if prediction != "" and prediction is not None:
                return prediction
        except Exception as e:
            logging.error(e)
    return ""
