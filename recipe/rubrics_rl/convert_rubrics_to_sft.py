import argparse
import json
import os
import re
from typing import Any, Dict, Iterable, List, Optional


def load_records(path: str) -> List[Dict[str, Any]]:
    if path.endswith('.jsonl'):
        records = []
        with open(path, 'r', encoding='utf-8') as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                records.append(json.loads(line))
        return records
    if path.endswith('.json'):
        with open(path, 'r', encoding='utf-8') as handle:
            data = json.load(handle)
        if not isinstance(data, list):
            raise ValueError(f'JSON file must contain a list: {path}')
        return data
    if path.endswith('.parquet'):
        import pandas as pd
        return pd.read_parquet(path).to_dict(orient='records')
    raise ValueError(f'Unsupported input format: {path}')


def dump_jsonl(records: Iterable[Dict[str, Any]], path: str) -> None:
    output_dir = os.path.dirname(os.path.abspath(path))
    os.makedirs(output_dir, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + '\n')


def normalize_images(value: Any, image_base_path: Optional[str]) -> List[Dict[str, Optional[str]]]:
    if value is None:
        return []
    if isinstance(value, str):
        value = [value]
    elif isinstance(value, dict):
        value = [value]
    elif not isinstance(value, list):
        return []

    images = []
    for item in value:
        if isinstance(item, dict):
            path = item.get('path')
            bytes_value = item.get('bytes')
            if path:
                path = resolve_image_path(path, image_base_path)
            images.append({'path': path, 'bytes': bytes_value})
            continue

        if not isinstance(item, str):
            continue
        path = resolve_image_path(item, image_base_path)
        images.append({'path': path})
    return images


def resolve_image_path(path: str, image_base_path: Optional[str]) -> str:
    if not path or os.path.isabs(path) or not image_base_path:
        return path

    candidates = [
        os.path.join(image_base_path, path),
        os.path.join(image_base_path, os.path.basename(path)),
    ]
    image_dir_name = os.path.basename(os.path.normpath(image_base_path))
    if path.startswith(image_dir_name + os.sep) or path.startswith(image_dir_name + '/'):
        stripped = path[len(image_dir_name) + 1:]
        candidates.append(os.path.join(image_base_path, stripped))

    for candidate in candidates:
        normalized = os.path.normpath(candidate)
        if os.path.exists(normalized):
            return normalized
    return os.path.normpath(candidates[0])


def get_question(sample: Dict[str, Any]) -> str:
    question = sample.get('question')
    if question:
        return str(question)
    extra_info = sample.get('extra_info')
    if isinstance(extra_info, dict):
        for key in ['prompt', 'question', 'query']:
            if extra_info.get(key):
                return str(extra_info[key])
    prompt = sample.get('prompt')
    if isinstance(prompt, list) and prompt:
        first = prompt[-1]
        if isinstance(first, dict) and first.get('content'):
            return str(first['content'])
    return ''


def get_answer(sample: Dict[str, Any]) -> str:
    answer = sample.get('answer')
    if answer:
        return str(answer)
    extra_info = sample.get('extra_info')
    if isinstance(extra_info, dict) and extra_info.get('answer'):
        return str(extra_info['answer'])
    reward_model = sample.get('reward_model')
    if isinstance(reward_model, dict) and reward_model.get('ground_truth') is not None:
        ground_truth = reward_model['ground_truth']
        if isinstance(ground_truth, list):
            return str(ground_truth[0]) if ground_truth else ''
        return str(ground_truth)
    return ''


def normalize_rubrics(value: Any) -> Optional[str]:
    def _drop_rubric_ids(obj: Any) -> Any:
        if isinstance(obj, list):
            new_items = []
            for item in obj:
                if isinstance(item, dict):
                    item = dict(item)
                    item.pop('id', None)
                new_items.append(item)
            return new_items
        return obj

    if value is None:
        return None
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return None
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return value
        parsed = _drop_rubric_ids(parsed)
        return json.dumps(parsed, ensure_ascii=False, indent=2)
    value = _drop_rubric_ids(value)
    return json.dumps(value, ensure_ascii=False, indent=2)


def strip_think_tags(cot: str) -> str:
    cot = cot.strip()
    match = re.search(r'<think>\s*(.*?)\s*</think>', cot, re.DOTALL)
    if match is not None:
        return match.group(1).strip()
    if cot.startswith('<think>'):
        cot = cot[len('<think>'):]
    if '</think>' in cot:
        cot = cot.split('</think>', 1)[0]
    return cot.strip()


def ensure_image_tokens(question: str, num_images: int) -> str:
    question = question.strip()
    if num_images <= 0:
        return question
    if '<image>' not in question:
        prefix = ''.join('<image>\n' for _ in range(num_images))
        return f'{prefix}{question}'.strip()
    return question


def build_merge_key(sample: Dict[str, Any]) -> str:
    if sample.get('id') is not None:
        return f"id::{sample['id']}"
    question = get_question(sample)
    answer = get_answer(sample)
    image_value = sample.get('image')
    if image_value is None:
        image_value = sample.get('images')
    try:
        image_part = json.dumps(image_value, ensure_ascii=False, sort_keys=True)
    except TypeError:
        image_part = str(image_value)
    return f'q::{question}||a::{answer}||i::{image_part}'


def merge_records(cot_records: List[Dict[str, Any]], rubrics_records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    cot_map = {build_merge_key(item): item for item in cot_records}
    merged = []
    missing = 0
    for rubric_item in rubrics_records:
        key = build_merge_key(rubric_item)
        cot_item = cot_map.get(key)
        if cot_item is None:
            missing += 1
            if rubric_item.get('id') is not None:
                question = get_question(rubric_item).replace('\n', ' ').strip()
                if len(question) > 160:
                    question = question[:160] + '...'
                print(
                    'Error: failed to match rubrics sample by id. '
                    f"id={rubric_item.get('id')} key={key} question={question}")
            continue
        item = dict(cot_item)
        for k, v in rubric_item.items():
            if k == 'cot' and item.get('cot'):
                continue
            item[k] = v
        merged.append(item)
    if missing:
        print(f'Warning: {missing} rubrics samples could not be matched with cot samples and were skipped.')
    return merged


def build_assistant_content(cot: str, rubrics_json: str) -> str:
    cot = strip_think_tags(cot)
    return f'<think>\n{cot}\n</think>\n```json\n{rubrics_json}\n```'


def convert_record(sample: Dict[str, Any], system_prompt: Optional[str],
                   image_base_path: Optional[str]) -> Optional[Dict[str, Any]]:
    cot = sample.get('cot')
    rubrics_json = normalize_rubrics(sample.get('rubrics'))
    question = get_question(sample)
    if not cot or not rubrics_json or not question:
        return None

    image_value = sample.get('images')
    if image_value is None:
        image_value = sample.get('image')
    images = normalize_images(image_value, image_base_path)
    user_content = ensure_image_tokens(question, len(images))

    messages = []
    if system_prompt:
        messages.append({'role': 'system', 'content': system_prompt})
    messages.append({'role': 'user', 'content': user_content})
    messages.append({'role': 'assistant', 'content': build_assistant_content(str(cot), rubrics_json)})

    result = {
        'messages': messages,
        'images': images,
    }
    for key in ['id', 'question', 'answer', 'cot', 'rubrics', 'ability']:
        if key in sample:
            result[f'_source_{key}'] = sample[key]
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Convert saved cot/rubrics outputs into ms-swift SFT JSONL data.')
    parser.add_argument('--input', type=str, default=None, help='Single input file that already contains both cot and rubrics.')
    parser.add_argument('--cot-input', type=str, default=None, help='Input file containing cot.')
    parser.add_argument('--rubrics-input', type=str, default=None, help='Input file containing rubrics.')
    parser.add_argument('--output', type=str, required=True, help='Output JSONL path for ms-swift SFT.')
    parser.add_argument('--image-base-path', type=str, default=None, help='Base path for resolving relative image paths.')
    parser.add_argument('--system-prompt', type=str, default=None, help='Optional system prompt. Default: do not write a system message into the dataset.')
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if bool(args.input) == bool(args.cot_input or args.rubrics_input):
        raise ValueError('Use either --input, or the pair --cot-input/--rubrics-input.')
    if args.input is None and not (args.cot_input and args.rubrics_input):
        raise ValueError('Both --cot-input and --rubrics-input are required when --input is not used.')

    if args.input:
        records = load_records(args.input)
    else:
        cot_records = load_records(args.cot_input)
        rubrics_records = load_records(args.rubrics_input)
        records = merge_records(cot_records, rubrics_records)

    converted = []
    skipped = 0
    for sample in records:
        row = convert_record(sample, args.system_prompt, args.image_base_path)
        if row is None:
            skipped += 1
            continue
        converted.append(row)

    dump_jsonl(converted, args.output)
    print(f'total_input: {len(records)}')
    print(f'total_output: {len(converted)}')
    print(f'total_skipped: {skipped}')
    print(f'output_path: {os.path.abspath(args.output)}')


if __name__ == '__main__':
    main()
