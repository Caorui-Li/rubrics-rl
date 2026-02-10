# Evaluation

统一多数据集评测入口，支持：

- 单次加载模型，顺序跑全部 benchmark
- 每个样本写回 `is_correct` / `score`
- 汇总关键指标到 `all_metrics.csv`（`overall/average`）
- 保存完整运行元信息到 `meta_info.json`
- `system_prompt` 支持路径或直接文本

## 目录结构

```text
evaluation/
  conf/
    run_all.yaml
    baseline_qwen25_vl_7b.yaml
    prompts/
      default.txt
  scripts/
    run_baseline.sh
    download_datasets.py
  <benchmark>/
    evaluate_*.py
    extract_calculate.py
  run_all.py
  utils.py
```

## 快速使用

### 1) 环境变量

```bash
export JUDGE_MODEL=...
export JUDGE_MODEL_BASE_URL=...
export JUDGE_MODEL_API_KEY=...
```

### 2) 下载数据（可选）

```bash
python evaluation/scripts/download_datasets.py
python evaluation/scripts/download_datasets.py --dataset MathVista
```

### 3) 跑 baseline

```bash
bash evaluation/scripts/run_baseline.sh
```

可追加 Hydra 覆盖参数：

```bash
bash evaluation/scripts/run_baseline.sh limit=10 tensor_parallel_size=4
```

## System Prompt

`run_all.yaml` 中的 `system_prompt` 支持两种输入：

- 文件路径：若路径存在，则读取文件内容
- 纯文本：若路径不存在，则直接当 prompt 文本

YAML 支持换行文本，可直接写：

```yaml
system_prompt: |
  You are a careful math assistant.
  Return only the final answer.
```

## 输出

输出目录结构：

```text
outputs/evaluation/<model_name>/<timestamp>/
  all_metrics.csv
  meta_info.json
  mathvista/
  mathvision/
  mathverse/
  mmk12/
  olympiadbench/
```

## 开发新 Benchmark

### 1) 新建目录

新增 `evaluation/<your_benchmark>/`，至少包含：

- `evaluate_<name>.py`
- `extract_calculate.py`

### 2) 实现 generation 接口

`evaluate_<name>.py` 需要提供：

```python
def evaluate_chat_model(args, llm, processor, stop_token_ids) -> list[str]:
    ...
```

返回值是该 benchmark 产出的原始结果文件名列表（如 `["xxx.json"]`）。

### 3) 实现 extraction/scoring 接口

`extract_calculate.py` 需要提供：

```python
def run_extract(
    output_dir,
    output_file,
    response_label="response",
    number=-1,
    output_label="extract",
    init_judge=True,
    judge_client=None,
    judge_model=None,
    judge_stats=None,
):
    ...
```

建议输出：

- `<xxx>_extract.json`（逐样本结果，包含 `score`）
- `<xxx>_score.json`（结构化指标）

### 4) 注册到 `run_all.py`

在 `SUITES` 中增加：

- `eval_module`
- `extract_module`

### 5) 配置数据源

在 `evaluation/conf/run_all.yaml` 的 `dataset_specs` 中新增该 benchmark 的数据集配置。
