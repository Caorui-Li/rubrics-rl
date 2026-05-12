# 训练后评测流程
## Step 1：将 FSDP checkpoint 转换为 HuggingFace 格式

verl 训练使用 FSDP 多卡分片保存，推理前需要合并为标准 HF 格式。

```bash
python -m verl.model_merger merge \
    --backend fsdp \
    --local_dir verl_checkpoints/global_step_80/actor \
    --target_dir dataset/models/rubrics-rl-multidata-step80
```


## Step 2：下载评测数据集

所有数据集统一下载到项目根目录的 `dataset/` 子目录（与 `--target_dir` 同级），通过 `HF_DATASETS_CACHE` 缓存到对应子目录。

### 下载全部数据集

```bash
python evaluation/scripts/download_datasets.py
```


下载完成后目录结构如下：

```
dataset/
  MathVista/
  MathVision/
  MathVerse/
  MMK12/
  OlympiadBench/
  models/
    rubrics-rl-multidata-step80/   ← Step 1 转换的 HF 权重
```

## Step 3：配置 System Prompt


```
evaluation/conf/prompts/train.txt
```

---

## Step 4：配置 Judge 模型环境变量

评测的答案提取/打分需要一个 judge LLM，通过环境变量指定。

```bash
# 使用 GPT-4o
export JUDGE_MODEL=gpt-4o
export JUDGE_MODEL_BASE_URL=https://api.openai.com/v1
export JUDGE_MODEL_API_KEY=<your_openai_key>

# 或使用 Gemini
export JUDGE_MODEL=gemini-2.0-flash
export JUDGE_MODEL_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai
export JUDGE_MODEL_API_KEY=<your_gemini_key>

# 或使用本地部署的模型
export JUDGE_MODEL=<model-name-on-server>
export JUDGE_MODEL_BASE_URL=http://<host>:8000/v1
export JUDGE_MODEL_API_KEY=EMPTY
```


## Step 5：启动评测

所有命令均在项目根目录 `verl-exp/` 下执行。

配置文件已建好：`evaluation/conf/rubrics_rl_multidata.yaml`

**运行前需修改其中两个字段**（当前值为占位符 `stepXX`）：

```yaml
checkpoint: dataset/models/rubrics-rl-multidata-stepXX
model_name: rubrics-rl-multidata-stepXX
```

修改完成后直接运行：

```bash
python evaluation/run_all.py --config-name rubrics_rl_multidata
```

### 常用追加参数

| 参数 | 说明 | 示例 |
|------|------|------|
| `tensor_parallel_size` | 推理时的张量并行数（需整除 GPU 总数） | `tensor_parallel_size=4` |
| `limit` | 每个 benchmark 只跑前 N 条，调试用 | `limit=10` |
| `suites` | 只跑指定 benchmark | `"suites=[mathvista,mathvision]"` |
| `gen_max_tokens` | 模型最大输出 token 数 | `gen_max_tokens=8192` |
| `system_prompt` | 覆盖 system prompt | `"system_prompt=Think step by step."` |

示例（只跑两个 benchmark，限制条数调试）：

```bash
python evaluation/run_all.py \
    --config-name rubrics_rl_multidata \
    "suites=[mathvista,mathvision]" \
    limit=20 \
    tensor_parallel_size=4
```
