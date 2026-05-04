# Rubrics RL 训练操作指南（wethink + geothought 混合数据）

以下命令均以仓库根目录 `verl-exp` 为相对路径基准。

Judge 使用外部 API（GPT / Gemini / DeepSeek 等），无需单独部署 server，只需一台 8 卡训练节点。

---

## 1. 数据下载

### 1.1 下载 wethink 数据集

```bash
hf download GlowLED/wethink-rubrics-20k-test --repo-type=dataset --local-dir ./data/wethink_rubrics
```

wethink 图片来自 LLaVA-CoT-100k，以 zip 分片形式存储（`image.zip.part-aa` … `part-ap`，共 16 个），需单独下载：

```bash
hf download Xkev/LLaVA-CoT-100k --repo-type=dataset --local-dir ./data/wethink_rubrics
```

> 分片总大小约几十 GB。下载后运行 `data/extract_wethink_images.py`（训练脚本 Step 1 会自动调用）拼接分片并解压所需图片到 `data/wethink_rubrics/images/`。

### 1.2 下载 geothought 数据集

```bash
hf download GlowLED/geothought-rubrics-17k-test --repo-type=dataset --local-dir ./data/geothought_rubrics
```

> geothought 图片已嵌入数据集的 Parquet 文件中，无需额外下载。脚本会自动从 Parquet 提取到 `data/geothought_rubrics/images/`。


---

## 2. 数据处理 + 训练（一键启动）

完成数据下载后，直接运行训练脚本。脚本会自动完成以下步骤：

1. 从 wethink zip 分片中提取所需图片 → `data/wethink_rubrics/images/`
2. 从 geothought Parquet 中提取所需图片 → `data/geothought_rubrics/images/`
3. 两份数据合并转换为 verl parquet（9:1 train/val split）→ `dataset/rubrics_mixed/`
4. 启动 RL 训练

```bash
cd verl-exp   # 仓库根目录
bash ./recipe/rubrics_rl/run_rubrics_rl_multidata.sh
```

---

## 3. 配置 Judge 模型

支持任意 OpenAI 兼容 API，按需选择一种：

**GPT（OpenAI）**
```bash
export JUDGE_MODEL="gpt-4o"
export LLM_AS_A_JUDGE_BASE="https://api.openai.com/v1"
export JUDGE_MODEL_API_KEY="<openai key>"
```

**Gemini**
```bash
export JUDGE_MODEL="gemini-2.0-flash"
export LLM_AS_A_JUDGE_BASE="https://generativelanguage.googleapis.com/v1beta/openai"
export JUDGE_MODEL_API_KEY="<gemini key>"
```

**DeepSeek**
```bash
export JUDGE_MODEL="deepseek-chat"
export LLM_AS_A_JUDGE_BASE="https://api.deepseek.com/v1"
export JUDGE_MODEL_API_KEY="<deepseek key>"
```

**本地 vllm server（备选）**
```bash
export JUDGE_MODEL="<model-name>"
export LLM_AS_A_JUDGE_BASE="http://<host>:8000/v1"
export JUDGE_MODEL_API_KEY="EMPTY"
```

---

## 4. 环境变量汇总

启动前设置以下变量（`必填` 未设置会报错退出，`选填` 有默认值）：

| 变量 | 必填 | 说明 | 默认值 |
|------|------|------|--------|
| `REF_MODEL_PATH` | 必填 | 被训模型路径（Qwen2.5-VL-7B-Instruct） | 无 |
| `JUDGE_MODEL` | 必填 | Judge 模型名，见第 3 节 | 无 |
| `LLM_AS_A_JUDGE_BASE` | 必填 | Judge API base URL，见第 3 节 | 无 |
| `JUDGE_MODEL_API_KEY` | 必填 | Judge API key，见第 3 节 | 无 |
| `WANDB_API_KEY` | 选填 | WandB 日志 key | `""` （不上传） |
| `SAVE_CHECKPOINT_DIR` | 选填 | checkpoint 保存路径 | `./verl_checkpoints` |
| `MAX_CONCURRENT_JUDGE_REQUESTS` | 选填 | judge 并发请求数，视 API 速率限制调整 | `6` |
| `JUDGE_DEBUG_JSONL_PATH` | 选填 | judge 输入输出调试日志路径 | `./judge_verify_debug.jsonl` |

示例：

```bash
export REF_MODEL_PATH=/path/to/Qwen2.5-VL-7B-Instruct
export JUDGE_MODEL="gpt-4o"
export LLM_AS_A_JUDGE_BASE="https://api.openai.com/v1"
export JUDGE_MODEL_API_KEY="sk-..."
export SAVE_CHECKPOINT_DIR=/path/to/checkpoints   # 可选
export WANDB_API_KEY=your_key                      # 可选
```

---

## 5. 训练默认参数

| 参数 | 默认值 |
|------|--------|
| `trainer.n_gpus_per_node` | 8 |
| `trainer.nnodes` | 1 |
| `trainer.total_epochs` | 4 |
| `data.train_batch_size` | 512 |
| `actor_rollout_ref.rollout.n` | 8 |
| train/val split | 9:1 |

---

## 注意事项

- 启动后观察日志，确认 judge server 返回正常；若出现超时报错，适当下调 `MAX_CONCURRENT_JUDGE_REQUESTS`。
- 若只需重新训练（图片和 parquet 已生成），可直接跳到第 4 步，注释掉脚本中的 Step 1~3。
- geothought 图片从 Parquet 中提取，需要 `pyarrow`：`pip install pyarrow`。
