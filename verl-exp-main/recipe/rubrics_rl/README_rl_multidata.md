# Rubrics RL 训练操作指南（wethink + geothought + virl39k 混合数据）

以下命令均以仓库根目录 `verl-exp-main` 为相对路径基准。

---

## 0. 设置数据存储路径

训练节点本地磁盘有限时，将数据集下载到大容量挂载盘，通过 `DATA_ROOT` 指定：

```bash
export DATA_ROOT=/mnt/storage/data   # 按实际挂载路径修改
```

不设置时默认为仓库目录下的 `data/`。**所有数据下载命令均使用此变量**，训练脚本和提取脚本也会自动读取。

---

## 1. 数据下载

### 1.1 下载 wethink 数据集

```bash
hf download GlowLED/wethink-rubrics-20k-test --repo-type=dataset --local-dir ${DATA_ROOT}/wethink_rubrics
```

wethink 图片来自 LLaVA-CoT-100k，以 zip 分片形式存储（`image.zip.part-aa` … `part-ap`，共 16 个），需单独下载：

```bash
hf download Xkev/LLaVA-CoT-100k --repo-type=dataset --local-dir ${DATA_ROOT}/wethink_rubrics
```

> 分片总大小约几十 GB。

### 1.2 下载 geothought 数据集

```bash
hf download GlowLED/geothought-rubrics-17k-test --repo-type=dataset --local-dir ${DATA_ROOT}/geothought_rubrics
```

> geothought 图片已嵌入数据集的 Parquet 文件中，无需额外下载。

### 1.3 下载 virl39k-rubrics 数据集

```bash
hf download GlowLED/virl-39k-rubrics-test --repo-type=dataset --local-dir ${DATA_ROOT}/virl39k_rubrics
```

virl39k rubrics 的图片来自 `TIGER-Lab/ViRL39K`，以 zip 分片形式存储，需单独下载到同一目录：

```bash
hf download TIGER-Lab/ViRL39K --repo-type=dataset --local-dir ${DATA_ROOT}/virl39k_rubrics
```


---

## 2. 数据处理（单节点，运行一次）

在任意一台节点上运行，提取图片并生成训练所需的 parquet：

```bash
cd verl-exp-main
export DATA_ROOT=/mnt/storage/data   # 各训练节点均可访问的共享路径
bash ./recipe/rubrics_rl/prepare_data_multidata.sh
```

完成后 parquet 输出到 `dataset/rubrics_mixed/`。

---

## 3. 训练

> **前提**：`dataset/rubrics_mixed/mixed_train.parquet` 和 `mixed_val.parquet` 已生成（即第 2 节已完成）。

### 单节点

```bash
cd verl-exp-main
export REF_MODEL_PATH=/mnt/storage/models/Qwen2.5-VL-7B-Instruct
export JUDGE_MODEL="gpt-4o"               # 见第 4 节选择其他 judge
export LLM_AS_A_JUDGE_BASE="https://api.openai.com/v1"
export JUDGE_MODEL_API_KEY="sk-..."
bash ./recipe/rubrics_rl/run_rubrics_rl_multidata.sh
```

### 多节点

编辑 `recipe/rubrics_rl/launch_multinode.sh`，填入各节点 IP：

```bash
NODES=(
    "192.168.1.10"   # node 0 — Ray head
    "192.168.1.11"   # node 1
    "192.168.1.12"   # node 2
    "192.168.1.13"   # node 3
)
```

然后在任意一台机器上运行一次：

```bash
export REF_MODEL_PATH=/mnt/storage/models/Qwen2.5-VL-7B-Instruct
export JUDGE_MODEL="gpt-4o"
export LLM_AS_A_JUDGE_BASE="https://api.openai.com/v1"
export JUDGE_MODEL_API_KEY="sk-..."
export DATA_ROOT=/mnt/storage/data
bash ./recipe/rubrics_rl/launch_multinode.sh
```

脚本会自动：① 在 `NODES[0]` 启动 Ray head；② 在其余节点启动 Ray worker；③ 在 head 节点运行训练命令（Ray 负责分发到所有节点）。

> **前提**：launcher 所在机器可免密 SSH 到所有训练节点；`dataset/rubrics_mixed/` 在各节点上路径一致（共享存储或提前同步）；各节点已安装 Ray。

---

## 4. 配置 Judge 模型

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

先在 judge 机器上启动服务：

```bash
vllm serve <JUDGE_MODEL_PATH> \
    --host 0.0.0.0 --port 8000 \
    --dtype bfloat16 \
    --tensor-parallel-size 1 \
    --gpu_memory_utilization 0.7
```

再设置以下变量（`JUDGE_MODEL` 与 `--served-model-name` 一致，默认为模型目录名）：

```bash
export JUDGE_MODEL="<model-name>"
export LLM_AS_A_JUDGE_BASE="http://<host>:8000/v1"
export JUDGE_MODEL_API_KEY="EMPTY"
```

---

## 5. 环境变量汇总

启动前设置以下变量（`必填` 未设置会报错退出，`选填` 有默认值）：

| 变量 | 必填 | 说明 | 默认值 |
|------|------|------|--------|
| `REF_MODEL_PATH` | 必填 | 被训模型路径（Qwen2.5-VL-7B-Instruct） | 无 |
| `JUDGE_MODEL` | 必填 | Judge 模型名，见第 3 节 | 无 |
| `LLM_AS_A_JUDGE_BASE` | 必填 | Judge API base URL，见第 3 节 | 无 |
| `JUDGE_MODEL_API_KEY` | 必填 | Judge API key，见第 3 节 | 无 |
| `DATA_ROOT` | 选填 | 数据集根目录，磁盘有限时指向大容量挂载盘 | `<repo>/data` |
| `NNODES` | 选填 | 训练节点总数（Ray 资源分配用） | `1` |
| `RAY_ADDRESS` | 选填 | Ray 集群地址，多节点时设为 `auto` | `""` （本地单节点） |
| `WANDB_API_KEY` | 选填 | WandB 日志 key | `""` （不上传） |
| `SAVE_CHECKPOINT_DIR` | 选填 | checkpoint 保存路径 | `./verl_checkpoints` |
| `MAX_CONCURRENT_JUDGE_REQUESTS` | 选填 | judge 并发请求数，视 API 速率限制调整 | `6` |
| `JUDGE_DEBUG_JSONL_PATH` | 选填 | judge 输入输出调试日志路径 | `./judge_verify_debug.jsonl` |

示例：

```bash
export DATA_ROOT=/mnt/storage/data                 # 数据放在大容量盘
export REF_MODEL_PATH=/mnt/storage/models/Qwen2.5-VL-7B-Instruct
export JUDGE_MODEL="gpt-4o"
export LLM_AS_A_JUDGE_BASE="https://api.openai.com/v1"
export JUDGE_MODEL_API_KEY="sk-..."
export SAVE_CHECKPOINT_DIR=/mnt/storage/checkpoints  # 可选
export WANDB_API_KEY=your_key                         # 可选
```

---

## 6. 训练默认参数

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

- 若只需重新训练（图片和 parquet 已生成），可跳过第 1、2 节，直接从第 3 节开始。
