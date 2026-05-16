# Rubrics SFT 启动说明

以下命令均以仓库根目录 verl-exp 为相对路径基准。

## 1. 克隆 ms-swift

```bash
git clone https://github.com/modelscope/ms-swift.git
```

## 2. 创建 swift 环境

```bash
conda create -n swift python=3.11 -y
conda activate swift
```

安装基础依赖：

```bash
pip install -U pip
pip install -e ./ms-swift
```

## 3. 下载数据到 `./dataset`

```bash
mkdir -p ./dataset/ViRL39K
```

```bash
huggingface-cli download khazic/ViRL-Rubric \
  --repo-type dataset \
  --local-dir ./dataset/ViRL39K
```
另外从 https://huggingface.co/datasets/TIGER-Lab/ViRL39K/tree/main 下载images


## 4. 转换为 SFT 数据

```bash
python ./recipe/rubrics_rl/convert_rubrics_to_sft.py \
  --cot-input ./dataset/ViRL39K/cot_filtered.jsonl \
  --rubrics-input ./dataset/ViRL39K/rubrics_filtered.jsonl \
  --image-base-path ./dataset/ViRL39K \
  --output ./dataset/ViRL39K/rubrics_sft.jsonl
```

转换后的输出文件为：

```bash
./dataset/ViRL39K/rubrics_sft.jsonl
```

## 5. 启动 8 卡 A100 SFT 训练

使用 8 张卡启动训练：

```bash
TRAIN_GPUS=0,1,2,3,4,5,6,7 \
NPROC_PER_NODE=8 \
NUM_TRAIN_EPOCHS=3 \
MODEL_PATH=Qwen/Qwen3-VL-32B-Thinking \
./recipe/rubrics_rl/run_rubrics_sft.sh
```

maybe更高吞吐的配置：

```bash
PER_DEVICE_TRAIN_BATCH_SIZE=32 \
PER_DEVICE_EVAL_BATCH_SIZE=32 \
GRADIENT_ACCUMULATION_STEPS=2 \
DATALOADER_NUM_WORKERS=16 \
DATASET_NUM_PROC=16 \
```

训练脚本会自动：

- 读取 `./dataset/ViRL39K/rubrics_sft.jsonl`
- 在 `./dataset/ViRL39K` 下切分训练集和验证集 ！！！！！！！！！
- 所以如果数据更新了，请把./dataset/ViRL39K/rubrics_sft_train.jsonl和./dataset/ViRL39K/rubrics_sft_val.jsonl删了，因为不会自动重复切分！！！！！！！！！
- 加载 `./recipe/rubrics_rl/sft_system_prompt.txt`
- 将输出保存到 `./ms-swift/output/rubrics_sft_qwen3_vl`
- 使用 wandb 离线模式记录训练日志

## 6. wandb 离线日志

当前 `./recipe/rubrics_rl/run_rubrics_sft.sh` 已默认启用 wandb 离线记录：

- `WANDB_MODE=offline`
- `WANDB_PROJECT=rubrics-sft`
- `WANDB_NAME=rubrics-sft-qwen3-vl`
- `WANDB_DIR=./ms-swift/output/wandb`

默认情况下，日志会写到：

```bash
./ms-swift/output/wandb
```

如果你想自定义 wandb 配置：

```bash
WANDB_PROJECT=my_rubrics_project \
WANDB_NAME=exp_8xa100 \
WANDB_DIR=./wandb_logs \
```
