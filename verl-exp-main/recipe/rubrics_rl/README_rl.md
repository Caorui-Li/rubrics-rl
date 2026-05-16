# Rubrics RL 训练操作指南（rubrics_pipeline_all）

以下命令均以仓库根目录 `verl-exp` 为相对路径基准。

本流程包含两个节点：

- Judge 节点：部署 OpenAI 兼容的 judge server
- RL 训练节点：8 卡训练，调用远端 judge

---

## 1. 数据准备（直接下载切分后的数据）

本次训练使用已切分的 9:1 数据：

- `rubrics_pipeline_all_train.jsonl`
- `rubrics_pipeline_all_val.jsonl`

直接从 Hugging Face 下载到./dataset/ViRL39K：

```bash
hf download TIGER-Lab/ViRL39K --repo-type=dataset \
    --local-dir ./dataset/ViRL39K

hf download CaoruiLi/test-sft \
  rubrics_pipeline_all_train.jsonl \
  --repo-type dataset \
  --local-dir ./dataset/ViRL39K

hf download CaoruiLi/test-sft \
  rubrics_pipeline_all_val.jsonl \
  --repo-type dataset \
  --local-dir ./dataset/ViRL39K

cd ./dataset/ViRL39K
unzip images

```

---

## 2. Judge Server 部署（另一节点，2卡）

```bash
 vllm serve <path-to-Qwen3-vl-32b-thinking> \
   --host 0.0.0.0 \
   --port 8000 \
   --dtype bfloat16 \
   --tensor-parallel-size 1 \ 
   --gpu_memory_utilization 0.7 
```

确保：

- 训练节点可以访问 `http://<JUDGE_IP>:8000/v1`

连通性测试（在训练节点执行）：

```bash
curl http://<JUDGE_IP>:8000/v1/models
```

---

## 3. 训练节点配置（8 卡）

### 3.1 设置 judge 地址

在训练节点设置环境变量：

```bash
export LLM_AS_A_JUDGE_BASE="http://<JUDGE_IP>:8000/v1"
export OPENAI_API_KEY="EMPTY"
export JUDGE_MODEL=<path-to-Qwen3-vl-32b-thinking>   # 这里需要与上面启动server时完全一致
export WANDB_API_KEY=*****
export MAX_CONCURRENT_JUDGE_REQUESTS=16    # 看情况调整
export JUDGE_DEBUG_JSONL_PATH=<path-to-record-judger-outputs>
export SAVE_CHECKPOINT_DIR=<path-to-save-ckpt>
export REF_MODEL_PATH=<path-to-qwen25vl-7b-it>
```


## 4. 启动 RL 训练

在训练节点执行：

```bash
cd verl-exp  # 也就是仓库的根目录
bash ./recipe/rubrics_rl/run_rubrics_rl.sh
```

!!!启动需要观察日志，server的返回是否正常，可能会有error，稍微下调一下server的并行度

默认参数：

- `trainer.n_gpus_per_node=8`
- `trainer.nnodes=1`
- `trainer.total_epochs=4`
- judge 由 `LLM_AS_A_JUDGE_BASE` 指定

---

