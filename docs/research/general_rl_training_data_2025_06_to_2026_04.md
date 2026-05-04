# 图片输入 RL 数据简表

## Zebra-CoT

- 链接：
  - 论文：<https://arxiv.org/abs/2507.16746>
  - 数据页：<https://huggingface.co/datasets/multimodal-reasoning-lab/Zebra-CoT>
- 是否开源：
  - 是，论文和 Hugging Face 数据页都已公开
- 任务类型：
  - 通用视觉推理
  - 科学图像推理
  - 2D/3D 视觉理解
  - visual logic
- 制作流程：
  1. 构造交错的 text-image reasoning traces。
  2. 覆盖科学推理、2D/3D visual reasoning、visual logic 等多类任务。
  3. 将问题图、推理图、文本推理链和答案组织成统一样本。

## OneThinker-train-data

- 链接：
  - 论文：<https://arxiv.org/abs/2512.03043>
  - 数据页：<https://huggingface.co/datasets/OneThink/OneThinker-train-data>
- 是否开源：
  - 是，论文明确说明 code、model、data 已发布
- 任务类型：
  - 通用视觉推理
  - 图像问答
  - 视频理解
  - 多任务视觉 RL
- 制作流程：
  1. 构造覆盖图像和视频任务的 OneThinker-600k 数据。
  2. 用强模型补充 CoT 标注，形成 OneThinker-SFT-340k 冷启动数据。
  3. 在此基础上继续做多任务 RL post-training。

## GeoThought

- 链接：
  - 论文：<https://arxiv.org/abs/2510.21881>
- 是否开源：
  - 论文公开了数据集名称与规模；具体数据下载页需以作者后续发布为准
- 任务类型：
  - 几何图像数学推理
  - diagram-based math reasoning
- 制作流程：
  1. 收集需要图形理解的几何数学题。
  2. 为每道题补视觉图形、分步解答、reasoning chain 和 reflection。
  3. 形成 Geo-Thought-6K 和 Geo-Thought-Augmented-10K。

## VisRefiner 使用的 screenshot-to-code 数据

- 链接：
  - 论文：<https://arxiv.org/abs/2602.05998>
- 是否开源：
  - 论文公开了方法；独立数据集页需以作者后续发布为准
- 任务类型：
  - screenshot-to-code
  - image-to-code
  - 视觉代码修正
- 制作流程：
  1. 输入目标 UI 截图，先生成代码。
  2. 渲染代码得到预测界面图。
  3. 比较预测图和目标截图，构造视觉差异。
  4. 把视觉差异和代码修改对齐，形成看图改代码数据。
  5. 再用于 RL self-refinement。

## WeThink Dataset

- 链接：
  - 论文：<https://arxiv.org/abs/2506.07905>
- 是否开源：
  - 论文明确宣称 open-source dataset
- 任务类型：
  - 通用视觉推理
  - 图像问答
  - 图片输入的数学推理子任务
  - 多模态 RL post-training
- 制作流程：
  1. 从多个视觉数据源中取图像。
  2. 用 multimodal QA synthesis pipeline 从图像自动生成 reasoning-centric QA。
  3. 为样本补 reasoning paths。
  4. 用这些图像问答数据配合 hybrid reward 做 RL。

## MIRG-RL Dataset

- 链接：
  - 论文：<https://arxiv.org/abs/2509.21788>
  - 项目页：<https://github.com/ZEUS2035/MIRG-RL>
- 是否开源：
  - 代码仓库已公开；数据开放情况以项目页为准
- 任务类型：
  - 多图推理
  - multi-image grounding
  - cross-image reference reasoning
- 制作流程：
  1. 从多图 grounding 任务出发，收集 object-level 和 image-level annotation。
  2. 结合两级标注生成 trajectory-style reasoning data。
  3. 构造轻量 reasoning-enhanced dataset。
  4. 配合 dual rewards 做多图推理训练。

## ReasonMap-Plus

- 链接：
  - 论文：<https://arxiv.org/abs/2510.02240>
- 是否开源：
  - 我目前没有核到稳定公开数据页
- 任务类型：
  - 地图视觉推理
  - 细粒度视觉理解
  - 结构化空间推理
- 制作流程：
  1. 在原始 ReasonMap 任务上扩展更细粒度的视觉问答。
  2. 为训练构造 dense reward 信号。
  3. 先做 perception cold start，再进入多阶段 RL。

## From Sight to Insight 使用的视觉推理训练数据

- 链接：
  - 论文：<https://arxiv.org/abs/2601.00215>
- 是否开源：
  - 我目前没有核到独立公开数据集页
- 任务类型：
  - 通用图像理解
  - visual puzzle reasoning
  - reward-driven visual reasoning
- 制作流程：
  1. 选择需要真实图像理解的视觉推理任务。
  2. 为这些任务设计 image understanding、thinking steps、answer accuracy 等 reward。
  3. 用 reward-driven RL 训练视觉推理模型。
