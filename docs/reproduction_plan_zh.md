# ReMemR1 缩小复现方案（Qwen3.5 版）

> **历史文档，已停止维护。** 本文记录的是早期“2B 全参数为主、4B 可选”的分析，
> 已被 [final_reproduction_plan_zh.md](./final_reproduction_plan_zh.md) 取代。
> 后续实现、训练、预算和对外表述不得再以本文的模型定位或参数为准。

> 平台：AutoDL  
> 推荐主硬件：1 张 RTX PRO 6000 Blackwell 96GB  
> 低成本/环境验证硬件：1 张 RTX 5090 32GB  
> 可选硬件：2 张 RTX 5090 32GB，但不作为首选  
> 推荐基础镜像：PyTorch 2.12.1 / Python 3.12 / Ubuntu 22.04 / CUDA 13.0  
> 论文版本：[2509.23040v5](../2509.23040v5.pdf)  
> 仓库基线：`cc514c092ca968a50c52cdcc2e2ba96362fce25a`  
> 方案核对日期：2026-07-16

## 1. 最终建议

这次复现不再把 Qwen2.5-3B 作为主训练模型，而采用以下分层：

| 模型 | 官方模型 ID | 定位 | 推荐硬件 |
|---|---|---|---|
| Qwen3.5-0.8B | `Qwen/Qwen3.5-0.8B` | 环境、模型适配、20 次稳定性循环和 RL smoke；预算不足时的最低复现 | 1 张 5090 |
| Qwen3.5-2B | `Qwen/Qwen3.5-2B` | 正式缩小复现的主模型 | 优先 1 张 PRO 6000 96GB；单张 5090 只做容量测试或短跑 |
| Qwen3.5-4B | `Qwen/Qwen3.5-4B` | 预算允许时的 10-20 step 规模验证，不作为核心验收 | 1 张 PRO 6000 96GB；2 张 5090 仅作备选 |

三个模型均使用官方 post-trained checkpoint。官方命名不带 `-Instruct`，不要把对应的 `-Base` 模型用于主实验。

正式结论优先来自 Qwen3.5-2B 的同预算对照：

1. Sequential Memory / no-callback，纯 outcome reward。
2. ReMemR1 callback，`alpha=1.0`，纯 outcome reward。
3. ReMemR1 callback，`alpha=0.8`，outcome reward 与 state reward 组合。

若预算不足，最低保留第 2、3 个条件，用同 checkpoint 的 learned/no/fixed-query 推理消融补充 callback 证据，并明确它弱于独立训练的 no-callback baseline。

环境选择明确如下：

- 选择 AutoDL 的 CUDA 13.0 / PyTorch 2.12.1 镜像作为驱动、CUDA toolkit 和系统基础。
- 不直接沿用镜像预装的 PyTorch 2.12.1。项目虚拟环境优先锁定 `torch==2.11.0` 的 cu130 wheel，以匹配当前 Qwen3.5 推理引擎生态。
- CUDA 12.8 / PyTorch 2.8 只作为旧 Qwen2.5/旧 verl 路线的回退，不作为 Qwen3.5 主路线。
- 第一版训练在完成第 8.5 节接口修复后使用仓库的 Hugging Face rollout，先避开旧 verl 与新 vLLM/SGLang 私有接口的冲突。只有修复后的 HF rollout 吞吐实测无法满足预算时，才迁移现代 verl + vLLM。

## 2. 复现边界与成功标准

论文原始训练规模远超本项目预算：

- 3B：16 张 H800，约 100 小时，约 1600 H800-GPUh。
- 7B：32 张 H800，约 80 小时，约 2560 H800-GPUh。
- 训练 batch 128、GRPO group 16、200-300 steps。
- 每条训练样本 200 documents，约 30K 输入 token。
- `5000 x 6` chunks，每个中间状态和最终状态最多生成 2048 token。

因此，本项目的目标是“机制与趋势的缩小复现”，而不是复刻论文表格的绝对数值。合格结果必须同时满足：

1. 跑通数据、recurrent rollout、callback、state/outcome reward、反向传播、checkpoint、合并和评测闭环。
2. 在相同基础模型、样本、seed、步数和解码配置下，对比 `alpha=0.8` 与 `alpha=1.0`。
3. 通过独立 no-callback 训练或同 checkpoint 推理消融，验证 callback 的方向性作用。
4. 在 HotpotQA 与 2WikiMultiHopQA、200 与 800 documents 上报告聚合结果，不挑选单个有利格子。
5. 明确 Qwen3.5 与论文 Qwen2.5 基座不同，不能把绝对准确率差异归因于 ReMemR1。

默认总预算按 500 元规划，其中 GPU 消费硬停止线为 450 元，至少保留 50 元用于磁盘、排障或最后评测。若实际预算低于该值，按第 12 节的降级顺序执行。

## 3. Qwen3.5 带来的关键变化

Qwen3.5 不是 Qwen2.5 的直接模型名替换。

### 3.1 模型结构

| 型号 | 官方语言模型规模 | checkpoint 总参数量（含视觉部分） | 层数 | 默认思考模式 |
|---|---:|---:|---:|---|
| 0.8B | 约 0.8B | 873,438,784 | 24 | non-thinking |
| 2B | 约 2B | 2,274,069,824 | 24 | non-thinking |
| 4B | 约 4B | 4,659,865,088 | 32 | thinking |

三者均是统一多模态 checkpoint：

- `model_type=qwen3_5`
- `architectures=["Qwen3_5ForConditionalGeneration"]`
- 语言部分采用每 4 层中 3 层 Gated DeltaNet 和 1 层 gated full attention 的混合结构。
- 原生上下文长度为 262,144 token，但本复现仍使用论文约 30K 的 recurrent 训练上下文，不因模型上限较大而扩大训练规模。
- checkpoint 包含视觉编码器和 MTP 参数；本项目是纯文本训练，不应无意中为这些模块分配完整训练状态。

### 3.2 强制关闭 Qwen 原生 thinking

三个模型的所有 `apply_chat_template` 调用都必须显式传：

```python
enable_thinking=False
```

尤其是 4B 默认会进入原生 thinking 模式。如果不关闭，会额外生成 `<think>...</think>`，消耗输出预算并干扰 ReMemR1 的 action 格式解析。不能依赖 Qwen3 的 `/nothink` 软开关，因为 Qwen3.5 不采用这一路径。论文 prompt 自己要求的任务级 `<thinking>...</thinking>` 仍然保留，二者不是同一种模板行为。

需要增加模板快照测试，确认：

- prompt 中没有意外的原生 `<think>` 引导，但保留任务级 `<thinking>` action 指令。
- generation boundary 正确。
- `<update>`、`<recall>`、最终答案格式能往返解析。
- 三种模型都关闭 Qwen 原生 thinking，并采用一致的 ReMemR1 任务 action 格式。

### 3.3 Gated DeltaNet 的训练约束

Qwen3.5 的主要层不是普通 full attention。正式训练前必须满足：

- 安装并锁定 `flash-linear-attention` 和 `causal-conv1d`。
- full-attention 层再使用经过 Blackwell smoke 的 FlashAttention 或 PyTorch SDPA。
- `use_remove_padding=false`。
- Ulysses sequence parallel size 为 1。
- 不应用仓库中的 Qwen2 专用 monkey patch。

现有 remove-padding 会把多条样本压成一条并令 `attention_mask=None`。full attention 可以通过 varlen 边界隔离样本，但 Gated DeltaNet 的 recurrent state 可能跨样本串联，造成静默训练污染，因此不能开启。

截至方案核对日，FLA 的 Blackwell backward hang 修复刚进入上游，正式 PyPI 版本不一定包含，且 forward hang 风险尚未完全关闭。必须先在 0.8B 上连续完成至少 20 个 forward/backward/optimizer 循环，无 hang、NaN 或状态串联异常后，才允许启动 2B。

## 4. 实验矩阵

### 4.1 必做工程实验

| 编号 | 模型 | 配置 | 步数 | 目的 |
|---|---|---|---:|---|
| G-1 | 无 GPU/CPU | HF rollout 接口、完整 Hydra config、callback modes、HF eval runner | 单元测试 | 清除当前代码的确定性阻塞 |
| G0 | 0.8B | 单状态、短序列 | 20 个优化循环 | 验证 Torch、GDN、FLA、BF16 和 Blackwell 稳定性 |
| G1 | 0.8B | 2 chunks、group 4、输出 128-256 | 1 step + 1 个恢复 step | 覆盖 rollout、reward、backward、保存、恢复续训、合并、评测 |
| G2 | 2B | 两个独立容量任务：group 4 x 1 step；group 8 x 2 steps | 合计 3 steps | 测显存、吞吐和费用，决定 5090 或 PRO 6000 |

G-1 到 G2 任何一项失败都不进入正式计费训练。G-1 中可以做纯 CPU 的接口和 Hydra 解析测试；需要模型 forward 的部分放到 G0/G1。

### 4.2 Qwen3.5-2B 正式训练

三个条件必须使用相同的：

- 基础 checkpoint revision。
- 训练和验证 sample manifest。
- 样本顺序与 seed。
- batch、group、chunks、输出长度、学习率、步数。
- 训练硬件类型和软件 lock。

| 条件 | Callback | 奖励 | 验证问题 |
|---|---|---|---|
| A. Sequential baseline | 无 | outcome only | 不允许回看历史时的顺序记忆基线 |
| B. ReMemR1 outcome-only | learned callback | `alpha=1.0` | callback 本身是否有帮助 |
| C. ReMemR1 full | learned callback | `alpha=0.8` | 多级 state reward 是否有帮助 |

A 必须从 prompt/action space 中移除 recall 行为，并且不把历史检索内容注入状态；不能让模型照常生成 `<recall>` 后仅在末端丢弃结果。A、B 都只使用 outcome reward，才能较干净地隔离 callback 机制。

推荐节奏：

1. 三个条件先各跑独立的 1-step pilot，验证 reward 分布和格式；pilot 不计入正式曲线。
2. 正式 run 从 step 0 启动，目标先设为 40；经过 step 10 时只检查日志中的非零 advantage、梯度、格式率与平均输出长度，不暂停。
3. step 40 保存完整 checkpoint，merge/export 并做外部小验证。
4. 明确以 `resume_mode=resume_path` 从各自 `global_step_40` 续到 80，再 merge/export 和评测。
5. 只有三个条件成本预测均未超预算且曲线仍改善时，才从各自 `global_step_80` 同步续到 120。
6. 不允许只延长表现较好的条件；每段的 resume path、目标 step 和输出目录都必须写入运行记录。

预算不足时，删除整个 A 条件，保留 B、C 各 80 steps；不要让 A 使用更少步数后与 B、C 作直接结论比较。

### 4.3 0.8B 预算下限

若 2B 在单 5090 上过慢，且 PRO 6000 的实时报价使预计总费用超过预算，则把 0.8B 升级为最低复现模型：

- B、C 两个条件各 50 steps。
- 保持 `5000 x 6` 和 200 documents。
- 结果只用于证明训练闭环及机制方向，不把 0.8B 的能力下限当成方法上限。

### 4.4 4B 可选规模验证

4B 只在以下条件全部满足时执行：

- 2B 核心对照已完成。
- 剩余 GPU 预算至少 100 元。
- PRO 6000 上 3 个完整 step 的实测成本可接受。
- `enable_thinking=False` 的模板测试通过。

先运行 `alpha=0.8` 的 10-20 steps scale check。单条件短跑只能说明工程可扩展性，不能证明 4B 上的奖励增益；只有预算足够做成对条件时才报告机制比较。

### 4.5 同 checkpoint 推理消融

对 B 或 C 的最终 checkpoint 运行：

- learned callback：解析模型生成的 `<recall>` query。
- no callback：禁用历史记忆召回。
- fixed-question callback：每一步用原始问题作检索 query，对应论文 rule-based callback。

这组实验几乎不增加训练成本，但属于 inference-time ablation，不能冒充独立训练的 MemAgent baseline。

### 4.6 作者 checkpoint 复验

`yrshi/ReMemR1-7B` 与 `BytedTsinghua-SIA/RL-MemoryAgent-7B` 降为可选的 pipeline sanity：

- 用于确认评测数据和论文方向大致一致。
- 不与 Qwen3.5 checkpoint 比较绝对精度。
- 预算不足时优先保证自己的 2B 对照，不优先跑作者 7B 全矩阵。

## 5. 推荐训练配置

### 5.1 主配置

| 参数 | 论文 | 2B 缩小主配置 |
|---|---:|---:|
| 训练 documents | 200 | 200 |
| 输入长度 | 约 30K | 约 30K |
| chunks | `5000 x 6` | `5000 x 6` |
| 单状态最大生成 | 2048 | 512；smoke 为 128-256 |
| train batch | 128 | 5090 从 1 起测；PRO 6000 从 4 起测 |
| GRPO group | 16 | 8；smoke 为 4 |
| PPO mini batch | 8 | 必须不大于 train batch：5090 为 1，PRO 6000 为 4 |
| PPO micro batch / GPU | 未单列 | 1 |
| 总步数 | 200-300 | 80；预算允许同步续到 120 |
| actor learning rate | `1e-6` | `1e-6` |
| warmup | 20 steps | 固定 8 steps；分段 resume 时不得改变 |
| KL coefficient | 0.001 | 0.001 |
| clip ratio | 0.2 | 0.2 |
| rollout temperature | 1.0 | 1.0 |
| 精度 | BF16 | BF16 |
| validation | 训练中周期执行 | 40 steps 和 final 暂停后做外部小验证 |
| checkpoint | 多份 | 当前 manager 只保留 1 份完整恢复状态；随后单独 merge/export model-only |

GRPO group 优先保留 8。最终答案奖励接近二值，group 太小容易使同一问题的全部轨迹得分相同，从而产生零 outcome advantage。

表中的 BF16 指 forward/backward mixed-precision 计算；正式主实验仍保留 FP32 actor parameter/Adam moments。若采用 BF16 参数和 BF16 Adam moments，只能作为单 5090 容量 smoke，并单独标注数值风险。

配置时还要保证 `data.train_batch_size * actor_rollout_ref.rollout.n` 能被实际 GPU world size 整除。双卡条件下应成组调整 train batch 与 PPO mini batch，不能只把 `N_GPU` 从 1 改为 2。

### 5.2 资源与 batch 矩阵

| 配置 | G1：0.8B smoke | G2：2B/5090 | 2B 正式/PRO 6000 | 可选 2B/双 5090 |
|---|---:|---:|---:|---:|
| GPU world size | 1 | 1 | 1 | 2 |
| train batch | 1 | 1 | 4 | 2 |
| rollout group `n` | 4 | G2a 为 4；独立 G2b 为 8 | 8 | 8 |
| PPO mini batch | 1 | 1 | 4 | 2 |
| PPO micro batch / GPU | 1 | 1 | 1 | 1 |
| actor/ref log-prob micro batch / GPU | 1 | 1 | 1 | 1 |
| HF generation micro batch | 1 | 1 | 1 | 1 |
| chunk size | 1024 | 5000 | 5000 | 5000 |
| chunks | 2 | 6 | 6 | 6 |
| response length | 128-256 | 512 | 512 | 512 |
| steps | 1 + 1 个 resume smoke | G2a 为 1；独立 G2b 为 2 | 分段 40 -> 80，按门控续到 120 | 先 3，再决定 |

本仓库校验的是 prompt-level `train_batch_size >= ppo_mini_batch_size`，因此不能把单卡 train batch 1 与 PPO mini batch 4 混用。若后续把 PRO 6000 的 train batch 提到 8，PPO mini batch 同步提到 8；所有条件保持相同。

0.8B 预算下限正式配置使用单 5090、train/PPO-mini batch 2/2、group 8、micro batch 1、`5000 x 6`、response 512、50 steps；若 2/2 OOM，则两个条件一起降为 1/1。4B 的工程 scale check 使用 PRO 6000、train/PPO-mini batch 1/1、group 4、micro batch 1、10-20 steps；只有做机制对照时才把 group 提到 8，并重新计费。

### 5.3 完整配置文件要求

下列内容是配置约束，不是可直接复制的零散命令。实施阶段必须生成并版本化完整的 G1、G2a、G2b、A、B、C 配置文件；不存在于当前 schema 的 `callback_mode`、`rollout.micro_batch_size`、`model_dtype`、`mixed_precision` 应先加入配置 schema，或在 Hydra CLI 中使用 `+` 前缀。

每份配置至少显式覆盖：

```text
trainer.nnodes=1
trainer.n_gpus_per_node=<1或2>
trainer.total_training_steps=<G1首段为1、恢复段为2；G2a为1；G2b为2；正式分段为40/80/120>
trainer.total_epochs=1
trainer.save_freq=<G1为1；G2为-1；80/120-step正式训练为40；50-step降级版为25或50>
trainer.test_freq=-1
trainer.val_before_train=false
trainer.save_best_val=false
trainer.max_actor_ckpt_to_keep=1
trainer.resume_mode=<首段为disable；续训为resume_path>
trainer.resume_from_path=<续训时显式指向global_step_40或global_step_80>
trainer.project_name=rememr1_qwen35
trainer.experiment_name=<model_condition_seed_hardware_run_id>
trainer.default_local_dir=<持久盘>/checkpoints/<model_condition_seed_hardware_run_id>
trainer.logger=['console']

recurrent.enable=memory
recurrent.memory.path=<A为新sequential agent；B/C为recurrent/impls/memory_revisit.py>
recurrent.memory.config.callback_mode=<A为none；B/C为learned>
recurrent.memory.config.chunk_size=<G1为1024；G2/正式为5000>
recurrent.memory.config.max_chunks=<smoke为2；正式为6>
recurrent.memory.config.max_prompt_length=1024
recurrent.memory.config.max_memorization_length=<smoke为128-256；正式为512>
recurrent.memory.config.max_final_response_length=<smoke为128-256；正式为512>

data.train_batch_size=<见5.2矩阵>
data.train_files=<固定manifest对应的parquet>
data.val_files=<固定小验证parquet>
data.max_response_length=<smoke为128-256；正式为512>
data.shuffle=false
data.truncation=center

algorithm.adv_estimator=grpo
algorithm.grpo_use_adv=false
algorithm.action_reweight=false
algorithm.alpha=<A/B为1.0；C为0.8>
reward_model.reward_metric=em

actor_rollout_ref.rollout.name=hf
actor_rollout_ref.rollout.n=<4或8>
actor_rollout_ref.rollout.tensor_model_parallel_size=1
actor_rollout_ref.rollout.do_sample=true
actor_rollout_ref.rollout.temperature=1.0
actor_rollout_ref.rollout.top_k=0
actor_rollout_ref.rollout.top_p=0.999
actor_rollout_ref.rollout.micro_batch_size=1
actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1
actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=<8192-12288实测值>

actor_rollout_ref.model.path=<Qwen3.5本地固定revision路径>
actor_rollout_ref.model.use_remove_padding=false
actor_rollout_ref.model.enable_gradient_checkpointing=true
actor_rollout_ref.actor.ulysses_sequence_parallel_size=1
actor_rollout_ref.actor.use_dynamic_bsz=false
actor_rollout_ref.actor.ppo_mini_batch_size=<见5.2矩阵>
actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1
actor_rollout_ref.actor.ppo_max_token_len_per_gpu=<8192-12288实测值>
actor_rollout_ref.actor.use_torch_compile=false
actor_rollout_ref.actor.use_kl_loss=true
actor_rollout_ref.actor.kl_loss_coef=0.001
actor_rollout_ref.actor.kl_loss_type=low_var_kl
actor_rollout_ref.actor.optim.lr=1e-6
actor_rollout_ref.actor.optim.lr_warmup_steps=<smoke为0；正式所有分段固定为8>
actor_rollout_ref.actor.fsdp_config.fsdp_size=<实际GPU数>
actor_rollout_ref.actor.fsdp_config.param_offload=<按5.4节硬件策略>
actor_rollout_ref.actor.fsdp_config.optimizer_offload=<按5.4节硬件策略>
actor_rollout_ref.actor.fsdp_config.mixed_precision={param_dtype:bf16,reduce_dtype:fp32,buffer_dtype:fp32}
actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1
actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=<8192-12288实测值>
actor_rollout_ref.ref.fsdp_config.param_offload=true
actor_rollout_ref.ref.fsdp_config.model_dtype=bf16
```

actor/ref 的单 GPU token budget 先从 8192-12288 测起，必须覆盖一个 5000-token chunk、question、memory 和 512-token response；不要照抄原脚本的 32768。最终值由实际 tokenized 最大状态加安全余量确定。

每份配置在 GPU 计费前执行并保存：

```bash
python -m verl.trainer.main_ppo <config/overrides> --cfg job --resolve
```

检查解析结果中不得残留默认的 8 GPUs、TP=2、train batch 1024、PPO mini batch 256、GAE 或 30 epochs 长跑配置。

G-1 还要用解析后的配置实际实例化一次 `MemoryDataset`；`data.truncation` 必须是 `center`，默认的 `error` 会在 dataset 构造阶段直接失败。A/B/C 的 `experiment_name` 与 `default_local_dir` 必须同时编码模型、条件、seed、硬件和唯一 run ID，禁止共用默认的 `checkpoints/verl_examples/gsm8k`。

[MemoryDataset](../recurrent/impls/memory_revisit.py) 会把数据层的最大 prompt 长度改写为 `max_chunks * chunk_size`。只修改训练脚本顶部的 `MAXLEN` 不会减少实际上下文。

### 5.4 Offload 策略

- 0.8B + 5090：保留 FP32 actor parameter/Adam moments，FSDP 以 BF16 mixed precision 计算；reference 使用 BF16。当前 worker 会无条件对 reference 启用 CPU offload，需把该行为计入吞吐测试。
- 2B + PRO 6000：同样优先保留 FP32 actor parameter/Adam moments、BF16 计算；reference 仍按当前代码 CPU offload。若峰值超过约 80GB，再启用 actor optimizer/parameter 的阶段式 offload。
- 2B + 单 5090：必须测试 optimizer、reference 和必要参数 offload、gradient checkpoint、microbatch 1。若为了启动而直接把 Adam state 降为 BF16，该结果只能算容量 smoke，不能作为正式主实验。
- 4B + PRO 6000：reference CPU offload 与 actor optimizer offload 为默认，并给激活和 rollout cache 留至少 10-20GB 余量。
- 2 张 5090：使用 FSDP shard，不使用 TP 模拟“64GB 单卡”；必须实测 host-mediated 通信。

## 6. 硬件选择

### 6.1 显存估算

按正式路线的 FP32 actor parameter/gradient/Adam moments 与 BF16 计算副本粗略估算，完整训练状态约为 16-20 byte/parameter，尚不含激活、KV/cache、reference CPU 状态、临时 all-gather 和 rollout：

| 模型 | 按 checkpoint 总参数估算的训练状态 | 判断 |
|---|---:|---|
| 0.8B | 约 14-17GB | 5090 舒适，端到端峰值预计约 18-26GB |
| 2B | 约 36-45GB | 单 5090 必须 offload；PRO 6000 更适合正式实验 |
| 4B | 约 75-93GB | PRO 6000 仍需 offload；单 5090 不可行 |

该表是容量规划，不是显存承诺。最终以 G2 的完整 step 峰值为准。

### 6.2 推荐顺序

1. 先租 1 张 5090，完成 0.8B 的 G0/G1。
2. 继续用该 5090 对 2B 做 3-step 容量测试。
3. 如果峰值超过 30GB、offload 导致单步过慢或主机内存抖动，切换 1 张 PRO 6000 96GB。
4. 只有 PRO 6000 缺货或实时报价明显高于 2 张 5090 总价，才测试双 5090。
5. 4B 默认直接使用 PRO 6000。

### 6.3 为什么不首选双 5090

- 两张 32GB 不是透明统一的 64GB。
- RTX 5090 无 NVLink；常见 GeForce RTX 50 平台不提供 CUDA GPU-GPU P2P。
- FSDP/NCCL 通信和频繁权重同步可能经过主机内存，GPU-hour 效率与稳定性通常不如单张 96GB。
- 4B 即使参数分片，每卡仍需要 offload 才能给激活与 rollout 留空间。

如果使用双卡，正式训练前必须记录：

```text
nvidia-smi topo -m
nvidia-smi topo -p2p p
CUDA p2pBandwidthLatencyTest
nccl-tests all_reduce_perf
```

只有 3 个完整 RL step 证明双卡稳定且“每 step 实际费用”低于 PRO 6000，才进入长跑。

### 6.4 CPU、内存和磁盘

- 0.8B / 单 5090：16 CPU cores，64GB RAM 起步，推荐 96GB。
- 2B / PRO 6000：24 CPU cores，128GB RAM 起步。
- 2B / 单 5090 offload：推荐 128GB RAM，否则容易把显存问题转化为 swap 问题。
- 4B / PRO 6000：推荐 192GB 以上 RAM。
- 持久数据盘：2B 推荐 200-250GB；若保留 4B 可恢复 optimizer checkpoint，推荐 300GB。
- Ray 临时目录必须放到数据盘，不能使用容量较小的默认 `/tmp/ray`。

## 7. 软件环境

### 7.1 基础镜像选择

选择：

```text
PyTorch 2.12.1 / Python 3.12 / Ubuntu 22.04 / CUDA 13.0
```

理由：

- 5090 与 PRO 6000 Blackwell 均为 `sm_120`，CUDA 13 路线更适合当前 Blackwell 软件栈。
- PyTorch 2.12 已把 cu130 作为主要 CUDA 路线；Linux 宿主驱动至少应为 580.65.06，并以镜像对应 wheel 的官方要求为准。
- 当前 vLLM 0.25.x 与 SGLang 0.5.15 系列通常锁定 Torch 2.11/CUDA 13 生态；CUDA 12.8/PyTorch 2.8 并不更贴合 Qwen3.5。

镜像中的 PyTorch 2.12.1 只作为基础，不作为最终 Python 环境。应新建隔离的 uv/venv，避免扩展 ABI 被预装包污染。

### 7.2 两个隔离环境

#### A. `remem-hf`：首版训练和正确性验证

建议锁定：

- Python 3.12
- Torch 2.11.0 cu130
- Transformers 5.14.x
- `flash-linear-attention`：固定到包含 Blackwell backward 修复的明确 commit
- `causal-conv1d`：固定 build/revision
- FlashAttention 2 或 SDPA：以 20-cycle smoke 结果选择
- 与本仓库兼容的 Ray、Hydra、TensorDict、datasets、pyarrow 等完整 lock

Transformers 原生支持 Qwen3.5 始于 5.2；若依赖 composite config 的 `AutoModelForCausalLM.from_config` 解包，应使用 5.13 以上。首版用 5.14.x，减少已知的 config 解包问题。

#### B. `remem-engine`：可选的高吞吐 rollout/评测

- vLLM 0.25.x
- Torch 2.11.0 cu130
- Transformers 和 CUDA Python 版本以该 vLLM/现代 verl 的已验证 lock 为准
- 只在现代 verl wrapper 或逐 API 移植通过后使用

不要把首版 HF 环境与 vLLM/SGLang 的依赖强行混装。最终锁文件必须来自实际通过 smoke 的实例，并保存 `pip check` 与 `pip freeze`。

### 7.3 Rollout 后端决策

首选顺序：

1. 先完成第 8.5 节的 HF rollout 接口修复，再用 `actor_rollout_ref.rollout.name=hf` 完成 0.8B 与 2B 3-step benchmark。
2. 若 HF rollout 预计能在预算内完成 2B，则继续使用，优先保证算法复现。
3. 若 HF rollout 明显过慢，则把 ReMemR1 的 recurrent manager、reward 和 dataset 移植到支持当前 vLLM 的现代 verl。
4. SGLang 0.5.15 只作为后续备选。

不能把仓库的 `sglang==0.4.6` 直接升级到 0.5.15：当前代码导入私有 `sglang.srt.entrypoints.verl_engine.VerlEngine`，该接口已在后续 SGLang 中删除，而旧 0.4.6 又早于 Qwen3.5 支持。vLLM 路径同样依赖旧私有构造和权重同步 API，升级前必须逐项移植。

SGLang 0.5.15 还引入 FA4 依赖，而仓库训练端使用 FA2 旧模块，可能产生包命名空间和 API 冲突，因此不作为第一选择。

## 8. Qwen3.5 兼容改造清单

以下工作是训练前置门槛，不是可选优化。

### 8.1 模型加载

[fsdp_workers.py](../verl/workers/fsdp_workers.py) 和 [model_merger.py](../scripts/model_merger.py) 使用了 Transformers 5 已删除的 `AutoModelForVision2Seq`，必须改造。

首版推荐纯文本 actor：

- 使用 `AutoModelForCausalLM` 从官方 Qwen3.5 checkpoint 提取语言模型。
- 使用 Transformers 原生实现并保持 `trust_remote_code=false`。
- 明确记录并白名单化被忽略的 `model.visual.*` 与 `mtp.*` 权重；不能忽略其他未知 key。
- 若所用 Transformers 版本不能自动解包 composite config，显式使用 `config.text_config`。
- `attn_implementation` 改为配置项，允许 `sdpa` 与经过验证的 flash 路径切换，不能硬编码 Qwen2 的 FA2 假设。
- reference 可直接以 BF16 初始化。actor 在 0.8B/5090 和 2B/PRO 6000 上保留 FP32 master parameter 与 optimizer，再用 FSDP BF16 mixed precision 计算；单 5090 的 2B 若需要更低初始化峰值，应实现流式加载和 CPU optimizer offload，而不是静默得到 BF16 Adam state。

加载后做两项一致性检查：

1. 官方 conditional model 的纯文本 logits 与提取后的 CausalLM logits 在固定短输入上数值一致。
2. 保存、合并、重新加载后 logits 仍一致。

### 8.2 Checkpoint 包装

纯文本 actor 保存后通常会成为 `Qwen3_5ForCausalLM` / text config。当前稳定 vLLM/SGLang 对官方 `Qwen3_5ForConditionalGeneration` 支持更成熟，对纯文本保存格式的 registry 支持可能滞后。

因此首轮评测也使用 Transformers。若后续改用 vLLM，必须选择并验证一种方式：

- 保留官方 composite config/参数命名，使用 engine 的 language-model-only 模式；或
- 为 `Qwen3_5ForCausalLM` 增加明确的 engine registry/loader 适配。

权重同步前后都要做固定输入 logits 一致性测试，不能只以“服务能启动”作为成功。

### 8.3 Chat template

[recurrent/utils.py](../recurrent/utils.py) 与 recurrent chat-template 注册逻辑目前按 Qwen2 tokenizer 设计，需要：

- 透传 `enable_thinking=False`，关闭 Qwen 原生的 `<think>...</think>` 模式。
- 基于 Qwen3.5 官方模板保留 generation 标记。
- 避免覆盖为 Qwen2.5 专用模板。
- 为 system/user、单轮/多轮、训练/评测各增加 snapshot test。

这里必须区分两种“thinking”：Qwen 原生 thinking 模式要关闭，但论文任务 prompt 明确要求模型输出任务级 `<thinking>...</thinking>` action。首版保留后者，不把它误删；模板测试应确认没有原生 `<think>`，同时允许任务级 `<thinking>`。若使用 OpenAI-compatible engine，训练和评测请求都要传等价的 `chat_template_kwargs={'enable_thinking': false}` 或服务端固定模板。

### 8.4 Position、padding 与 kernel

- `use_remove_padding=false`。
- sequence parallel size 为 1。
- 禁用 Qwen2 monkey patch 与 Liger 中未经 Qwen3.5 验证的 patch。
- 测试 MRoPE/position ids 在纯文本路径的形状。
- 分别测试 GDN forward、GDN backward、cache generation 和 gradient checkpoint。
- 不把“FlashAttention 通过”当成“GDN 通过”；两者是不同 kernel 路径。

### 8.5 HF rollout

仓库已有 [hf_rollout.py](../verl/workers/rollout/hf_rollout.py)，但当前版本不能直接用于 recurrent 训练：[fsdp_workers.py](../verl/workers/fsdp_workers.py) 会传 `pad_to` 以及 `max_tokens`、`n` 等 generation kwargs，而 `HFRollout.generate_sequences` 目前只接受 `prompts`，第一次调用就会产生 `TypeError`。此外，它按固定 `config.response_length` 补齐，不能正确反映逐轮长度。

G-1 必须先完成：

- 让 `generate_sequences` 接受 `pad_to` 和受控的 `**generation_kwargs`。
- 将调用级 `max_tokens` 映射为本次 `max_new_tokens`。
- 尊重调用级 `n`，避免在 recurrent manager 已展开轨迹后再次按全局 group 倍增。
- 按本次 `pad_to`/response length 构造 sequences、position ids、attention mask 和 response mask。
- 对“无 kwargs、短中间状态、最终状态、`n=1`、group rollout”分别增加单元测试。

接口修复后再验证：

- FSDP `summon_full_params` 在 1 卡和 2 卡均不 hang。
- Qwen3.5 `generate(use_cache=True)` 与 GDN cache 正常。
- `num_return_sequences=8` 不产生不可接受的显存峰值。
- 生成后的 position ids、attention mask、response mask 正确。
- 切回 train mode 后梯度和 optimizer state 正常。

若双卡 HF rollout 需要每次收集完整参数并导致严重通信开销，直接改用单卡 PRO 6000，不在双 5090 上长期排障。

### 8.6 Callback 模式与训练条件 A

当前评测代码虽然暴露了 `nocallback` 参数，但实际检索路径没有使用它，fixed-question 模式也未实现，入口还检查了不匹配的 `recurrent_revisit` API 名；训练端每轮都会解析 query 并执行检索。因此第 4 节的 A 条件和三种推理消融都属于前置开发任务，不能只改配置名称。

应新增统一的 `callback_mode={learned,none,fixed_question}`：

- `learned`：解析 `<recall>` 并用生成 query 检索。
- `none`：不执行历史检索，也不把召回内容注入下一状态。
- `fixed_question`：始终以原始问题检索。

推理消融可共用 ReMemR1 prompt；但训练条件 A 要使用单独的 sequential agent/prompt，从 action space 中移除 recall 指令和 query 生成，而不是生成后丢弃。A、B 固定相同最大 response budget，并额外报告实际生成 token 和 GPU-hour，避免把 A 的低成本误当成算法收益。

用一个含唯一历史证据的确定性样例验证三种模式得到不同 query/召回结果，并检查训练和评测使用同一枚举语义。

### 8.7 Transformers 评测 runner

当前 [run_eval.py](../taskutils/memory_eval/run_eval.py) 只会启动 SGLang，单纯参数化 GPU 数不能实现“首轮使用 Transformers”。G-1 需要新增直接加载合并 CausalLM 的 Transformers runner，复用相同 recurrent prompt、callback modes、逐样本输出与 metric 计算。

- 小规模 G1/G2 和首轮 32-sample 评测使用该 runner。
- runner 的每次 chat-template 调用显式关闭 Qwen 原生 thinking。
- 若后续切到 vLLM/SGLang，使用同一固定样例做逐步 query、memory 和最终 logits/文本对齐。
- 现有 OpenAI-compatible 评测入口只在 engine checkpoint loader 通过后使用。

### 8.8 Ray 临时盘与 checkpoint

[main_ppo.py](../verl/trainer/main_ppo.py) 当前把 Ray `_temp_dir` 硬编码为 `/tmp/ray`。应改为读取 `RAY_TMPDIR`（缺省时才回退到 `/tmp/ray`），并用单元测试或启动日志确认实际路径。

当前 FSDP checkpoint manager 每次都会保存 optimizer 等完整恢复状态，不会自动产生轻量 model-only checkpoint。因此正式流程为：

1. `save_freq=40`，通过 retention 只保留最新 1 份完整恢复状态。
2. 在 40/final checkpoint 上单独运行 merge/export，生成可评测的 model-only 目录。
3. 导出和重载 logits 校验通过后，按保留策略删除旧导出；不要把 `save_best_val=false` 描述成自动保留 best。

## 9. AutoDL 无 GPU 准备阶段

不挂载 GPU 时完成：

1. 固定仓库 commit、论文 v5、数据 revision 和模型 revision。
2. 建立 `remem-hf` 环境，生成 lock、`pip check`、`pip freeze` 和 wheel 清单。
3. 把 `HF_HOME`、`HF_DATASETS_CACHE`、`PIP_CACHE_DIR`、`TORCH_EXTENSIONS_DIR`、`RAY_TMPDIR`、数据、checkpoint 和日志全部放到持久盘。
4. 下载 Qwen3.5-0.8B 与 2B；4B 仅在已决定运行时下载。
5. 下载并校验 `hotpotqa_train_32k.parquet` 和 `hotpotqa_dev.parquet`；训练 parquet 约 2.16GB。
6. 生成固定 manifest：smoke 8 条、pilot 64 条、正式训练池 512 条。
7. 生成 HotpotQA 与 2WikiMultiHopQA 的 200/800-document 评测集，以及 800-document distant-evidence 子集。
8. 首轮每个评测格只取固定 32 条；需要时扩到 64 条。
9. 运行 parser、TF-IDF、奖励函数、dataset、完整 Hydra config 与 chat-template 的 CPU 测试。
10. 完成 Qwen3.5 模型加载、HF rollout 接口、callback modes、Transformers eval runner、Ray 临时盘、merger 和小 batch 边界 bug 的代码改造。
11. 记录所有文件 SHA256、模型 revision、样本 ID 与完整配置。

无 GPU 阶段可以编译/缓存 wheel，但不能证明 Blackwell kernel 可运行。GPU kernel、NCCL、显存和吞吐测试必须留到短时 GPU smoke。

不要运行 [run_memory_debug.sh](../run_memory_debug.sh)。该文件包含作者机器路径和明文 W&B API key；该 key 应视为已泄露并由所有者吊销或轮换。

## 10. GPU 阶段与门控

### 10.1 硬件 smoke

挂 1 张 5090 后依次检查：

1. 驱动、`nvidia-smi`、CUDA runtime。
2. `torch.cuda.get_device_capability()` 返回 `(12, 0)`。
3. BF16 matmul 和短 backward。
4. full-attention kernel 的 forward/backward。
5. GDN/FLA 的 forward/backward。
6. Qwen3.5-0.8B `generate(use_cache=True)`。
7. Ray 单 worker 和持久盘临时目录。

### 10.2 G0：20-cycle kernel 稳定性

固定同一批短输入，连续执行 20 次：

```text
forward -> loss -> backward -> optimizer.step -> zero_grad
```

记录每轮耗时、峰值显存、loss、gradient norm 和是否出现 NaN/hang。不能只运行一次 forward。

### 10.3 G1：完整 RL smoke

配置：

- 1-2 个问题。
- group 4。
- 2 chunks。
- 最大生成 128-256。
- 1 个完整 optimizer step。
- 保存 1 次 checkpoint。

必须覆盖：

- 模型初始化。
- recurrent rollout。
- `<update>` 与 `<recall>` 解析。
- TF-IDF callback。
- outcome/state/format reward。
- old/ref log-prob。
- backward 与 optimizer step。
- checkpoint 保存、恢复、合并。
- 合并模型单条端到端评测。

### 10.4 G2：2B 三步容量测试

从同一基础 checkpoint 启动两个独立任务：G2a 使用 group 4 跑 1 step，G2b 使用 group 8 跑 2 steps；二者都设置 `save_freq=-1`，使用正式的 `5000 x 6` 与输出 512，并分别记录：

- 中位 step time。
- rollout、reward、log-prob、update 各阶段耗时。
- GPU 峰值和 reserved memory。
- CPU RAM、数据盘读写与 offload 流量。
- 每 step 实际实例费用。

如果单张 5090：

- 峰值超过约 30GB；
- 出现频繁 OOM 重试；
- CPU RAM 接近耗尽；
- 或 projected cost 高于 PRO 6000；

则停止该路线，切换 PRO 6000，不通过继续缩小 group 来掩盖正式配置不可行。

### 10.5 正式训练

- 每个条件使用独立且唯一的输出目录。
- step 0-40 首段使用 `resume_mode=disable`；不得依赖 `auto` 猜测目录。
- step 40 外部验证完成后，使用 `resume_mode=resume_path` 和该条件的明确 `global_step_40` 路径续到 80；可选 120 段同理从 `global_step_80` 续训。
- 每段终点保存完整恢复 checkpoint，通过 retention 只保留当前最新 1 份；确认下一段已成功恢复后再清理旧 checkpoint，随后按第 8.8 节单独 merge/export model-only。
- 训练中不运行完整 5-cell 评测。
- 每 10 steps 只汇总当前训练 batch 的格式、reward、advantage、gradient norm 和输出长度，不把它称为验证集结果。
- 在 40 steps 和 final 暂停训练，用固定小验证 manifest 做外部评测；这与 `trainer.test_freq=-1` 一致。
- G1 额外执行一次从 `global_step_1` 续到 step 2 的恢复 smoke，确认 optimizer、scheduler、数据位置和 global step 均连续。
- 任一条件发生 NaN、格式崩溃或 outcome reward 连续为零时，三个条件一起暂停排查。

## 11. 评测与统计

### 11.1 核心评测格

| 数据集 | documents | 作用 |
|---|---:|---|
| HotpotQA | 200 | ID、接近训练长度 |
| HotpotQA | 800 | ID 长上下文外推 |
| 2WikiMultiHopQA | 200 | OOD |
| 2WikiMultiHopQA | 800 | OOD 长上下文 |
| 2WikiMultiHopQA distant-evidence | 800 | callback 与远距离证据专项 |

第一轮每格 32 个固定样本；方向接近 0 或置信区间过宽时扩到 64。所有条件使用相同 QA ID 和上下文构造。

### 11.2 解码

- 训练 rollout：temperature 1.0。
- 主评测：与仓库口径一致的 temperature 0.7、top-p 0.95。
- 若增加 greedy 评测，单独成表，不能与采样结果混合。
- 所有 Qwen3.5 评测显式 `enable_thinking=False`。
- 尽可能固定服务端 seed，并记录后端和版本。

### 11.3 必报指标

- Exact Match。
- Token F1。
- Substring EM。
- 最终 `\boxed{}` 解析成功率。
- 中间 `<update>` 与 `<recall>` 格式成功率。
- callback 触发率。
- 空 query、重复 query、无关 query 比率。
- 检索内容包含答案词/支持事实的比例。
- memory 中答案实体的保留率。
- 每样本、每 chunk、每完整 trajectory 耗时。
- 峰值 GPU 显存、主机内存、checkpoint 大小。
- 200 到 800 documents 的性能下降幅度。

### 11.4 统计

- 保存逐样本预测、完整 response、callback query 和检索结果。
- 对同一批样本做 paired bootstrap 95% confidence interval。
- 对 EM 使用 McNemar exact test。
- 预注册主指标只对 HotpotQA/2WikiMultiHopQA x 200/800 的 4 个完整格做等权 macro average，并同时展示各格。
- distant-evidence 是 2Wiki 800 的专项子集，只作为 callback 次指标，不再进入主 macro，避免重复计权。
- 单 seed、32/64 样本的结果只能称为方向性趋势。

## 12. 预算与降级

AutoDL 5090 公开页面在方案核对日前后的普通价约为 2.93-3.14 元/卡时，创建实例时仍以实际整机报价为准。PRO 6000 的公开实时价无法稳定核实，不在方案中虚构固定单价。

以下是容量规划范围，不是报价承诺：

| 项目 | 粗略实例时 |
|---|---:|
| 0.8B 环境、20-cycle、完整 RL smoke | 1-4 小时（单 5090） |
| 0.8B 两条件各 50 steps | 合计约 6-15 小时（单 5090） |
| 2B 两条件各 50 steps | 合计约 14-35 小时 |
| 2B 两条件各 80 steps | 合计约 22-56 小时 |
| 2B 第三个同长度训练条件 | 在两条件基础上再增加约 50% |
| 4B 10-20 step scale check | 必须由 3-step 实测外推 |

每次 smoke 后按整机价格重算：

```text
单条件剩余训练费 = (最近 3 个完整 step 的中位秒数 / 3600)
                 x 该条件剩余 step 数
                 x AutoDL 页面显示的元/实例时

预计总费用 = 各条件剩余训练费之和
           + 首次 kernel 编译/模型加载时间
           + checkpoint/merge/export 时间
           + 计划评测实例时
```

A/B/C 的 step time 不同，应分别估算后求和，不能用最快条件外推全部训练。

预算分配建议：

- 20%：环境与排障。
- 60%：正式训练。
- 20%：合并、评测和一次失败重跑。

硬件决策：

- 0.8B：选 1 张 5090。
- 2B：若 PRO 6000 价格不高于 2 张 5090 的整机总价，优先 PRO 6000。
- 4B：即使 PRO 6000 略贵于双 5090，也优先 PRO；避免无 P2P 和 OOM 返工。

超过预算时严格按以下顺序降级：

1. 删除可选作者 7B 复验。
2. 删除 4B scale check。
3. 正式 2B 从三个训练条件降为 B、C 两个条件。
4. 两个 2B 条件从 80 steps 同步降为 50 steps。
5. 评测维持 5 格，但每格从 64 降为 32。
6. 2B 仍不可承受时，改为 0.8B 的 B、C 各 50 steps。
7. 最后才把 `max_chunks` 从 6 降为 4，并在报告中改称 micro reproduction。

不优先缩短上下文，因为长程信息覆写和 callback 正是论文要验证的核心。

## 13. 当前仓库的其他阻塞点

### 13.1 安装说明不可直接使用

[README](../README.md#installation) 没有完整 lock，并混合了旧 vLLM、旧 SGLang 与 CUDA 12.6 安装方式。原建议的 vLLM 0.9、SGLang 0.4.6 均早于 Qwen3.5 支持，必须被本方案的隔离环境替代。

### 13.2 小 batch 边界 bug

[memory_revisit.py](../recurrent/impls/memory_revisit.py) 使用：

```python
active_indices = self.active_mask.nonzero().squeeze().cpu().numpy()
```

只剩一个 active sample 时会得到不可迭代的 0-D 标量，应改为 `.flatten()`，并增加单 active trajectory 测试。

### 13.3 评测入口

[run_eval.py](../taskutils/memory_eval/run_eval.py) 当前存在：

- 总 GPU 数写死为 8。
- ReMemR1 checkpoint 仍为占位路径。
- 默认遍历约 15 个模型和 16 个任务。
- concurrency 为 128/256。
- readiness 循环缺少合理硬超时。
- 默认 append 旧结果。
- 全部请求失败时可能除零。

该入口只作为后续 engine 评测路径：需要参数化为 1/2 卡，一次只加载一个模型，并显式传 tasks、checkpoint、GPU 数、并发、输出目录和 5-10 分钟启动 timeout，首轮 concurrency 设为 4-16。首轮 Transformers 评测由第 8.7 节的新 runner 承担，不能假设修改 GPU 数就会切换后端。

### 13.4 数据处理

[process_test.py](../taskutils/data_synthesis/process_test.py) 默认生成 50-6400 documents 全八档且使用 seed 42；论文评测口径为 seed 4。

需要支持：

- `--lengths 200,800`
- `--seed 4`
- 固定 sample manifest
- distant-evidence 数据
- 相同 QA ID 的跨长度上下文构造

### 13.5 日志、checkpoint 与临时盘

- 关闭在线 W&B，使用 console 或 offline。
- 修复脚本中的 W&B 占位配置，不泄露任何 key。
- `save_best_val` 关闭，`max_actor_ckpt_to_keep=1`。
- 可恢复 checkpoint 包含 optimizer，可能远大于 model-only 权重。
- 先按第 8.8 节移除 `/tmp/ray` 硬编码，再让 `RAY_TMPDIR` 指向持久盘，并限制 reward worker CPU 数。
- 每次实验使用新目录，避免默认 auto-resume 或旧评测结果混入。

## 14. 论文与代码口径差异

最终报告必须记录：

1. 论文基础模型是 Qwen2.5-3B/7B，本复现改为 Qwen3.5-2B，绝对精度不可直接对齐。
2. 论文要求评测 seed 4，当前数据脚本默认 seed 42。
3. 论文部分文字使用 `<callback>/<memory>`，代码和提示词主要使用 `<recall>/<update>`。
4. 论文正文部分写 `\box{}`，实际提示词和代码要求 `\boxed{}`。
5. 2Wiki 的部分 3B MemAgent 数字在不同表格间存在冲突。
6. 论文称检索开销低于 0.2%只指 TF-IDF 检索；callback query 的自回归生成仍有明显开销。
7. 当前评测代码主要使用第一条 gold answer；正式复现应对所有合法答案取最大分。
8. Qwen3.5 使用 Gated DeltaNet、MRoPE 与统一多模态 checkpoint，所做 text-only 适配和 backend 变化属于额外工程偏差。
9. 0.8B/2B 默认关闭、4B 默认开启 Qwen 原生 thinking；本复现三者都关闭原生 thinking，但保留论文任务级 `<thinking>` action。
10. 论文训练描述为约 6 个 chunks，而当前仓库部分默认配置会得到 8 个；本复现显式固定 `max_chunks=6`，以论文口径为准。

## 15. 验收标准与交付物

### 15.1 工程验收

- G-1 的 HF rollout 接口、三种 callback modes、Transformers eval runner 和 Ray 临时盘测试全部通过。
- G1/G2a/G2b/A/B/C 的完整 Hydra 配置可 `--cfg job --resolve`，且没有残留 8-GPU、GAE、大 batch 或 30-epoch 默认值。
- 0.8B 连续 20 个优化循环无 hang/NaN。
- 2B 至少完成 3 个正式长度 RL steps。
- callback、reward、backward、checkpoint、合并、重载、单条评测全部通过。
- 原模型、提取后的 CausalLM、合并后模型的固定输入 logits 对齐。
- 格式解析率在短验证集达到约 95% 后再长跑。
- 无无限 readiness、旧结果混入、静默忽略未知权重或全请求失败仍输出指标。

### 15.2 机制验收

优先满足：

- C（`alpha=0.8`）相对 B（`alpha=1.0`）在 4 个主评测格的 macro average 上方向更好。
- B 相对 A 在 800 documents 或 distant-evidence 上方向更好。
- learned callback 相对 no/fixed callback 至少在长上下文格子上有方向性收益。

若置信区间跨 0，只能写“方向一致但统计不足”。若没有趋势，应如实报告负结果，不能挑单一长度或 checkpoint 宣称成功。

### 15.3 实施阶段交付物

- AutoDL 无 GPU 准备脚本。
- `remem-hf` lock 与可选 `remem-engine` lock。
- Qwen3.5 模型加载、template、padding、HF rollout、merger 适配及测试。
- G1/G2a/G2b/A/B/C 的完整配置，以及 5090/PRO 6000 资源变体。
- 固定 train/validation/eval manifests。
- 单独的 sequential agent 与统一 `callback_mode`，预算版至少支持 B/C。
- learned/no/fixed callback 推理消融及确定性测试。
- 直接 Transformers 评测 runner 与参数化 engine 评测脚本。
- 可配置 Ray 临时目录和独立 model-only merge/export 流程。
- 环境、显存、耗时和费用记录。
- paired bootstrap、McNemar 与结果表生成脚本。
- 中文复现报告。

## 16. 推荐执行顺序

```text
无 GPU：锁环境、下载 0.8B/2B 和数据
  -> G-1：修 HF rollout、完整配置、callback modes、HF eval、Ray/checkpoint 流程
  -> G-1 CPU 单元测试与 Hydra --cfg job --resolve
  -> 单 5090：0.8B kernel/GDN 20-cycle
  -> 单 5090：0.8B 完整 RL smoke
  -> 单 5090：2B 正式长度 3-step 容量测试
  -> 成本/显存门控
       -> 单 5090 可接受：继续
       -> 边界或过慢：切换单 PRO 6000 96GB
  -> 2B B/C 各 1-step pilot
  -> 预算允许时加入 A 的 1-step pilot
  -> 各条件从零正式训练到 40，merge/export + 外部小验证
  -> 显式 resume_path：40 -> 80
  -> 预算允许时显式 resume_path：80 -> 120
  -> 合并、重载和 logits 校验
  -> 5 格评测与 callback 消融
  -> 可选 4B 10-20 step scale check
  -> 统计分析、实际费用与偏差报告
```

## 17. 版本依据

兼容性判断按 2026-07-16 的上游状态制定，执行时必须冻结 revision：

- Qwen3.5 模型卡：  
  <https://huggingface.co/Qwen/Qwen3.5-0.8B>  
  <https://huggingface.co/Qwen/Qwen3.5-2B>  
  <https://huggingface.co/Qwen/Qwen3.5-4B>
- PyTorch 2.12 CUDA 策略：  
  <https://pytorch.org/blog/pytorch-2-12-release-blog/>
- Transformers 5.2 的 Qwen3.5 支持与后续 composite-config 修复：  
  <https://github.com/huggingface/transformers/releases/tag/v5.2.0>  
  <https://github.com/huggingface/transformers/pull/45770>
- vLLM Qwen3.5 recipe：  
  <https://docs.vllm.ai/projects/recipes/en/latest/Qwen/Qwen3.5.html>
- SGLang 旧 `VerlEngine` 删除：  
  <https://github.com/sgl-project/sglang/pull/7326>
- FLA Blackwell 修复跟踪：  
  <https://github.com/fla-org/flash-linear-attention/pull/1000>
- RTX 5090 P2P 说明：  
  <https://forums.developer.nvidia.com/t/p2p-issue-using-two-rtx-5090-gpus/326776/8>
- AutoDL GPU 性能参考：  
  <https://www.autodl.com/docs/gpu_perf/>

上游版本变化很快。方案中的版本号不是“永远最新”，而是可复现起点；一旦 G0-G2 通过，应立即冻结 wheel、commit、镜像 ID 和完整 lock，正式训练期间不再升级。
