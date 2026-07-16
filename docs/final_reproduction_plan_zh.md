# ReMemR1 最终缩小复现方案：Qwen3.5-4B LoRA-GRPO

> **文档状态：唯一权威方案（Source of Truth）**
> 项目名称：**基于 Qwen3.5-4B LoRA-GRPO 的 ReMemR1 缩小机制复现**
> 平台：AutoDL
> 正式硬件：1 x RTX PRO 6000 Blackwell 96GB
> 论文：[2509.23040v5.pdf](../2509.23040v5.pdf)，ICLR 2026 版本
> 仓库设计基线：`cc514c092ca968a50c52cdcc2e2ba96362fce25a`
> 当前方案分支：`reproduction/qwen35-plan`
> 最后核对：2026-07-16

本文档是唯一权威方案。早期 2B 全参数方案已删除；
[reproduction_implementation_handoff_zh.md](./reproduction_implementation_handoff_zh.md)
只保留实施入口，不维护独立参数。后续实现、租卡、训练、评测和简历表述均以本文档为准。

---

## 0. 一页结论

### 0.1 最终固定决策

| 项目 | 最终选择 |
|---|---|
| 正式模型 | `Qwen/Qwen3.5-4B` post-trained checkpoint |
| 项目性质 | ReMemR1 **机制与趋势的缩小复现**，不是论文绝对数值复刻 |
| 训练方法 | BF16 mixed-compute LoRA-GRPO；不做全参数训练，不做 QLoRA |
| LoRA | rank 32，`lora_alpha=64`，dropout 0，bias none；text language model 的 `all-linear` |
| 正式硬件 | 单张 RTX PRO 6000 Blackwell 96GB |
| Rollout | 修复后的 Hugging Face rollout；首轮不迁移 vLLM/SGLang |
| 正式条件 B | learned callback，奖励 `algorithm.alpha=1.0`，只使用 outcome advantage |
| 正式条件 C | learned callback，奖励 `algorithm.alpha=0.8`，组合 outcome/state advantage |
| 正式训练 | B、C 独立从同一 base 初始化；各 0→40 steps，评测后再同步 resume 到 80 |
| 训练输入 | 每样本 200 documents、约 30K token；chunk `5000 x 6` |
| 生成长度 | 中间 memory/callback state 最多 768 token；final answer 最多 512 token |
| Batch | train batch 4，PPO mini batch 4，所有 micro batch 1，GRPO group 8 |
| 优化器 | LoRA LR `5e-6`；constant-with-warmup；所有正式分段 warmup 固定为 8 |
| 正式评测 | HotpotQA/2WikiMultiHopQA x 200/800 documents |
| 评测样本 | 40-step 每格 32 个固定 QA；80-step 每格 64 个固定 QA |
| Callback 消融 | learned / none / fixed_question |
| 工程门控 | 0.8B 验证 Qwen3.5/GDN/kernel；2B 验证 LoRA/FSDP/checkpoint；4B 验证正式容量 |

这里有两个名字相似但含义完全不同的参数：

- `lora_alpha=64` 是 LoRA 缩放系数；
- `algorithm.alpha=0.8/1.0` 是论文式 outcome/state advantage 混合权重。

任何配置、日志和报告都必须写全名，不能只写“alpha”。

### 0.2 为什么主线是 4B LoRA + PRO 6000

选择 4B 是为了提高一次得到可观察行为和可用 QA 效果的概率；选择 LoRA 是为了把优化器、梯度和可训练参数成本压到单卡范围；选择 96GB 显存不是因为“4B 权重本身需要 96GB”，而是为了同时容纳：

- 4B actor、reference、LoRA/FSDP 状态；
- 每条约 30K token 的 recurrent 输入；
- group 8 的多轨迹生成；
- 6 个中间状态和最终回答；
- old/reference log-prob、反向传播激活和运行时显存峰值；
- 足够的安全余量，避免第一次正式训练就依赖 CPU offload 或双卡通信排障。

LoRA 会改变论文原本的全参数优化口径，理论上也可能降低性能上限，但它不改变 ReMemR1 的状态转移、callback、奖励或 GRPO 对照逻辑。对“几百元预算、单次尽量成功、可面试展示”的目标，这是更合理的工程折中。

### 0.3 最终成功应长什么样

项目不是只赌“C 的一个准确率数字高于 B”。最终应形成多层证据链：

1. Base → LoRA-GRPO 后，任务格式、memory 更新和 callback 行为可测地改善；
2. 相同设置下，C（`algorithm.alpha=0.8`）与 B（`1.0`）的完整四格对比；
3. C checkpoint 的 learned / none / fixed_question 推理消融；
4. 从 200 到 800 documents 的退化曲线；
5. 固定 distant-evidence 样例中的 callback query、命中证据和最终回答轨迹；
6. 峰值显存、step time、总费用和失败门控记录。

如果 C 没有显著超过 B，也必须完整报告。只要工程闭环、控制变量、消融和负结果分析完整，仍是有效的缩小复现；不得筛掉不利格子或把负结果包装成论文结论。

---

## 1. 复现目标、边界和声明

### 1.1 目标

本项目要回答三个缩小版问题：

1. 在 Qwen3.5-4B 上，ReMemR1 的 recurrent memory + learned callback 能否被 LoRA-GRPO 训练起来？
2. 加入 step-level state reward（C）相对纯 outcome reward（B）是否带来更好的正确率、callback 行为或长上下文鲁棒性？
3. learned callback 相对禁用 callback 和固定问题检索，是否更能召回远距离支持证据？

### 1.2 可以和不可以声称的内容

完成后可以说：

> 在单张 RTX PRO 6000 96GB 上，将 ReMemR1 适配到 Qwen3.5-4B，并用 LoRA-GRPO 完成 200-document 长上下文训练闭环；通过 B/C 奖励对照和 learned/none/fixed callback 消融评估机制趋势。

不能说：

- “完整复现了论文结果”；
- “复现了论文的 Qwen2.5-3B/7B 全参数训练”；
- “达到论文收敛”；
- 只凭单个样例或单个评测格声称 state reward/callback 一定有效；
- 把 Qwen3.5、LoRA、不同输出长度带来的差异归因于 ReMemR1 本身。

### 1.3 最低验收层级

| 层级 | 必须交付 |
|---|---|
| 工程闭环 | load → recurrent rollout → reward → backward → adapter update → save/resume → export/merge → eval |
| 对照闭环 | Base、B、C 使用固定 base revision、数据顺序、seed、步数和解码配置 |
| 机制闭环 | learned / none / fixed_question 真正改变检索动作，不是只改输出字符串 |
| 结果闭环 | 四格表、训练曲线、行为指标、案例、资源成本均可复查 |
| 表述闭环 | 明确“Qwen3.5-4B LoRA 缩小机制复现”，不冒充原论文训练规模 |

---

## 2. 论文方法：本项目究竟在复现什么

以下口径来自论文第 2 节、图 2/图 4、式 (6)-(9) 和附录 C。

### 2.1 Recurrent memory 与 callback

一个样本包含问题 `Q`、所有合法答案集合 `Y` 和长文档 `C`。文档被切成
`c0...c(T-1)` 顺序输入。普通“memorize while reading”仅保留当前 memory；ReMemR1 将状态扩展为当前 memory 和 callback query：

~~~text
question + previous memory + current chunk + recalled history
                            |
                            v
              new memory + callback query
                            |
                            v
                  retrieve old memories
                            |
                            v
                    next recurrent step
~~~

每一步模型既更新 memory，也可以生成 query 检索历史 memory。最终阶段使用最新 memory 和历史 memory 生成答案。核心贡献是让原本只向前覆盖的 memory 轨迹能够“回看”，形成非线性证据路径。

### 2.2 多级奖励

论文将监督拆成两层：

- **Outcome reward**：在终止状态判断最终答案是否正确；
- **State reward**：在每个中间状态评价 memory 信息增益、callback 检索增益和格式正确性。

State reward 由三部分相加：

~~~text
R_state,t = r_memory,t + r_callback,t + r_format,t
~~~

其中 callback reward 比较“加入检索内容前后，对合法答案的 recall 是否增加”；format reward 检查中间状态标签和最终 boxed answer 格式。

同一问题采样 G 条轨迹。Outcome reward 在轨迹组内中心化，state reward 在相同步骤的状态之间中心化。仓库对应设置是 `grpo_use_adv=False`，即遵循论文描述不再除以标准差。总体 advantage 为：

~~~text
A_t = algorithm.alpha * A_out
    + (1 - algorithm.alpha) * A_state,t
~~~

因此：

- B：`algorithm.alpha=1.0`，保留 learned callback 的动作空间，但不给 state reward 权重；
- C：`algorithm.alpha=0.8`，80% outcome advantage + 20% state advantage。

B/C 的差异只应是奖励混合权重；模型、数据、初始化、生成、训练步数和评测都必须一致。

### 2.3 论文原始训练规模

论文附录 C.2 给出的主设置是：

| 项目 | 论文 |
|---|---:|
| 基座 | Qwen2.5-3B/7B Instruct |
| 训练 | BF16、FSDP、仓库路径为全参数 actor 优化 |
| 训练数据 | HotpotQA；每样本 200 documents，约 30K token |
| Chunk | 5000 token x 约 6 |
| Train / micro batch | 128 / 8 |
| GRPO group | 16 |
| 中间和最终最大生成 | 2048 token |
| Actor LR / warmup | `1e-6` / 20 steps |
| KL / clip | 0.001 / 0.2 |
| 收敛步数 | 200-300 |
| 3B 资源 | 16 x H800，约 100 小时 |
| 7B 资源 | 32 x H800，约 80 小时 |

原论文资源约为 1600/2560 H800-GPUh，无法在本项目预算内原样复刻。

### 2.4 保留项与变化项

| 维度 | 本项目是否保持 | 说明 |
|---|---|---|
| Recurrent memory 流程 | 目标保持 | 顺序读 chunk、更新 memory、最终作答；历史容器需从无序 set 修为有序记录 |
| Learned callback | 目标保持 | 生成 query、检索历史 memory、回注下一状态；检索器需恢复论文 word-recall |
| Outcome/state reward | 修复后保持 | 当前 parser/max-vs-average 与论文不完全一致；G-1 方程级对齐后 B/C 才能检验 alpha |
| GRPO + reference KL | 保持 | group 缩为 8，KL 仍为 0.001 |
| 200 docs / 约 30K token | 保持 | 正式训练不再缩短到玩具上下文 |
| `5000 x 6` | 保持 | 保留论文训练的 chunk 粒度 |
| 基础模型 | 改变 | Qwen2.5 → Qwen3.5-4B |
| 参数更新 | 改变 | 全参数 → LoRA rank 32 |
| Batch/group | 缩小 | 128/16 → 4/8 |
| 生成上限 | 缩小 | 2048 → 中间 768、最终 512 |
| 训练步数 | 缩小 | 200-300 → 40，门控后 80 |
| 硬件 | 缩小 | 多机 H800 → 单卡 PRO 6000 96GB |
| Rollout backend | 改变 | 论文 SGLang → 首版修复后的 HF rollout |

即使跑满 80 steps，每个条件也只有 `80 x 4 = 320` 次 prompt exposure 和
`320 x 8 = 2,560` 条 trajectory。论文约有 25,600-38,400 次 prompt exposure、
409,600-614,400 条 trajectory；本项目分别缩小约 80-120 倍和 160-240 倍。因此它只能检验
机制和方向性趋势，不能声称达到论文收敛或复现论文效果量。

---

## 3. 为什么 LoRA 可行，以及它会改变什么

### 3.1 可行性判断

LoRA 不改模型前向结构，只在选定线性层旁增加低秩增量。ReMemR1 的关键机制发生在轨迹生成、历史检索和奖励计算层，因此可在 LoRA actor 上保持。rank 32、alpha 64 对 4B 模型是一个偏稳妥而非极限压缩的设置。

预期变化：

- optimizer state、可训练梯度和 checkpoint adapter 体积显著下降；
- 冻结 base 降低灾难性漂移风险，较适合短步数 RL；
- 训练速度仍不会“非常快”，因为主要成本还包括多轮自回归生成、reference/old log-prob 和长上下文前向；
- 表达上限可能低于全参数训练，尤其是需要大幅改变输出协议时；
- 论文绝对准确率和收敛步数不可直接比较。

### 3.2 为什么不做 QLoRA

首轮不做 4-bit/8-bit base：

- 96GB 已提供足够容量余量；
- Qwen3.5 Gated DeltaNet、PEFT、FSDP、量化 kernel 和 Blackwell 的组合会新增一组兼容风险；
- RL 中 reference/log-prob 数值一致性比 SFT 更敏感；
- 本项目优先“一次跑通并得到可信对照”，而不是最低显存纪录。

### 3.3 LoRA 精确范围

目标语义是：**text language model 内所有合适的 `nn.Linear`**，包括：

- Gated DeltaNet 的线性投影；
- full-attention 的 q/k/v/o 投影；
- MLP 的 gate/up/down 投影。

必须排除：

- token embedding 和 `lm_head`；
- vision encoder/projector；
- MTP 或其他非文本主干；
- 未经验证的 tied/output 模块。

实现不能仅照抄 Qwen2 的 `q_proj/v_proj` 列表。加载 text-only model 后先解析并保存 resolved module manifest；若 `all-linear` 的实际匹配结果不满足白名单/黑名单断言，立即失败。

---

## 4. Qwen3.5 适配原则

### 4.1 不是模型 ID 的简单替换

Qwen3.5 checkpoint 是统一 conditional-generation checkpoint，语言主干混合 Gated DeltaNet 和 full attention。当前仓库存在三类直接阻塞：

1. Transformers 5 已移除仓库使用的 `AutoModelForVision2Seq`；
2. 当前 loader 和 merger 按旧 Qwen2/普通 CausalLM 假设加载；
3. Qwen2 的 remove-padding/monkey patch 不可直接套到 Qwen3.5 GDN。

首版必须提供 text-only loader/merger。Transformers 5.14 已有 Qwen3.5 text CausalLM 映射，
因此优先使用原生 `AutoModelForCausalLM`/`Qwen3_5ForCausalLM` strict-load；
只有原生路径无法从固定 checkpoint 正确构建时，才实现自定义 key extraction：

- 优先验证官方 text mapping，不重复实现已有 loader；
- 若必须自定义，用明确白名单处理 visual/MTP keys，未知 key 直接报错；
- conditional model 与 text-only model 在固定纯文本输入上 logits 对齐；
- full-attention 先允许 SDPA，FlashAttention 只在 Blackwell smoke 通过后启用。

### 4.2 Thinking 的两种含义

所有 Qwen chat-template 调用显式设置 `enable_thinking=False`，避免 Qwen 原生思考模式生成思考内容。
固定 revision 的 0.8B/2B/4B tokenizer 实测都会在 assistant generation prompt 末尾保留
`<think>\n\n</think>\n\n`。这是官方模板用于 non-thinking 模式的**空哨兵**，不是模型生成的
thinking 内容；必须保留，不能为了追求字面上“没有 `<think>`”而从 token 序列中手工删除。

但 ReMemR1 任务协议自身的输出标签 `<thinking>`、`<update>`、`<recall>`，
以及 prompt 中的 `<memory>`/`<recalled_memory>` 包装仍按仓库语义保留。论文称其为
memory/callback，当前实现使用 update/recall；关闭的是 Qwen 原生 thinking，不是删除任务级中间状态。

需要做 snapshot test，覆盖 0.8B、2B、4B：

- 每次调用都显式传入 `enable_thinking=False`；允许固定官方模板的空 `<think></think>` 哨兵，
  但任何非空、未闭合或额外的原生 `<think>` payload 都失败；
- 任务标签仍完整；
- train、rollout、eval 使用完全相同的 template contract。

### 4.3 GDN、padding 和 kernel

正式固定：

- `use_remove_padding=false`；
- Ulysses sequence parallel size = 1；
- 禁用 Qwen2 monkey patch；
- gradient checkpointing 开启；
- BF16 compute；
- GDN/FLA 和 `causal-conv1d` 使用通过 sm_120 backward 验证的固定 revision；
- 不以一次 forward 成功替代连续 optimizer-loop 稳定性测试。

0.8B G0 必须同时覆盖 GDN 路径和 full-attention 路径的 forward/backward；否则不能说明 Qwen3.5 kernel 已可训练。

选择 SDPA 仍不足以绕开当前 `verl/workers/actor/dp_actor.py` 顶层对
`flash_attn.bert_padding` 的无条件 import。首版要把 padding helpers 改为
`use_remove_padding=true` 分支内的 lazy import，或提供无 FlashAttention fallback；在
`use_remove_padding=false` 时，PPO worker 必须能在没有可用 FlashAttention build 的环境中导入和启动。

---

## 5. LoRA、FSDP、rollout 和 checkpoint 设计

### 5.1 当前仓库不能直接开跑

仓库已有 SFT LoRA 参考代码：

- `verl/trainer/fsdp_sft_trainer.py` 使用 `get_peft_model` 和 `LoraConfig`；
- `verl/trainer/config/sft_trainer.yaml` 已有 rank、alpha 和 `all-linear`；
- `verl/utils/fsdp_utils.py` 已有 LoRA leaf wrap policy。

但 PPO actor 主链路尚未注入 PEFT，当前 HF rollout、FSDP 包装和 checkpoint merger 也不了解 adapter。正式训练前必须实现并测试，不能把 LoRA 当成几个 Hydra 参数。

### 5.2 Actor 与 reference

固定设计：

- 仅 actor 注入 LoRA；
- reference 是相同 revision 的纯 text base model，不挂 adapter；
- actor 的 base parameters 冻结，只有 adapter `requires_grad=True`；
- optimizer 参数列表只允许包含 `requires_grad=True` 参数；
- `use_orig_params=True`，支持同一根 FSDP 中冻结 base 与可训练 adapter 的 mixed `requires_grad`；
- 单卡首条稳定路线固定为 **root-only FSDP**，不启用 nested LoRA leaf auto-wrap，使其与 HF rollout 的
  `summon_full_params(..., recurse=False)` contract 一致；
- actor 首条稳定路径保留 FP32 original/master parameters，由 FSDP mixed precision 做 BF16 forward/backward；reference 直接 BF16；
- adapter/optimizer master state 保持 FP32；
- actor param offload、actor optimizer offload、reference param offload 首轮全部关闭。

现代码会按 `role='ref'` 无条件创建 `CPUOffload(offload_params=True)`。实现必须删除该
role hardcode，改为读取 reference 自己的 FSDP config；不能只在 YAML 写 `param_offload=false`。
若未来改用 nested wrap，必须同步实现递归 summon/adapter 管理并重跑 G1/G2，不属于首轮路线。
`use_orig_params=True` 不自动解决同一 handle 的混合 dtype；当前稳定路线让 actor original parameters
统一为 FP32。未来若把 frozen base 改成 BF16、adapter 保持 FP32，必须另行处理 dtype uniformity、
ignored states 或分离 wrap，并完整重跑 G1/G2。

“BF16 LoRA-GRPO”在本方案中指 BF16 mixed compute；为稳定性保留 FP32 actor master 并不等于做全参数训练。若完整 G2 峰值超过 80-85GB，才按第 12 节顺序引入 offload/低精度优化。

构建时必须打印并保存：

- resolved target module names；
- 所有 trainable parameter names；
- trainable/total parameter count 和比例；
- base revision、Transformers/PEFT/Torch 版本；
- FSDP policy、dtype 和 offload 状态。

### 5.3 必须通过的 LoRA 不变量

以下断言任何一个失败都不得进入 G2：

1. LoRA 通常零初始化；step 0 时 enabled actor、disabled actor 和 reference logits 一致是预期；
2. 对 adapter 做确定性非零扰动或一次已验证 optimizer update 后，enabled logits 必须与 disabled/base 有非零差异；
3. 用 adapter forward hook/call counter 证明 `generate()` 走过 adapter；普通一步更新不强求 greedy token 改变，
   如需比较生成文本则使用足够大的确定性测试扰动；
4. 同一非零状态下 disable adapter 后，logits 必须回到 reference tolerance 内；
5. 一个 optimizer step 后至少一个 adapter tensor 非零变化；
6. 同一步后所有冻结 base tensor checksum 不变；
7. optimizer state 中不存在 frozen base parameter；
8. adapter save/reload 的固定输入 logits 与保存前一致；
9. merge-and-unload 后模型的 logits/生成与未 merge adapter 版本在预设 tolerance 内一致。

数值 tolerance 必须在测试代码中固定并记录；禁止在失败后临时放宽到“能过为止”。

### 5.4 HF rollout contract

当前 caller 会传入 `pad_to`、`max_tokens` 和 `n`，而
`verl/workers/rollout/hf_rollout.py` 当前只接受 `prompts`，第一次 recurrent 调用就会报
`TypeError`。修复后的接口至少为：

~~~python
generate_sequences(prompts, pad_to=None, max_tokens=None, n=None, **kwargs)
~~~

行为契约：

- `max_tokens` 控制本次真实最大生成，中间为 768、最终为 512；
- `pad_to=1024` 保持各 action tensor 可拼接；
- `n` 只在组尚未展开时生效；
- trainer 已将 prompt 展开为 group 8 后，每次 recurrent 调用必须用 `n=1`，不得再次生成 8 倍；
- HF generation micro batch 首版为 1；
- microbatch 分块使用 ceil 逻辑，active batch 不能整除时也不得超出配置；
- attention mask、position ids、EOS mask 和 response mask 与 padding 后长度一致；
- FSDP full-parameter context 下 adapter 必须保持启用；
- 单个 active sample 也必须正常返回。

正式首版不迁移 vLLM/SGLang。只有 G2 实测 HF rollout 使 40-step 双条件确定超出预算，且所有正确性测试已有 oracle，才单独立项迁移现代 engine。

### 5.5 Checkpoint 与导出

每个阶段终点或明确指定的 export point 产生两类产物。普通 step-20 恢复点不要求执行昂贵 merge。

**A. 可恢复训练 checkpoint**

至少包含：

- FSDP actor state（允许为可靠恢复而包含 frozen base）；
- adapter optimizer 和 scheduler；
- `global_step`；
- RNG、dataloader position 和数据 manifest hash；
- base revision、text-only mapping、LoRA config 和 resolved target manifest；
- 完整 resolved Hydra config。

恢复顺序必须是：按固定 revision 构建 text-only base → 注入相同 LoRA manifest → FSDP wrap → load actor/optimizer/scheduler/extra state。G1 和 G2 都要在**新进程**中验证。

**B. 便携推理产物**

- adapter-only `safetensors`；
- `adapter_config.json`；
- base model ID + revision；
- tokenizer/template revision；
- 可选的 `merge_and_unload()` BF16 model-only 目录。

当前普通 `scripts/model_merger.py` 不能正确解释 PEFT namespace，必须适配或增加专用 exporter。

当前 checkpoint manager 会在写新点之前执行 retention 删除，无法满足“新点验证后再删旧点”。首轮固定
`max_actor_ckpt_to_keep=null`，禁止训练进程自动 prune；只在新 checkpoint 完成 load、单条
generate 和 hash 校验后，才由显式工具手工安全删除旧 optimizer shard。后续若实现临时目录原子写入、
校验成功后 rename/prune，必须用保存失败的故障注入测试证明不会丢失最近恢复点。

### 5.6 统一协议 parser 与论文奖励对齐

当前仓库不能直接视为论文奖励的精确实现：

- state reward 对多个合法答案求平均，而论文式 (5)/(6) 对 `Y` 取最大；
- format reward 只检查 `<update>`，没有验证可选 `<recall>` 的唯一性/非空；
- update parser 只删除 recall wrapper，可能把 `<thinking>/<update>` 包装整体存入 memory；
- previous-memory regex 非 DOTALL，多行 memory 可能解析失败；
- 训练 reward parser 与外部 eval parser 不是同一个行为契约。

G-1 必须建立一个统一、结构化 parser，供 memory state、reward、日志和 eval 复用：

~~~text
IntermediateAction:
  thinking: optional/recorded
  update: exactly one non-empty payload
  recall: zero or one non-empty query

FinalAction:
  boxed_answer: exactly one non-empty payload for strict format
~~~

Memory 和 history 只保存解析出的 `update` payload，不保存 thinking/tag wrapper。所有跨行标签使用
DOTALL、非贪婪和完整数量校验。Thinking 合法率可以作为指标，但除非论文奖励定义明确要求，不擅自把它加入
`r_format`。

实现一个参数方向明确的 `word_recall(a, b)`，严格按论文定义和式 (5)/(6) 的参数顺序计算：

- `r_memory,t`：新旧 memory 对合法答案的信息增益；
- `r_callback,t`：加入 recalled memory 后的额外信息增益；
- 两者都对所有 `y in Y` 取最大，不求平均；
- `r_format,t`：中间 update/可选 recall、最终 boxed answer 的协议合法性；
- `R_state,t = r_memory,t + r_callback,t + r_format,t`。

方程级 fixtures 必须能区分：max vs average、`recall(a,b)` 参数反转、多行 memory、空/重复 recall、
多个 boxed answers 和训练/评测解析差异。只有这些 tests 通过，B/C 才能称为论文式 alpha 对照。

### 5.7 有序 history 与论文检索器

当前 `history_memory = set()` 会折叠重复 memory，丢掉 step/order/provenance，并使并列结果受 hash
迭代顺序影响；当前 TF-IDF cosine 也不同于论文的 word-overlap recall。正式路径必须改成有序记录：

~~~text
MemoryRecord(step_id, update_text, source_chunk_ids, source_doc_ids)
~~~

要求：

- 保留每一步和重复项，不用 set 去重；
- 论文主线使用 `E(X, q) = argmax_x word_recall(q, x)`；
- 相同最高分按最小 `step_id` 确定性 tie-break，并在报告中声明这是实现细节；
- retriever 返回完整 record，下一轮只注入 `update_text`；
- 日志保存 selected step、score、query 和 provenance；
- none/fixed_question/learned 共用同一个检索函数；
- TF-IDF 若保留，只能作为额外偏离实验，不能用于主 B/C。

这既是论文机制对齐，也是 callback distance、重复 state 和 supporting-document proxy 可计算的前提。

---

## 6. AutoDL 环境与资源

### 6.1 镜像选择

选择用户给出的：

~~~text
PyTorch 2.12.1 / Python 3.12 / Ubuntu 22.04 / CUDA 13.0
~~~

它作为驱动、CUDA toolkit 和系统基础。项目使用隔离环境，首选兼容矩阵目标：

| 组件 | 目标 |
|---|---|
| Python | 3.12 |
| Torch | 2.11.x cu130；以 AutoDL/官方可安装 build 为准并锁定完整版本 |
| Transformers | 5.14.x |
| PEFT | 与 Transformers 5.14.x 兼容的明确版本 |
| flash-linear-attention | 包含 Blackwell/sm_120 backward 修复的明确 commit |
| causal-conv1d | 通过 sm_120 测试的明确 revision/build |
| Full attention | 首先 SDPA；FlashAttention 通过 smoke 后再启用 |
| Ray/Hydra/TensorDict | 以完整测试通过的版本生成 lock |

不直接照抄 README 的旧 vLLM 0.9、SGLang 0.4.6 或 CUDA 12.6 安装组合。CUDA 12.8 / PyTorch 2.8 只作为明确验证后的回退，不在训练中途临时切换。

安装完成后必须保存：

- lock/requirements；
- `pip freeze`；
- `torch.__version__`、CUDA runtime/driver；
- GPU 型号、compute capability（应含 `(12, 0)`）；
- 关键 kernel commit 和 build log。

### 6.2 实例规格

正式实例建议：

| 资源 | 最低 | 推荐 |
|---|---:|---:|
| GPU | 1 x PRO 6000 96GB | 同左 |
| CPU | 24 cores | 32 cores 或以上 |
| RAM | 128GB | 192GB |
| 持久盘 | 250GB | 300GB 或以上 |

0.8B/2B gate 可使用 1 x RTX 5090 32GB；若租卡切换成本或环境重建成本更高，也可直接在 PRO 6000 上完成所有 GPU gates。

双 5090 不是统一 64GB 显存，且会引入 FSDP 通信、设备拓扑和两份运行时余量。本项目不以双 5090 为正式首选。

### 6.3 路径与运行约束

无 GPU 阶段先设置到持久盘：

~~~bash
export HF_HOME=/root/autodl-tmp/cache/huggingface
export TORCH_HOME=/root/autodl-tmp/cache/torch
export RAY_TMPDIR=/root/autodl-tmp/ray
export TMPDIR=/root/autodl-tmp/tmp
~~~

实际路径以 AutoDL 挂载点为准，但不得继续使用代码硬编码的 `/tmp/ray`。同时固定：

- `ray_init.num_cpus` 不超过实例可用 CPU；
- reward worker 数量有上限，不能直接使用全部 logical cores；
- W&B 默认关闭或 offline，正式结果至少保留 console + JSONL；
- 不运行、不打印 `run_memory_debug.sh`，其中的明文 W&B key 应视为已泄露并轮换；
- 模型、数据、checkpoint 和 eval outputs 均写持久盘。

---

## 7. 数据与 manifests

### 7.1 数据来源

训练使用仓库要求的 `hotpotqa_train_32k.parquet`，基础验证使用
`hotpotqa_dev.parquet`；评测生成 HotpotQA 和 2WikiMultiHopQA。

无 GPU 阶段完成下载、SHA256 和 schema 检查。任何自动下载都要有超时、重试和最终 hash，不能在 GPU 计费开始后才发现 2GB 数据或模型未缓存。

现有训练 parquet 主要消费拼接后的 `context`，现有 eval 生成器也不保证完整 doc ID、supporting-fact
provenance 或跨长度嵌套。若 schema 缺字段，必须在无 GPU 阶段从原始数据/生成步骤重建结构化 sidecar；
不能对一个已拼接字符串事后假装拥有可靠 provenance。

### 7.2 正式训练 manifest

主训练 seed 固定为 42。生成版本化 manifest，至少记录：

- 数据源文件 hash；
- QA ID 和顺序；
- 所有合法 gold answers；
- 200 个 document ID、顺序和 supporting-fact 标记；
- chunk token 边界；
- tokenizer revision；
- 随机 padding 文档的 seed。

每个 document 必须在拼接前获得稳定 ID；源数据没有 ID 时使用规范化文本 hash，并保存 title/text hash
映射。每个 5000-token chunk 保存覆盖的 document IDs/supporting-fact IDs。正式 dataset 构造时校验 sidecar
与拼接 context 的 hash 和顺序一致，否则拒绝训练。

建议先固定 512 条 formal-train manifest；80 steps x batch 4 只消费前 320 条，不发生 wrap。B/C 必须读取同一 manifest、相同顺序和相同 LoRA 初始化 seed。开发数据另建 manifest，绝不混入正式训练曲线。

正式输入约束：

- 200 documents；
- tokenizer 后约 30K token，超长按确定性规则处理；
- `chunk_size=5000`；
- `max_chunks=6`；
- `data.truncation=center`。

`MemoryDataset` 会把 dataset 级 `max_prompt_length` 改为 `6 x 5000`；配置审查不能误把问题 prompt 的 1024 token 上限当成总文档长度。

### 7.3 正式评测 manifest

为每个数据集固定 64 个 QA，建立 200-doc 和 800-doc 两个版本：

- 两种长度使用相同 QA ID 和相同 QA 顺序；
- 40-step 评测使用 64 条 manifest 的前 32 条；
- 80-step 评测使用完整 64 条；
- 数据生成器必须先为同一 QA 采样一个确定性的 800-document pool，再把其前 200 个文档构成 200-doc 版本；
- 两种长度都必须保存全部 doc ID、顺序、supporting-fact 位置和最终 context hash；
- Base/B/C 和三种 callback modes 读取完全相同的 manifest；
- 每次评测写全新输出目录，不 append 到旧结果。

评测脚本不再默认生成/运行 50-6400 全档、128 样本和 8 GPU 服务。只生成本方案需要的 200/800 manifests，避免无 GPU 阶段不必要的 CPU/RAM/磁盘峰值。

G-1 必须用 schema test 验证：64 个 QA 完全配对、200 是 800 的前缀、支持证据未丢失、doc/chunk provenance
可反查。现有生成器不满足时先修改生成器，不得把该要求降级为“尽量记录”。

评测不能复用训练的 `max_chunks=6` 上限。Eval runner 对每条样本按完整 tokenized context 动态计算
`ceil(context_tokens / 5000)`，直到消费 manifest 的全部 200/800 documents；任何模型/安全上限导致的
截断都必须让该样本失败，而不是悄悄只评前 30K token。

---

## 8. 最终超参数与配置契约

### 8.1 正式 B/C 公共参数

| 类别 | 参数 | 值 |
|---|---|---|
| Model | path | `Qwen/Qwen3.5-4B` + 固定 revision |
| Model | text-only / thinking | true / `enable_thinking=False` |
| LoRA | rank / alpha / dropout / bias | 32 / 64 / 0 / none |
| LoRA | target | text LM `all-linear`，按 manifest 排除项断言 |
| Precision | actor / compute / ref | FP32 master / BF16 mixed / BF16 |
| FSDP | `use_orig_params` | true |
| FSDP | offload | actor param=false，optim=false，ref param=false |
| Data | train batch | 4 |
| PPO | mini / micro / epochs | 4 / 1 per GPU / 1 |
| Log-prob | actor/ref micro | 1 / 1 per GPU |
| GRPO | estimator / group | grpo / 8 |
| GRPO | rollout config `n` / recurrent call `n` | 8 / 1 |
| Recurrent | question prompt / chunk / chunks | 1024 / 5000 / 6 |
| Recurrent | memory / final max tokens | 768 / 512 |
| Rollout tensor | `data.max_response_length` / `pad_to` | 1024 / 1024 |
| Token budget | actor/ref/rollout max per GPU | 12288 起步，G2 实测确认 |
| Sampling | temperature / top-p / top-k | 1.0 / 1.0 / 0 |
| Optimizer | AdamW LR / weight decay / grad clip | `5e-6` / 0.01 / 1.0 |
| Scheduler | style / warmup | constant-with-warmup / 固定 8 steps |
| GRPO | normalize by std | false（`grpo_use_adv=False`） |
| PPO | KL loss / coefficient / type | true / 0.001 / low_var_kl |
| PPO | clip low/high / entropy | 0.2/0.2 / 0 |
| Recurrent | action reweight | false |
| Reward | training metric | EM；对全部合法 gold 取最大 |
| Runtime | gradient checkpointing | true |
| Runtime | remove padding / Ulysses SP | false / 1 |
| Runtime | rollout backend / generation micro | HF / 1 |
| Runtime | `use_torch_compile` / dynamic batch | false / false |
| Validation | internal val | 关闭；使用外部固定 manifest runner |

B/C 唯一算法差异：

| 条件 | Callback | `algorithm.alpha` |
|---|---|---:|
| B | learned | 1.0 |
| C | learned | 0.8 |

**Seed contract**

- 增加唯一的 `reproduction.run_seed=42`，并显式派生 data、model-init、rollout seeds；
- 每个 Ray model worker 在加载/注入 PEFT 前统一设置 Python、NumPy、Torch CPU/CUDA RNG；
- 模型/adapter 初始化在各 rank 使用同一 model-init seed，分布式同步后保存初始 adapter hash；
- rollout RNG 按 run seed、global step、sample/trajectory index 确定性派生，B/C 使用相同规则；
- B/C pilot 和正式 run 在 step 0 断言 adapter state hash、首个 batch IDs 和首轮 sampled token hash 一致；
- resume 必须恢复 checkpoint RNG/global step，不重新执行 step-0 seed。

Fused kernel 可能不保证 bitwise 重现，但这不允许省略 seed 与初始 hash 契约；不可重复部分要单独记录。

### 8.2 分段训练配置

正式运行不是一个自动暂停的 80-step job，而是四个清晰任务：

| 任务 | 起点 | `total_training_steps` | 恢复模式 |
|---|---|---:|---|
| B40 | base + 同 seed LoRA init | 40 | disable |
| C40 | base + 同 seed LoRA init | 40 | disable |
| B80 | 明确的 B `global_step_40` | 80 | resume_path |
| C80 | 明确的 C `global_step_40` | 80 | resume_path |

所有四份配置仍写 `lr_warmup_steps=8`。恢复 scheduler/global step 后不会重新 warmup；绝不能在 80-step 配置中把 warmup 改成 16 或按第二段重新计数。

每份配置必须显式覆盖：

- `trainer.nnodes=1`、`trainer.n_gpus_per_node=1`；
- `reproduction.run_seed=42` 及派生 seed 字段；
- `trainer.total_epochs=1`；
- `trainer.save_freq=20`；
- `trainer.test_freq=-1`；
- `trainer.val_before_train=false`；
- `trainer.save_best_val=false`；
- `trainer.max_actor_ckpt_to_keep=null`，禁止 manager 在新点写成前删除旧点；
- `trainer.logger=['console']` 或受控 offline logger；
- `data.shuffle=false`；
- `data.truncation=center`；
- `data.filter_overlong_prompts=true` 和受控 worker 数；
- `recurrent.enable=memory`、`data.context_key=context`；
- `data.train_files=<formal manifest parquet>`、`data.val_files=<固定可构造的 validation parquet>`；
- `algorithm.adv_estimator=grpo`、`reward_model.reward_metric=em`；
- `actor_rollout_ref.actor.use_torch_compile=false`、`use_dynamic_bsz=false`；
- actor/reference 三个 offload 开关全部为 false，且运行时断言 reference 没有被 role hardcode 覆盖；
- checkpoint contents、base revision、LoRA manifest 和 extra-state 内容显式列出；
- `actor_rollout_ref.rollout.tensor_model_parallel_size=1`；
- `trainer.resume_mode=disable/resume_path`，不使用 auto；
- B80/C80 的 `trainer.resume_from_path` 必须是已校验的绝对 `global_step_40` 路径；
- 唯一的 `trainer.default_local_dir`、`project_name` 和 `experiment_name`。

目录必须编码模型、条件、seed、阶段和 run ID，例如：

~~~text
checkpoints/qwen35-4b-lora-r32/B/seed42/20260716-b40/
checkpoints/qwen35-4b-lora-r32/C/seed42/20260716-c40/
~~~

配置文件提交前逐份执行 `--cfg job --resolve`，检查没有残留 TP=2、8 GPU、30 epochs、GAE、
SGLang、Qwen2 patch、大 batch、默认 GSM8K 路径或空的正式 resume path。即使关闭 internal validation，
当前 trainer 仍会构造 val dataset，因此 `data.val_files` 不得为空或指向不存在文件。

---

## 9. 分阶段执行计划

任何 gate 失败都先修复并重跑该 gate，不带病进入下一阶段。

### 9.1 G-1：无 GPU 准备与代码完成

在 AutoDL 同一实例不挂 GPU 时完成：

1. 建立隔离环境和 lock 骨架，预下载 0.8B/2B/4B 模型及 tokenizer；
2. 下载/校验数据，生成 train/eval manifests；
3. 实现 Qwen3.5 text-only loader、key whitelist 和 merger；
4. 实现 PPO actor-only PEFT、target manifest、FSDP/optimizer contract；
5. 全链路显式 `enable_thinking=False`；
6. 实现统一协议 parser，并按论文式 (5)/(6) 修正 all-gold max、state/format/callback reward；
7. 把 history set 改为有序 MemoryRecord，主检索器改为确定性 word-recall；
8. 修复 HF rollout 的 `pad_to/max_tokens/n`、ceil microbatch 和 group contract；
9. 实现 learned / none / fixed_question callback modes；
10. 将单 active index 的 `.squeeze()` 改为 `.flatten()` 并加测试；
11. 为 `dp_actor.py` 的 FlashAttention padding import 增加 lazy/fallback 路径；
12. 配置 `RAY_TMPDIR` 和受控 CPU worker 数；
13. 改造数据生成器，生成嵌套 200/800、doc/chunk/supporting-fact provenance；
14. 实现 Transformers recurrent eval runner、硬超时和失败退出；
15. 实现多合法答案取最大分的 EM/F1/substring evaluator；
16. 实现 resumable FSDP checkpoint、adapter export、merge/reload；
17. 生成 G0、G1-step1/resume2、G2a、G2b-step1/resume2、B/C pilot、B40/C40/B80/C80 完整配置；
18. 对每份配置运行 `--cfg job --resolve`；
19. CPU 实例化 formal dataset，确认 truncation、schema、嵌套 manifest 和 provenance 全部通过。

G-1 验收：所有不需要 CUDA 的单元测试通过，所有链接/模型/data revisions 已固定，GPU 启动后不再做大下载或临时设计配置。

### 9.2 G0：Qwen3.5-0.8B kernel gate

硬件：1 x 5090 32GB 或正式 PRO 6000。

使用与正式路径相同的 text-only loader、LoRA/FSDP dtype、gradient checkpoint 和 kernel 组合，验证：

- CUDA capability 包含 `(12, 0)`；
- BF16 GDN/FLA 和 full-attention 各自 forward/backward；
- conditional model 与 text-only model logits 对齐；
- Qwen 原生 thinking 已关闭；
- 连续 20 次 `forward → loss → backward → optimizer.step → zero_grad`；
- 无 hang、NaN、非法内存访问和持续显存增长；
- 仅 adapter 参数变化，base checksum 不变。

一次成功不算通过；必须连续 20 个 optimizer loops。

### 9.3 G1：Qwen3.5-2B LoRA/FSDP/checkpoint gate

硬件：优先 1 x 5090 32GB；若 FP32 actor master + BF16 reference 的初始化/峰值接近上限，
直接在相同软件 lock 的 PRO 6000 上运行。G1 验证的是 2B 接口闭环，不是 5090 容量，不为省 gate
费用临时打开正式配置没有的 offload。

缩短配置：

| 参数 | G1 |
|---|---:|
| train / PPO mini batch | 1 / 1 |
| group | 4 |
| chunks | `1024 x 2` |
| memory/final output | 256 / 256 |
| micro batch | 1 |

G1 必须是两个可独立 resolve/执行的任务：

| 配置 | total steps | save freq | resume |
|---|---:|---:|---|
| G1-step1 | 1 | 1 | `resume_mode=disable` |
| G1-resume2 | 2 | 1 | `resume_mode=resume_path` + 绝对 `global_step_1` |

流程：

1. 从 base + LoRA init 跑 1 个完整 recurrent RL step；
2. 保存 `global_step_1`；
3. 结束进程；
4. 新进程显式 resume 到 step 2；
5. export adapter；
6. reload adapter 并跑 HF eval；
7. merge adapter，reload merged model，再做 logits/生成对齐。

G1 覆盖 PEFT/FSDP、adapter optimizer、scheduler、RNG/dataloader、checkpoint、跨进程 resume、export、merge 和 eval；0.8B kernel gate 不能替代它。

### 9.4 G2：Qwen3.5-4B 正式容量 gate

硬件：1 x PRO 6000 96GB。G2a/G2b 均从相同 base revision 开始，是两个独立任务。

**G2a：低风险启动**

- `algorithm.alpha=0.8`，覆盖完整 state-reward 路径；
- 先验证 4B conditional → text-only 固定输入 logits、strict missing/unexpected-key whitelist；
- 验证 text-only save/reload 后 logits 一致；
- train batch 1；
- PPO mini batch 1；
- group 4；
- `5000 x 6`；
- memory/final 768/512；
- 1 个完整 RL step。

**G2b：最终容量 + 恢复**

- `algorithm.alpha=0.8`；
- 使用第 8 节全部正式参数：batch 4、group 8、micro 1；
- `G2b-step1`：`total_training_steps=1`、`save_freq=1`、resume disabled；
- `G2b-resume2`：`total_training_steps=2`、`save_freq=1`、显式绝对
  `resume_from_path=.../global_step_1`；
- 第一个完整 step 后进程自然结束并保存；
- 结束进程并从该 checkpoint 显式 resume；
- 再跑 1 个完整 step；
- 导出 4B adapter-only，reload 并做 logits/生成对齐；
- 执行 `merge_and_unload()`，以 BF16 model-only reload 后再做 logits/生成对齐；
- 记录两类产物的 hash、文件大小、CPU RAM、GPU 峰值和耗时。

每一步记录 rollout、reward、old/ref log-prob、update、save 的分项耗时，以及 `nvidia-smi` 峰值和 PyTorch allocator peak。

进入正式训练的硬门槛：

- 无 OOM、hang、NaN、allocator retry 风暴或系统 swap；
- 建议峰值低于 80GB，80-85GB 需解释并保留额外监控，超过 85GB 先降风险；
- adapter gradient/更新非零，base checksum 不变；
- outcome/state reward 有合理方差，至少一组形成非零 advantage；
- 没有系统性格式失败或所有输出被截断；
- checkpoint 可在新进程恢复；
- 4B adapter export/reload 和 merged-model reload 均通过；
- reference 参数按配置常驻 CUDA，监控中没有每个 state 的 CPU↔GPU 搬运抖动；
- B/C 40-step 双条件的保守费用投影在预算内。

### 9.5 正式前 B/C pilots

B、C 各从相同 base 和相同 LoRA 初始化做独立 1-step pilot。Pilot 用于确认：

- 两条件只有 `algorithm.alpha` 不同；
- B/C 都有 reward variance 和有效 advantage；
- C 的 state reward 分量非零且映射到正确 action；
- learned callback query、空 query、重复 query 和格式可观测；
- 日志、目录和 checkpoint 不互相覆盖。

Pilot checkpoint 不进入正式训练曲线，也不作为 40-step 起点。

### 9.6 正式 0→40

1. B40、C40 分别从零开始；
2. 两边使用相同 manifest/order/seed 和同一 base revision；
3. 可顺序运行，不能共享或覆盖目录；
4. step 40 完成 resumable checkpoint、adapter export 和 merge/reload 校验；
5. 对 Base/B40/C40 运行每格 32 条外部评测；
6. 生成中期曲线、格式/callback 统计和费用实测。

### 9.7 门控后同步 40→80

只有同时满足以下条件才继续：

- B40/C40 checkpoint 均可恢复；
- 训练没有 reward/format collapse；
- callback/state-reward 监控说明实现确实生效；
- 双条件剩余训练 + 最终评测的保守成本不超过硬停止线；
- 继续 80 的判断不依赖“只延长当前领先的一边”。

随后从明确的 `global_step_40` 分别恢复 B80/C80。两边同步延长；不允许只训练表现更好的条件。完成后以完整 64 条/格评测作为最终主结果。

---

## 10. 评测设计与效果证据链

### 10.1 四个主格

每次主评测都完整报告：

| 数据集 | 200 docs | 800 docs |
|---|---:|---:|
| HotpotQA（ID） | 必做 | 必做 |
| 2WikiMultiHopQA（OOD） | 必做 | 必做 |

40-step 阶段每格 32 个固定 QA；80-step 阶段每格 64 个固定 QA。所谓“每格”是一个数据集 x 一个文档规模。

### 10.2 模型/方法矩阵

主结果：

1. Base Qwen3.5-4B；
2. B40/B80：learned callback，`algorithm.alpha=1.0`；
3. C40/C80：learned callback，`algorithm.alpha=0.8`。

最终 callback 消融使用共同最终 endpoint 的同一个 `C_final` adapter：优先 C80；若预算门控停在
40 steps，则使用 C40，并把每格样本量明确标为 32 而不是 64。

4. C_final + learned；
5. C_final + none；
6. C_final + fixed_question。

`none` 必须真正禁止检索并不把 recalled state 注入下一步；`fixed_question` 必须在每个允许 callback 的步骤用原问题作为 query；`learned` 使用模型产生的 query。不能只通过文件名包含 `nocallback` 来推断行为。

### 10.3 解码和运行规则

主评测使用确定性 greedy decode：

- `do_sample=false`；
- temperature/top-p 不参与采样；
- 开启可用的 deterministic settings，并以相同 model/mode/manifest 重跑 token hash 作为验证目标；
- runner 并发从 1 开始，按显存/稳定性升到 2-4；
- 每条记录保存 prompt/template revision、完整 trajectory、query、retrieved state IDs、最终原始输出和解析答案；
- eval `max_chunks` 按样本动态计算，不继承训练值 6；
- 逐样本断言 `processed_doc_count == manifest_doc_count`，并记录 context tokens、实际 chunks 和截断标志；
- 每次运行使用新目录和完成标记；
- 单条失败记为失败样本，不能静默跳过；
- 模型加载、单请求和全任务都设硬超时，子进程提前退出立即失败。

GDN/fused CUDA kernel 仍可能造成极小非确定性。如果 smoke 重跑 token 不完全一致，必须记录环境、logit
tolerance 和差异率，固定一次 canonical output 供 paired scoring；不能声称 bitwise deterministic，也不能在
B/C 间使用不同的并发或 kernel 设置。

若需要温度采样，只能作为明确标注的次要实验，不能和 greedy 主表混合。

### 10.4 答案指标

每个预测对 `answers[]` 中**所有合法答案**逐一评分并取最大，不能只用 `answers[0]`。

必须报告：

- normalized Exact Match；
- token F1；
- substring EM；
- boxed answer 提取成功率；
- 无 boxed answer 时的 fallback extraction 成功率，并与 strict boxed 指标分开。

最终表同时给出四格、四格 macro average 和 200→800 退化量。主要对照 delta：

- C - B；
- C learned - C none；
- C learned - C fixed_question；
- B/C - Base。

### 10.5 格式和 callback 行为指标

至少记录：

- 任务输出中的 `<thinking>`、`<update>`、`<recall>` 和 boxed 标签合法率；
- 每条 trajectory 的 callback 触发次数和触发率；
- valid、empty、malformed、duplicate query 比例；
- callback 指向的历史 step 和平均回看距离；
- 检索为空、检索重复同一 state 的比例；
- memory/final 输出达到长度上限的截断率；
- recalled text 对 gold answer/supporting fact 的 lexical hit；
- 通过 doc/chunk provenance 计算的 supporting-document hit proxy。

Lexical hit 和 provenance proxy 只能称为“检索命中代理指标”，不能冒充严格因果归因。

### 10.6 统计规则

本预算默认只有一个训练 seed，因此：

- 报告这是 single-seed 缩小实验；
- 四格 delta 使用相同 QA 的 paired comparison；
- 对 EM/F1 及主要 delta 做 10,000 次 paired bootstrap，报告 95% CI；
- 不因 CI 不显著而隐藏结果，也不把单 seed 小样本说成普遍结论；
- 40-step 的 32 条是中期门控，不作为最终论文式结论；
- 80-step 的 64 条是本项目正式主表。

### 10.7 Distant-evidence 案例

在看模型输出前，从 manifest 固定 6-10 个 supporting facts 相距较远的 QA ID。最终为每例并排展示：

- chunk/supporting-doc 时间线；
- learned query；
- retrieved memory step；
- 是否命中支持证据；
- none/fixed_question 的差异；
- 最终答案。

成功和失败案例都保留。案例只能解释聚合指标，不能替代聚合指标。

### 10.8 应交付的结果图表

- B/C reward、KL、adapter grad norm、format rate 训练曲线；
- Base/B/C 四格主表；
- learned/none/fixed_question 消融表；
- 200→800 性能退化图；
- callback 触发、有效 query、support-hit proxy 图；
- distant-evidence 轨迹图；
- 峰值显存、step time、GPU 小时和人民币费用表。

---

## 11. 预算与计费门控

### 11.1 默认预算

在用户没有给出更精确金额前，使用以下计划值：

- 总预算：500 元；
- GPU 消费硬停止线：450 元；
- 至少预留 50 元给持久盘、排障和最终评测。

这是停止规则，不是对 80-step 一定能在 500 元内完成的承诺。AutoDL 实际卡价、step time 和评测时长必须实测。

### 11.2 投影公式

记：

- `P`：PRO 6000 实际元/小时；
- `S`：G2b 完整正式 step 的保守秒数，取两步较大值或中位数再加余量；
- `N_remaining`：从当前时点到目标还剩的 B+C 总 step 数；
- `E_remaining`：通过每格少量样本实测得到的剩余评测 GPU 小时；
- `C_done`：gates/pilots 已消耗费用；
- `C_disk`：预计磁盘费用。

投影：

~~~text
C_target = C_done
         + P * (N_remaining * S / 3600 + E_remaining)
         + C_disk
~~~

正式训练前目标 40 时 `N_remaining=80`，目标 80 时 `N_remaining=160`；B/C40 已完成后
续到 80 时 `N_remaining=80`，不能再次把已完成的 80 个 B+C steps 重复计费。对 rollout 抖动、
checkpoint 和失败重启再加至少 20% contingency。GPU 部分比较 450 元停止线，总费用比较 500 元上限。

每个 GPU 任务写一条 ledger：

~~~text
run_id, gpu_type, hourly_price, start/end, billed_hours,
steps, median_step_s, peak_vram_gb, output_dir, status
~~~

### 11.3 降级顺序

若预算或吞吐不满足，按以下顺序降级，不能临场任意改一边：

1. 保留 4B、B/C、group 8 和完整四格，只完成 40 steps，不续 80；
2. 缩减次要案例/额外采样，但保留主表和 callback 消融；
3. 在正式训练开始前，将 B/C **共同**降为 group 4、train batch 2，并重新做 G2；不得让 B/C 配置不同；
4. 仍无法完成时才把正式模型降为 2B，并明确项目变为次级方案；
5. 不使用 QLoRA、双 5090 或未验证的新 rollout engine 作为训练中途的临时救火。

如果 B/C40 已完成，优先把完整 40-step 结果做扎实，而不是为追求“80”牺牲评测、消融或 checkpoint 可靠性。

---

## 12. 风险登记与降级触发

| 风险 | 早期信号 | 处理 |
|---|---|---|
| Qwen3.5 text loader 错误 | conditional/text logits 不一致、未知 keys | 停在 G-1/G0，修正 mapping 和 whitelist |
| GDN/FLA Blackwell 不稳定 | backward hang、NaN、illegal access | 固定已修复 commit；必要时更换已验证 build，不进长跑 |
| PEFT + FSDP 混合参数失败 | flatten/dtype/requires-grad 异常 | 保持 root-only FSDP + `use_orig_params=True`，单元测试 |
| HF rollout 重复 group | batch 突然 x8、OOM | trainer 展开后 call-level `n=1`，shape assertion |
| Adapter 未用于生成 | actor 输出始终等于 base | adapter enable/disable logits test，检查 FSDP summon context |
| 96GB 仍 OOM | 峰值 >85GB、allocator retry | 先检查重复 n/长度；再降低 token batch；最后才启用 ref offload |
| CPU offload 极慢 | PCIe 利用高、step time 激增 | 正式默认全关闭；只有容量失败才逐项启用并重做 G2 |
| Reward 全同/advantage 为零 | group 内方差为 0 | 检查解析、gold list、sampling/group；不得盲目长跑 |
| Reward 与论文错位 | all-gold average、multiline 解析失败 | 统一 parser + 式 (5)/(6) fixtures；G-1 不通过不开训 |
| History/provenance 丢失 | set 去重、无 step/doc IDs | 有序 MemoryRecord + word-recall + deterministic tie-break |
| 格式崩溃 | boxed/update/recall 合法率大面积为 0 | 检查 template、Base 对照和 format reward 映射 |
| Callback 形同虚设 | query 空/重复、retrieval 无注入 | mode integration tests + trajectory dump |
| Checkpoint 无法恢复 | missing adapter keys/global step | G1/G2 新进程 resume 硬门槛 |
| 磁盘爆满 | optimizer shards 累积 | 300GB 盘、保存后校验、再安全 prune |
| Ray 填满系统盘 | `/tmp/ray` 增长 | `RAY_TMPDIR` 指向持久盘并监控 quota |
| 评测无限等待 | 模型加载失败仍轮询 | 硬超时、检测子进程、失败返回非零 |
| 200/800 数据不可配对 | IDs/支持证据不一致 | 重建嵌套 manifest；schema/hash gate 失败即停止 |
| 800-doc 只处理前 30K | eval 误用训练 `max_chunks=6` | 动态完整迭代；doc/token/chunk consumption assertion |
| C 未超过 B | delta 小或 CI 跨 0 | 完整报告行为/消融/退化；不选择性延长或改口径 |
| MFU 报告错误 | flops counter 不识别 qwen3_5/PRO | 补支持前不报告 MFU，只报 wall-time/tokens/s |

Offload 回退顺序：先 reference param offload，再评估 actor optimizer offload；actor param offload 最后。任何回退都会改变吞吐，必须重新做完整 G2b 和预算投影。

---

## 13. 必改代码范围和预期交付

### 13.1 核心代码

| 路径/模块 | 必要改动 |
|---|---|
| `verl/workers/fsdp_workers.py` | Qwen3.5 text-only loader、actor-only PEFT、root-only FSDP、`use_orig_params=True`、trainable-only optimizer、PEFT 前统一 seed、移除 ref CPU-offload hardcode、可配 attention |
| `verl/workers/actor/dp_actor.py` | FlashAttention padding helper lazy import/fallback；SDPA + no-remove-padding 路径可独立启动 |
| `verl/workers/rollout/hf_rollout.py` | `pad_to/max_tokens/n` contract、ceil microbatch、adapter generate、shape/mask assertions |
| `recurrent/impls/memory_revisit.py` + retriever | 统一 parser、有序 MemoryRecord、论文 word-recall、callback modes、`.flatten()`、provenance |
| `verl/trainer/ppo/metric_utils.py` + 统一 parser | 式 (5)/(6) 的 all-gold max、参数方向、multiline/format/callback reward；训练/eval 同 parser |
| `recurrent/generation_manager.py` 及 trainer | group 只展开一次、action/reward shape 和 logging |
| `verl/utils/checkpoint/fsdp_checkpoint_manager.py` + trainer orchestration | LoRA-aware state、scheduler/RNG；由 trainer 继续协调独立的 dataloader/extra state 和 adapter export |
| `scripts/model_merger.py` 或新 exporter | PEFT adapter reload、merge-and-unload、text-only save |
| `verl/trainer/main_ppo.py` | `RAY_TMPDIR`，去除硬编码 `/tmp/ray` |
| `taskutils/memory_eval/` 或新 runner | 单卡 Transformers recurrent eval、三 modes、硬超时、全 gold max、全新输出目录 |
| `taskutils/data_synthesis/` / manifest builder | 结构化 doc IDs、嵌套 200/800、supporting-fact 与 chunk provenance、schema/hash 校验 |
| `verl/utils/flops_counter.py` | 可选补 Qwen3.5/PRO；否则报告中禁用 MFU |

具体实现可以拆分文件，但行为契约不能省略。

### 13.2 配置和环境

预期新增版本化文件：

~~~text
configs/reproduction/g0_qwen35_08b.yaml
configs/reproduction/g1_step1_qwen35_2b_lora.yaml
configs/reproduction/g1_resume2_qwen35_2b_lora.yaml
configs/reproduction/g2a_qwen35_4b_lora.yaml
configs/reproduction/g2b_step1_qwen35_4b_lora.yaml
configs/reproduction/g2b_resume2_qwen35_4b_lora.yaml
configs/reproduction/b_pilot_qwen35_4b_lora.yaml
configs/reproduction/c_pilot_qwen35_4b_lora.yaml
configs/reproduction/b40_qwen35_4b_lora.yaml
configs/reproduction/c40_qwen35_4b_lora.yaml
configs/reproduction/b80_qwen35_4b_lora.yaml
configs/reproduction/c80_qwen35_4b_lora.yaml
configs/reproduction/eval_qwen35_4b.yaml
requirements/ 或 environment/ 下的明确 lock
~~~

文件名可按仓库结构微调，但必须能从文件名区分 gate、模型、条件和阶段。

### 13.3 测试

至少覆盖：

- text-only/conditional logits；
- template snapshots 和 `enable_thinking=False`；
- LoRA target allow/deny list；
- optimizer 无 frozen params；
- base checksum / adapter update；
- B/C step-0 adapter hash、batch IDs、首轮 sampled-token hash 一致；
- HF rollout 的 1 个 active sample、不可整除 microbatch、`n=1/8`、不同 max/pad；
- learned/none/fixed_question 的确定性行为；
- 论文式 word-recall retrieval、ordered duplicates、tie-break 和 provenance；
- 式 (5)/(6) 的 max-vs-average、参数方向、多行/空/重复标签 fixtures；
- outcome/state/format/callback reward 及 B/C advantage 组合；
- multi-gold max evaluator；
- checkpoint fresh-process resume；
- checkpoint 保存失败故障注入不删除最近有效恢复点；
- adapter reload/merge logits；
- formal dataset 的 `truncation=center`；
- 训练/eval manifest schema、200⊂800 嵌套、doc/chunk/supporting-fact 反查；
- eval 动态 chunks 完整消费 200/800，任何截断均失败；
- 无可用 FlashAttention 时 SDPA/no-remove-padding worker import；
- 所有 Hydra config resolve。

---

## 14. 最终验收清单

### 14.1 工程验收

- [ ] G-1 所有 CPU tests 和配置 resolve 通过；
- [ ] 统一 parser、论文奖励、word-recall 检索和嵌套 manifest 的 fixtures 通过；
- [ ] G0 0.8B 连续 20 optimizer loops 无 hang/NaN；
- [ ] G1 2B 完成 step 1 → 新进程 resume step 2 → export/merge/eval；
- [ ] G2a/G2b 4B 正式长度通过，G2b 完成新进程 resume；
- [ ] 4B adapter export/reload 与 merged BF16 model reload 的 logits/生成对齐；
- [ ] 只有 adapter 更新，base checksum 不变；
- [ ] HF rollout 没有 group 二次扩张；
- [ ] B/C 目录、日志、checkpoint 完全隔离；
- [ ] B/C40 和 B/C80 的 resume 路径明确、可加载；
- [ ] 峰值显存、分项耗时和费用 ledger 完整。

### 14.2 科学验收

- [ ] Base/B/C 使用相同 base revision、manifest、seed 和 eval decode；
- [ ] B/C 唯一算法差异是 `algorithm.alpha`；
- [ ] 四个主格无缺失、不挑格；
- [ ] 每个合法 gold answer 都参与评分；
- [ ] learned/none/fixed_question 行为确实不同；
- [ ] EM/F1/sub-EM、格式、callback、support proxy 均有结果；
- [ ] 200→800 退化和 paired delta 已报告；
- [ ] 成功与失败 distant-evidence 案例均保留；
- [ ] single-seed/LoRA/Qwen3.5/缩步数限制写入报告。

### 14.3 “有面试效果”的判定

理想结果是 C 在四格 macro F1/EM、800-doc 鲁棒性或 callback support proxy 上优于 B，并且 learned 优于 none/fixed。最低可展示结果不要求每个数字都正向，但至少应同时具备：

1. Base→RL 后协议/格式和可解析率明显改善；
2. learned callback 产生非平凡、可解释的回看轨迹；
3. 至少一个预先定义的聚合机制指标或 paired delta 呈正向；
4. 完整解释其余负结果、方差和资源约束；
5. 工程闭环和成本数据可复查。

如果这些条件也未满足，不能靠挑案例宣布成功；应将项目表述为“完成工程适配与负结果复现”，并保留后续增加步数/seed 的路线。

---

## 15. 简历、面试和项目产出

### 15.1 必备材料

最终归档：

- 一页项目摘要；
- 方法/工程架构图；
- 环境 lock 和完整配置；
- 数据 manifests/hashes；
- 训练和评测命令；
- Base/B/C 主表与 callback 消融表；
- 训练曲线、长上下文退化图、案例图；
- adapter、merged checkpoint metadata 和恢复说明；
- 显存/耗时/费用表；
- known limitations 和失败记录。

### 15.2 简历表述模板

数字必须在实验完成后替换，不能预填或捏造：

> 在单张 RTX PRO 6000 96GB 上将 ReMemR1 适配至 Qwen3.5-4B，设计并实现 LoRA-GRPO 的 PEFT/FSDP、recurrent HF rollout、adapter checkpoint/merge 与长上下文评测闭环；在 HotpotQA/2WikiMultiHopQA 的 200/800-document 设置上完成 outcome-only vs multi-level reward 及 learned/none/fixed callback 对照，以 [实测指标] 验证 [实际观察到的趋势]，总成本 [实测金额]。

### 15.3 面试叙事主线

1. 为什么完整论文规模不可行，以及如何保留核心变量；
2. 为什么 4B、为什么 LoRA、为什么不做 QLoRA；
3. 为什么 96GB 不是“模型权重需要”，而是长上下文 RL 的容量/稳定性预算；
4. Qwen3.5 的 GDN、统一 checkpoint 和 thinking 带来了哪些适配；
5. 如何保证 B/C 只差 reward alpha；
6. 如何证明 adapter 真在训练、base 没变化、checkpoint 真可恢复；
7. 如何用 callback 消融和 distant evidence 支撑机制解释；
8. 哪些结果没有复现、为什么不能过度声称。

---

## 16. 新实现会话启动指令

新会话先执行只读检查，然后从 G-1 开始实现。可直接使用：

~~~text
当前工作区是 D:\codespace\python\ReMemR1。
请先阅读 docs/final_reproduction_plan_zh.md；它是唯一权威方案。
论文是 2509.23040v5.pdf。

目标是“基于 Qwen3.5-4B LoRA-GRPO 的 ReMemR1 缩小机制复现”：
- 正式模型 Qwen/Qwen3.5-4B；
- 单张 RTX PRO 6000 96GB；
- LoRA r32/alpha64/dropout0、text all-linear；
- B: algorithm.alpha=1.0，C: 0.8；
- batch/mini=4/4，group=8，5000x6，memory/final=768/512；
- B/C 先各 40 steps，评测后同步 resume 到 80。

请先检查 git status，保留现有文档变更，不要回退用户改动。
本轮从文档第 9.1 节 G-1 开始编码和测试：
Qwen3.5 text-only loader、PPO PEFT/FSDP、HF rollout contract、
callback modes、checkpoint/export/merge、Transformers eval runner、
完整 configs 和 CPU tests。

在 G-1 全部通过前不要启动长训练；不要自动 commit/push，
除非我明确要求。遇到实现与方案冲突时，给出代码证据并同步更新方案和测试。
~~~

---

## 17. 文档优先级与变更规则

发生冲突时按以下顺序处理：

1. 本文档中的最终项目决策；
2. 论文方法和明确超参数；
3. 当前仓库的实际代码行为；
4. 历史方案/旧交接文档；
5. README 中未经锁定的旧环境命令。

本文档不是替代测试的“愿望清单”。实现发现新阻塞时，应先用最小复现和代码位置证明，再同步修改本文档、配置和测试；不得只改运行命令形成不可追踪的口头例外。

最终对外项目名称固定为：

> **基于 Qwen3.5-4B LoRA-GRPO 的 ReMemR1 缩小机制复现**
