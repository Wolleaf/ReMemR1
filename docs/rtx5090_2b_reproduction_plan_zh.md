# ReMemR1 单 RTX 5090 + Qwen3.5-2B 候选实施方案

> 状态：候选方案，尚未替代 `final_reproduction_plan_zh.md`。
> 制定日期：2026-07-17。
> 代码基线：`cc6c330ce1e8a1f922c28e84dc3f28b298914636`。
> 目标硬件：1 x NVIDIA GeForce RTX 5090 32GB。
> 正式模型：`Qwen/Qwen3.5-2B`，revision `15852e8c16360a2fea060d615a32b45270f8a8fc`。

本文给出从当前“RTX PRO 6000 96GB + Qwen3.5-4B”方案迁移到“单 RTX 5090 32GB +
Qwen3.5-2B”的完整候选设计。它用于评审方案、估算成本和指导后续实现，不授权直接启动 GPU
长训练。只有用户确认后，才应同步修改权威交接文档、配置矩阵、云端硬件门禁和正式运行入口。

---

## 0. 执行摘要

### 0.1 最终建议

5090 路线不应尝试把现有 4B 正式配置原样塞进 32GB 显存，而应把正式模型降为 2B，保留
ReMemR1 的核心机制与对照：

- 保留 Qwen3.5 text-only、LoRA-GRPO、recurrent memory、learned callback；
- 保留 B（outcome-only）与 C（outcome + state reward）成对训练；
- 保留 learned / none / fixed_question callback 消融；
- 保留 HotpotQA / 2WikiMultiHopQA、200 / 800 documents 四格评测；
- 保留 checkpoint/resume、adapter export/merge、数据和评测 provenance；
- 正式训练从 B/C40 开始，B/C80 仅在显存、吞吐、费用和恢复门禁通过后同步续训；
- 不引入 QLoRA，不把 actor master 或 Adam 状态静默降为 BF16，不使用双 5090。

### 0.2 相对当前 4B 方案的固定变化

| 项目 | 当前 4B / PRO 6000 | 候选 2B / 5090 |
|---|---:|---:|
| 正式模型 | Qwen3.5-4B | Qwen3.5-2B |
| GPU | PRO 6000 96GB | GeForce RTX 5090 32GB |
| train / PPO mini batch | 4 / 4 | 2 / 2 |
| GRPO group | 8 | 4 |
| 每个 optimizer step 的 trajectories | 32 | 8 |
| actor / compute / reference | FP32 / BF16 / BF16 | 不变 |
| 长上下文 | `5000 x 6` | 不变 |
| memory / final 最大生成 | 768 / 512 | 不变 |
| B/C 训练 | 40，按门控续到 80 | 不变 |
| offload | 全关 | G2 在全关与仅 ref offload 中预先选定 |
| 主评测 | 2 数据集 x 200/800 docs | 不变 |

这一路线换取的是更高的交付概率和更低的租卡门槛，代价是模型能力、单步采样量和机制信号强度
低于 4B 方案。最终报告必须明确这是“单卡资源受限的 2B 缩小机制复现”，不能沿用 4B 结果口径。

### 0.3 当前实现边界

当前云端一键流水线只执行 CPU preparation、G0 和 G1，不会启动 G2、B/C pilot 或正式长训练。
因此现在切换方案不会推翻已经实现的长跑代码；正式 2B 容量门禁和 B/C 入口本来就需要新增。

当前代码会因 GPU 名称和 `>=90 GiB` 门禁直接拒绝 5090。在完成 5090 profile 适配并提交前，
不得在 5090 上执行现有 `start_gpu_gates.sh`。

### 0.4 预期后果

| 维度 | 相对 4B / PRO 6000 方案的预期变化 |
|---|---|
| 工程跑通 | 远高于“5090 硬跑 4B”，但正式长度是否稳定仍由 G2 决定 |
| 正向科学信号 | 2B 能力和 group 4 采样更弱，C-B 差异可能更小或 CI 跨 0 |
| 训练时间 | 卡本身便宜，但长生成和 R1 offload 会拉长时间；若纯 step 实测为 15/30/60 分钟，则 86 个 L1 step 的纯计算分别为 21.5/43/86 小时，另加启动、保存、产物和评测 |
| 评测时间 | 800-doc 完整消费可能成为主要耗时，必须另做样本级 probe |
| 简历价值 | 完成 L1 后足够作为单卡资源受限机制复现；只完成 G0/G1 则只能称工程框架 |
| 结果口径 | 强调可审计闭环、控制变量和负结果，不追求与论文或 4B 的绝对数值可比 |

这里不预填“科学成功率”百分比。工程成功由容量和恢复门禁定义；C 是否优于 B 在真实成对实验前无法
可信预测。方案成功也不要求 C 必须为正，只要求按预注册矩阵完整报告。

---

## 1. 项目定位与可声明范围

### 1.1 研究问题

候选方案只回答以下三个缩小问题：

1. 在 Qwen3.5-2B 上，LoRA-GRPO 能否跑通 ReMemR1 recurrent memory 与 learned callback 的
   完整训练、恢复和评测闭环？
2. 在完全相同的 base、数据、seed、采样预算和训练长度下，C 的多级 state reward 相对 B 的
   outcome-only reward 是否产生可观察的准确率、格式或 callback 行为差异？
3. 同一个 C endpoint 使用 learned / none / fixed_question callback 时，长距离证据召回和最终答案
   是否呈现一致的机制趋势？

### 1.2 可以声明

完成最低交付后可以说：

> 在单张 RTX 5090 32GB 上，将 ReMemR1 适配到 Qwen3.5-2B，并用 LoRA-GRPO 完成
> 200-document 长上下文训练闭环；通过 B/C 奖励对照和 learned/none/fixed callback 消融，
> 评估 200/800-document 条件下的机制趋势、资源成本与失败边界。

### 1.3 不可以声明

- 完整复现论文的 Qwen2.5-3B/7B 全参数训练或绝对指标；
- 完成当前 4B / PRO 6000 方案；
- 只凭一个 seed、单个案例或单个评测格证明方法普遍有效；
- 把 2B、LoRA、group 4、batch 2 和 offload 带来的差异全部归因于 ReMemR1；
- C 未显著超过 B 时筛选 checkpoint、只延长领先条件或重跑 seed 直到得到有利结论。

### 1.4 交付层级

| 层级 | 必须完成 | 项目含义 |
|---|---|---|
| L0：门禁闭环 | G-1、G0、G1、G2 全部通过 | 证明环境、2B 训练和正式长度容量可用 |
| L1：最低正式交付 | B/C pilot、B40/C40、32 QA/格主评测、callback 消融 | 已可形成完整简历项目和技术报告 |
| L2：目标交付 | B80/C80、64 QA/格主评测、完整资源与恢复证据 | 候选方案的最佳正式结果 |
| L3：正向主证据 | 预注册 `C-B` 四格 macro answer EM 为正且 paired 95% CI 不跨 0 | 加分项，不是工程成功的必要条件 |

负结果只要数据、控制变量、恢复链、评测和统计完整，仍属于有效 L1/L2 交付。
Learned-control 是预注册 secondary mechanism evidence，必须单独报告，但不能替代 L3 primary。

---

## 2. 硬件、系统与软件锁

### 2.1 GPU 硬门禁

正式 profile 固定为：

| 字段 | 要求 |
|---|---|
| GPU count | 恰好 1 |
| GPU name | NVIDIA GeForce RTX 5090；允许厂商前缀差异，不允许其它型号冒充 |
| VRAM | 标称 32GB，运行时可见容量至少 31 GiB |
| Compute capability | `(12, 0)` / `sm_120` |
| CUDA runtime/toolkit | 13.0 / 13.0 |
| BF16 | matmul、FLA/GDN forward/backward 和 optimizer loop 全部通过 |
| Kernel evidence | 绑定 GPU name、UUID、driver、CUDA、kernel commit 和 build log |

RTX 5090 与 PRO 6000 在本项目所需 kernel 路径上同为 `sm_120`，因此 cu130 lock、固定 FLA 与
causal-conv1d revision、BF16 probe 和 `TORCH_CUDA_ARCH_LIST=12.0` 可以复用。若实际镜像只有
CUDA 12.8，必须停止并重封环境 lock，不能把 12.8 当作 13.0 等价运行。

GPU preflight 还要求恰好一张可见 GPU、启动时无其它 compute process，且空闲显存至少 29 GiB。
不满足时应落盘失败并关机，不能与未知进程争抢显存后把 OOM 归因于配置。

### 2.2 实例资源

| 资源 | 最低 | 推荐 |
|---|---:|---:|
| CPU | 24 cores | 32 cores 或以上 |
| CPU preparation RAM | 48 GiB | 64-96 GiB |
| GPU host RAM（R0） | 96 GiB | 128 GiB |
| GPU host RAM（R1） | 128 GiB | 160-192 GiB |
| 持久盘可用空间 | 200 GiB | 250 GiB |
| 系统 | Ubuntu 22.04 | 同左 |
| Python | 3.12.2 持久隔离环境 | 同左 |

offload、Ray 临时目录、checkpoint、模型 cache 和评测结果必须全部落在持久盘。禁止把显存压力转化为
系统盘 swap 或 `/tmp/ray` 爆盘。R0 不应仅因实例 RAM 低于 128 GiB 被预先排除；R1 则必须满足更高
RAM 门禁，并在 G2 中证明峰值低于可用内存 80%、无 swap 和无持续 page-fault 抖动。

### 2.3 自动关机与计费边界

沿用现有云端状态机：

- 每个 CPU/GPU phase 使用全局持久锁、`nohup + setsid`、独立 launcher 和原始退出码；
- 成功、non-retryable scientific-stop 和终态失败均先同步日志、匹配的 terminal marker、checkpoint/evidence，再请求 guest shutdown；
- 锁冲突、状态无法落盘、sync 失败、授权复验失败、`--keep-running` 和 `--dry-run` 不关机；
- guest shutdown 后仍必须在 AutoDL 控制台确认停止计费；
- G2、B/C40、B/C80 必须是三个独立付费决策，不允许一条命令无门控地跑完全部长训练。

---

## 3. 2B 显存模型与容量策略

### 3.1 规划基线

固定 2B revision 的 text-only language model 约 1.882B 参数。按当前实现估算：

| 组成 | 规划占用 |
|---|---:|
| FP32 actor base | 约 7.01 GiB |
| BF16 reference | 约 3.51 GiB |
| 按当前 target manifest 估算的 all-linear LoRA r32 约 33.64M 参数及 FP32 grad/Adam | 约 0.50 GiB |
| FSDP BF16 materialization/unshard buffer | 保守约 3.5 GiB |
| 模型与训练状态规划基线 | 约 14.5-16 GiB |

剩余显存还要容纳激活、KV/cache、old/reference log-prob、CUDA allocator 和临时张量。主要峰值风险是
大词表 logits：词表约 248,320，若单个 BF16 logits 张量达到 `12288 x 248320`，理论大小约
5.68 GiB；若该路径实际 materialize FP32，同形状约 11.37 GiB。当前 actor 在裁剪 response 前可能
先产生整段 logits，因此不能凭静态参数估算跳过 G2，也不能把 14.5-16 GiB 静态基线当成运行峰值。

HF rollout 复用 actor FSDP module，不额外常驻一份 vLLM 模型，这是单 5090 路线能够成立的重要前提。

### 3.2 不改变的数值口径

- actor master parameter 保持 FP32；
- forward/backward mixed precision 使用 BF16；
- reference 使用 BF16；
- LoRA rank/alpha/dropout/bias 固定为 32/64/0/none；
- text LM all-linear target manifest 与排除项断言不变；
- gradient checkpointing 开启；
- `use_orig_params=True`、root-only FSDP、sequence parallel size 1；
- 不使用 QLoRA、BF16 Adam、双卡 FSDP 或未经验证的新 rollout engine。

### 3.3 预注册 offload profiles

容量选择只能在正式 B/C 开始前根据工程指标完成，不能根据 reward 或评测结果挑 profile。

| Profile | actor param | actor optimizer | ref param | 地位 |
|---|---:|---:|---:|---|
| R0 | false | false | false | 首选，吞吐最好 |
| R1 | false | false | true | R0 仅容量指标黄区/OOM 时的唯一正式 fallback |
| R2 | false | true | true | 仅诊断；LoRA Adam 很小，收益有限 |
| R3 | true | true | true | 不自动采用；触发方案重新决策 |

选择规则：

1. 先以 R0 跑完整 G2a/G2b/length-stress；
2. 只有确认属于显存/headroom/allocator 的容量失败，才允许从 R0 改 R1，并从 G2a 重新开始完整链；
3. R1 若通过且保守费用投影在预算内，将 R1 固定到 B/C 全部配置；
4. R1 仍失败时停止。不得临时降 actor dtype、缩一边的 group 或直接进入 R3；
5. 最终选中的 profile 写入自哈希 `capacity-profile.json`，所有 B/C segments 均验证该文件与各自
   sealed resolved config 一致；评测验证该 profile hash 和源 checkpoint identity。

R0 的容量失败或黄色结论必须先发布完整 terminal evidence 并结束当前 launcher；不得在同一个 launcher
里自动重试 OOM 或直接串行切到 R1。操作者检查失败原因后，使用显式 R1 参数和批准 marker 启动新的
capacity launcher。锁冲突、resume/schema/数值/科学失败均不产生 R1 批准资格。

R1 批准 marker 必须一次性、自哈希，并绑定 active profile/commit、R0 terminal 与 capacity-evidence
hash、预注册的唯一容量原因和最新预算投影；旧 pipeline 的 marker 或仅存在一个同名空文件均无效。

Checkpoint schema、optimizer/RNG、predecessor 或 resume 语义失败不会被 offload 修复。这类失败必须在
同一个 R0/R1 profile 下修复并重跑，不能用“切 R1”掩盖实现错误。

CPU 阶段必须预先 compose、resolve 并封存 R0/R1 两套正式配置；GPU G2 只能从这些 sealed config
中选择一个，不能运行时修改 `ref.param_offload`。`capacity-profile.json` 同时记录完整 active tree SHA
和所选 profile config-set SHA，而不是生成一份未经过 CPU 封存的新配置。

评测进程不加载训练期 reference model，因此评测 runtime 本身不需要继承 `ref.param_offload`；但其
verified package 必须绑定产生 checkpoint 的 `capacity-profile.json` hash、resolved training config
hash 和 checkpoint hash，不能把“推理不使用 offload”误写成训练 profile 发生变化。

---

## 4. 正式配置契约

### 4.1 公共参数

| 类别 | 参数 | 值 |
|---|---|---|
| Model | path/revision | Qwen3.5-2B / `15852e8c16360a2fea060d615a32b45270f8a8fc` |
| Model | text-only / native thinking | true / 显式 `enable_thinking=False` |
| LoRA | rank / alpha / dropout / bias | 32 / 64 / 0 / none |
| Precision | actor / compute / reference | FP32 / BF16 / BF16 |
| Data | formal train documents | 200 |
| Data | max prompt / response tensor | 30000 / 1024 |
| Recurrent | chunk / chunks | 5000 / 6 |
| Recurrent | question / memory / final max | 1024 / 768 / 512 |
| Data | train batch | 2 |
| PPO | mini / micro / epochs | 2 / 1 per GPU / 1 |
| GRPO | group / recurrent call `n` | 4 / 1 |
| Trajectories | 每 optimizer step | 8 |
| Token budget | actor/ref/rollout max per GPU | 12288，G2 实测确认 |
| Rollout | backend / generation micro | HF / 1 |
| Sampling | temperature / top-p / top-k | 1.0 / 1.0 / 0 |
| Optimizer | AdamW LR / weight decay / grad clip | `5e-6` / 0.01 / 1.0 |
| Scheduler | style / warmup | constant-with-warmup / 8 steps |
| KL | loss / coefficient / type | true / 0.001 / low_var_kl |
| PPO | clip / entropy | 0.2 / 0 |
| Runtime | dynamic batch / compile | false / false |
| Validation | trainer internal | 关闭，使用固定外部 runner |
| Seed | run | 42，派生 seed 和顺序全部封存 |

相对 4B 方案，每步 trajectories 从 32 降为 8。40/80 steps 不再代表相同算力或样本暴露量，报告中
必须同时给出 prompt groups、trajectories、生成 token、有效 advantage group 数和 wall time。

`batch=2, group=4` 是本候选方案预先固定的资源档：每步仍有 8 条 trajectory，但覆盖两个 prompt
 group，trajectory 数是旧 `batch=4, group=8` 的四分之一。它比 `batch=1, group=8` 更强调 prompt
覆盖，代价是每个 group 的 reward variance 更容易不足，因此 pilot 的非零 advantage 门禁不可省略。
如果 G2 显示 group 8 很宽裕，它也只能作为另一个预注册 profile 重新做成对实验，不能在 B/C 中途升级。
总 FLOPs 和 wall time 还受模型从 4B 降到 2B、长 prompt、固定启动开销和 offload 影响，不能由
trajectory 比例直接推断。

不含 pilots 时，每个 arm 的累计暴露为：

| Profile/endpoint | steps | prompt groups | trajectories |
|---|---:|---:|---:|
| 候选 2B / batch2 / group4 | 40 | 80 | 320 |
| 候选 2B / batch2 / group4 | 80 | 160 | 640 |
| 原 4B / batch4 / group8 | 40 | 160 | 1280 |
| 原 4B / batch4 / group8 | 80 | 320 | 2560 |

因此 2B 到 80 steps 只追平旧 4B 40 steps 的 prompt-group 数，trajectory 仍只有一半；2B L1 的
40-step 单 seed 结果应称探索性机制对照，不能称样本量匹配的 4B 复现。

### 4.2 B/C 唯一差异

| 条件 | Callback | `algorithm.alpha` | Advantage |
|---|---|---:|---|
| B | learned | 1.0 | outcome only |
| C | learned | 0.8 | 80% outcome + 20% state |

B/C 必须使用相同的 base revision、LoRA 初始化 hash、manifest、sample order、seed、batch/group、
offload profile、硬件、软件 lock、步数、解码配置和评测输入。Resolved config 的差异只允许：

- `algorithm.alpha` 及由它决定的预期 advantage 组合；
- B/C 独立的 experiment name、checkpoint/log/adapter 输出路径；
- step-zero fingerprint 路径及 C 对 B fingerprint 的显式引用。

其它科学或运行参数出现差异一律 fail closed。

### 4.3 分段训练

| 任务 | 起点 | total steps | save freq | resume |
|---|---|---:|---:|---|
| B pilot | base + seed42 LoRA init | 3 | 3 | disable |
| C pilot | base + 同一 LoRA init hash | 3 | 3 | disable |
| B20 | base + seed42 LoRA init | 20 | 20 | disable |
| C20 | base + 同一 LoRA init hash | 20 | 20 | disable |
| B40 | 已验证的 B `global_step_20` | 40 | 20 | resume_path |
| C40 | 已验证的 C `global_step_20` | 40 | 20 | resume_path |
| B60 | 已验证的 B `global_step_40` | 60 | 20 | resume_path |
| C60 | 已验证的 C `global_step_40` | 60 | 20 | resume_path |
| B80 | 已验证的 B `global_step_60` | 80 | 20 | resume_path |
| C80 | 已验证的 C `global_step_60` | 80 | 20 | resume_path |

Pilot 不进入正式曲线，也不能作为 B20/C20 起点。正式训练有意按 20-step immutable stages 切分，
使每段都从已验证 checkpoint 显式恢复，避免长进程中断后只能从零重跑。B40/C40、B60/C60 和
B80/C80 都保留 `lr_warmup_steps=8` 并恢复原 optimizer/scheduler/RNG，不能把新 segment 当作新 warmup。
同层 B/C 可顺序运行但不能共享目录；任何续段都必须同步决定，不允许只延长领先方。

Endpoint resume probe 由 CPU handoff 封存身份的独立 full-state verifier 执行：加载 checkpoint、
optimizer、scheduler、RNG、global step 和 predecessor 后立即退出，不做 optimizer update。它读取对应
sealed config 和 checkpoint extra state，不能把 B60/B80 训练 config 通过临时 override 改成 probe。

### 4.4 Active experiment profile 与配置封存

候选 2B handoff 只允许绑定一个 active experiment profile：
`rtx5090-32g-qwen35-2b-v1`。该 profile 的 assets、data、cache、resolved configs、pipeline state 和
outputs 必须位于独立 namespace；旧 4B profile 只能保留在 Git 历史或独立 inactive namespace，不能
与 2B 配置平铺进同一个 active handoff。

CPU 阶段必须生成并 exact-key 校验以下 active resolved inventory：

| 类别 | 逻辑任务 | 封存份数 |
|---|---|---:|
| G0/G1 | G0、G1 step1、G1 resume2 | 3 |
| 训练/容量 | G2a、G2b step1/resume5、G2 length stress、B/C pilot、B/C20/40/60/80 | 14 x R0/R1 = 28 |
| 正式评测 | eval40、eval80 | 2 |
| 合计 | 单一 2B active inventory | 33 |

十四个训练/容量逻辑任务可以由公共 source config 加 R0/R1 overlays compose，避免维护 28 份机械复制的
源 YAML；但 CPU 输出必须是 28 份完整、不可变、可单独哈希的 resolved YAML。`index.json`、handoff
和 artifact verifier 都必须要求这 33 个 config IDs 恰好出现一次，缺失、多余或混入 4B config
一律失败。

G2 只能从 active index 中选择已经封存的 R0 或 R1 config 集。`capacity-profile.json` 至少绑定
experiment profile ID、CPU handoff hash、active config tree hash、所选 profile、十四个所选训练/容量
config ID/逐文件 SHA/selected-set SHA、G2/length-stress attempts 与 predecessor、GPU UUID/VRAM、
G2 evidence hash 和自身 hash。R0/R1 的 attempt、checkpoint、log 和 output roots 必须互不重叠，
失败的 R0 证据不得由
R1 覆盖或 resume。B/C phases 不接受 CLI/Hydra 对 offload、batch、group、model 或数据身份的临时覆盖。

由于具体 predecessor attempt 和输出目录在 CPU 封存时尚不存在，唯一允许的运行时注入是 pipeline
内部的 path-binding allowlist：已验证的 `resume_from_path`，以及本次 attempt 的 checkpoint/output/
log/adapter/eval-cell paths。Pipeline 必须验证 predecessor logical ID、digest、step、profile 和 path
containment 后生成 runtime-bound config 及其 SHA；用户 CLI 不能直接提供这些 override。除这组路径和
attempt identity 外，runtime-bound config 与 sealed config 的 leaf diff 必须为空。

每次 retry 的 checkpoint、adapter、merged model 和 eval cell output 也必须写入唯一 attempt-scoped
目录；严格验证并同步后只原子发布 canonical pointer。Resume/adoption 只能引用含 digest 的 immutable
attempt，不能把共享 `checkpoints/...` 目录原地覆盖，也不能把旧 failed attempt 改写成 success。

---

## 5. 分阶段 GPU 门禁

### 5.1 G0：0.8B kernel gate

沿用现有 G0：batch/mini/group `1/1/4`、`1024 x 2`、20 个连续真实 optimizer loops，offload 全关。

必须验证 sm_120、CUDA 13、BF16 GDN/FLA/full-attention forward/backward、无 NaN/hang/非法内存、
adapter 更新非零和 base checksum 不变。

### 5.2 G1：2B 接口与恢复 gate

沿用现有 2B 短配置：batch/mini/group `1/1/4`、`1024 x 2`、step 1 保存，结束进程后显式
resume 到 step 2，offload 全关。

G1 还必须完成 adapter export/reload、merged model reload、固定输入 logits/greedy token 对齐和独立
2-QA recurrent eval。G1 只证明接口闭环，不能替代正式长度 G2。

### 5.3 G2a：正式长度低风险容量 gate

| 参数 | 值 |
|---|---:|
| 模型 | Qwen3.5-2B |
| train / mini / group | 1 / 1 / 4 |
| 长上下文 | `5000 x 6` |
| memory / final | 768 / 512 |
| steps | 1 |
| reward path | C-style `algorithm.alpha=0.8`，覆盖较重的 state reward 路径 |
| offload | 先 R0；仅容量结论后由新 launcher 显式从头跑 R1 |

G2a 证明单条正式 prompt 的完整 rollout、reward、log-prob、backward 和 checkpoint 路径可运行；若
随机采样提前 EOS，它本身不能证明 768/512 生成上限或 12288 token tensor 的最坏容量。

### 5.4 G2b：目标正式 profile + 新进程恢复

| 参数 | 值 |
|---|---:|
| train / mini / group | 2 / 2 / 4 |
| micro batch | 1 |
| 长上下文 | `5000 x 6` |
| steps | step 1 -> 新进程 resume 并继续到 total step 5 |
| reward path | C-style `algorithm.alpha=0.8` |
| offload | 与本轮 G2a 相同 |

G2b 的 step 2-5 同时构成 bounded memory stress。完成后必须 export/reload adapter、merge/reload model
并验证 provenance，但此时还不能发布 `capacity-profile.json`。

因此每个 R0/R1 G2 还必须运行预注册、只用于容量的 length-stress fixture：保持同一模型、batch/group、
precision、offload 和真实 actor/reference/update 路径，禁用提前停止并覆盖配置允许的 memory/final 生成
上限与 12288 token tensor/logits 路径。该 fixture 的样本、强制长度和输出明确标为 non-scientific，
不进入 reward 曲线；正式随机 G2 另行记录每轮生成长度、总 token volume 和峰值，不能用短输出冒充
最坏容量证据。Telemetry 必须记录实际 logits dtype、shape、numel、理论 bytes，以及它与 activation、
unshard 和 log-prob 临时张量的重叠窗口。

Length stress 恰好执行 1 个 target-profile optimizer loop（batch2/group4），成功或失败后立即终止，
不产出可续训 checkpoint，并使用独立的阶段实测 timeout 与硬上限。其固定样本、强制长度策略、ordered
IDs/hash 和 2B tokenizer identity 由 CPU 构建为独立 non-scientific bundle 并写入 handoff；GPU 不得
临时生成 fixture 或改用 formal train bundle。

只有同一 profile 的 G2a、G2b、length-stress、artifact/provenance 和费用门禁全部通过后，才原子发布
`capacity-profile.json`。B/C 不允许使用未通过完整 G2 链或尚未封存的配置。

### 5.5 G2 通过线

| 指标 | 绿色 | 黄色 | 红色/失败 |
|---|---|---|---|
| PyTorch peak allocated | `<=27 GiB` | `>27` 且 `<29 GiB` | `>=29 GiB` 或 OOM |
| PyTorch peak reserved | `<=28 GiB` | `>28` 且 `<30 GiB` | `>=30 GiB` 或 OOM |
| NVML/整卡 peak used | `<=29 GiB` 且 headroom `>=2 GiB` | headroom 1-2 GiB | headroom `<1 GiB` 或 OOM |
| step3->5 post-step resident baseline | 增长 `<=512 MiB` | 512 MiB-1 GiB | 单调增长且差值 `>1 GiB` |
| allocator | 无 alloc retry | 有零星压力信号 | 重复 retry/碎片化失败 |
| CPU RAM | 峰值低于实例可用量 80% | 80%-90% | 接近耗尽或使用 swap |
| step | 5 steps 与 fresh-process resume 均完成 | 抖动明显 | hang/超时/非零退出 |
| 数值 | finite loss/grad，adapter 更新非零 | 需解释异常 | NaN/Inf/base 变化 |
| GRPO | 至少一组 reward variance 和非零 advantage | 方差偏低 | 全组恒定且无法解释 |
| 格式 | 无系统性失败/全截断 | 截断偏高 | parser/格式整体崩溃 |

R0 只有因 PyTorch/NVML headroom 或 allocator 容量指标进入黄色时才必须测试 R1；不能直接用贴近
满显存的 R0 长跑。数值、格式、GRPO、resume 或 host-RAM 黄色必须在同一 profile 修复，不能触发
offload 切换。R1 只有容量进入绿色、其它门禁全部通过且费用投影通过才可封存。任何 offload 变化都
要求完整重跑 G2a/G2b 和 length-stress fixture，旧 step time 与显存证据立即失效。

Telemetry 必须同时采集 PyTorch allocated/reserved 和独立 NVML/`nvidia-smi` 整卡高水位，并按
rollout、reward、actor log-prob、reference log-prob、update 和 save 分段。G2 的 5 steps 只证明
bounded stress；B/C pilots 的全部 3 steps 及每个正式 20-step segment 的前 5 steps 继续执行增长
watchdog，越线时两边共同暂停。

每个 step 开始前 reset PyTorch peak counters，并为 NVML sampler 打开独立 step window；容量判定取
各 step、各分段的 peak。泄漏趋势不比较进程生命周期累计 high-water，而是在相同 cleanup + CUDA
synchronize 点记录 post-step resident allocated/reserved/NVML used，检查 step3、4、5 是否单调以及
`resident_step5 - resident_step3`。正式各 segment 也使用相同采样语义。

---

## 6. 正式训练与停止规则

### 6.1 B/C pilots

B、C 各做固定 3-step pilot。CPU 预先封存同一有序 6-group pilot manifest；两边从相同 LoRA init、
prompt IDs/order 和 seed 开始，禁止看到 reward 后换 prompt 或 seed 重抽。门禁要求：

- step-zero adapter hash、首 batch IDs 和 step1 sampled token hash 一致；
- step2-3 继续读取相同 prompt order，但允许 actor 经不同 reward 更新后生成轨迹合法分叉；
- 两条件的科学参数只差 `algorithm.alpha`，完整 leaf diff 必须恰好等于 4.2 的 allowlist；
- 跨 3 steps 汇总后，每个 arm 至少 2/6 groups 产生非零 total advantage；
- C 至少一组 centered state advantage 非零并映射到正确 action；
- 在共享 step1 trajectory 上逐元素验证 `A_B=A_out`、`A_C=0.8*A_out+0.2*A_state`；
- learned callback query、空 query、重复 query、检索命中和格式均可观测；
- 日志、checkpoint 和 adapter 目录完全隔离，pilot 只在 step3 保存一次。

基础设施失败只能用同一 manifest 显式 retry，并保留旧 attempt；协议实现错误必须两边修复重跑。
若固定 6 groups 仍不满足 advantage 门禁，这是预注册的能力/科学停止结果，不允许通过重抽样“修好”。
Pipeline 应将它发布为带完整证据的 non-retryable `scientific-stop` 终态，而不是普通 `failed-stage`；
相同 identity 下 `--retry-failed-stage` 必须拒绝反复抽样，且该终态不能成为后续训练的成功 predecessor。

该终态使用独立且稳定的 launcher 契约：`exit-code=42`、`.scientific-stop` 内容为 `42`，并在
`terminal.json` 记录 `outcome=scientific-stop`、`retryable=false` 和 evidence hash；`.success`、`.failed`
与 `.scientific-stop` 必须恰好存在一个且和 exit code 匹配。它在 durable sync 后可以发布
`shutdown-safe` 并按授权请求关机，但不能被伪装为 success/failed，也不能产生可复用 predecessor。

### 6.2 B/C 0 -> 40

1. B20、C20 分别从相同 base 和同一 LoRA init identity 开始；
2. 验证 step20 checkpoint 后，用 sealed B40/C40 config 在新进程显式恢复并继续到 total step40；
3. 同层 B/C 可顺序运行，不能并发争抢 GPU；
4. 每 10 steps 只输出训练健康指标，不用训练 batch 冒充 validation；
5. B20/C20 以及所有后续 20-step segment 各自前 5 steps 都继续显存 watchdog，并用新
   `T_compute` 重新投影 450/500 元两条预算线；
6. 任一投影越线时当前 stage 发布受控停止证据、另一 arm 不继续，不能为“跑齐”而突破硬线；
7. step20/40 保存后先校验完整 checkpoint，再按保留策略安全清理更旧点；
8. step40 另做 full-state fresh-process resume probe，再完成 adapter export、merge/reload；
9. 对 Base/B40/C40 运行 32 QA/格主评测和 C40 callback 消融；
10. 发布训练曲线、行为指标、显存/时长/费用和失败记录。

B/C40 完整结束即达到最低正式交付 L1。即使 C 没有超过 B，也先完成预注册评测和负结果报告。

### 6.3 是否续到 B/C80

只有同时满足以下条件才继续：

- B40/C40 checkpoint 都能从新进程恢复；
- 两边没有 reward、format、callback 或长度 collapse；
- G2 封存的 capacity profile 未改变；
- B/C40 配对评测已完成，但续训决策不依赖哪一边当前领先；
- 将 `T_L2_increment_raw + E_remaining_raw` 代入第 8 节成本公式后，`C_gpu_target <= 450` 且
  `C_total_target <= 500`；
- B80/C80 都有足够预算，不能只续其中一个。

满足后先用 sealed B60/C60 config 从明确的 `global_step_40` 续到 total step60，再用 B80/C80 config
从已验证的 step60 续到 total step80。最终 endpoint 另做 full-state fresh-process resume probe，并完成
64 QA/格最终评测。候选方案不规划 120 steps；优先把 B/C80、callback 消融、统计和报告做完整。

### 6.4 禁止的临场救火

- 不把 group 只在 B 或 C 一边改小；
- 不把 actor/Adam 改成 BF16 正式口径；
- 不在正式训练开始后切 QLoRA、vLLM/SGLang 或模型 revision；
- 不因 C 暂时落后而改 seed、挑 checkpoint 或只延长 C；
- 不把 G1 短序列显存结果当成 G2 正式长度证据；
- 不在 R1 失败后自动进入 actor param offload；此时必须重新决策硬件或最低方案。

---

## 7. 评测与科学证据

### 7.1 四个主格

| 数据集 | 200 documents | 800 documents |
|---|---:|---:|
| HotpotQA（ID） | 必做 | 必做 |
| 2WikiMultiHopQA（OOD） | 必做 | 必做 |

CPU 对四格各封存一个有序 64-QA manifest。L1 必须使用每格同一 manifest 的前 32 条，L2 使用完整
64 条；Base、B、C 和三种 callback modes 的 QA IDs 与顺序完全相同。800-doc 评测动态消费全部
chunks，不能继承训练的 6-chunk 限制；任何静默截断均失败。

### 7.2 模型与 callback 矩阵

主结果：

1. Base Qwen3.5-2B；
2. B40/B80：learned callback，`alpha=1.0`；
3. C40/C80：learned callback，`alpha=0.8`。

共同最终 C endpoint 的推理消融：

4. C_final + learned；
5. C_final + none；
6. C_final + fixed_question。

去重后每个交付层级恰好有 20 个唯一 run cells：四个 dataset/document 格分别运行
`Base+learned`、`B+learned`、`C+learned`、`C+none`、`C+fixed_question`。这里 C learned 同时服务
主结果和 callback 消融，只计算一次；不得误实现为 Base/B/C 与三种 callback 的全笛卡尔积。
因此 L1 verified package 含 `20 x 32 = 640` 次 model-QA，L2 最终 package 含 `20 x 64 = 1280`
次 model-QA。L2 只有 Base cell 的已验证前 32 条可在所有 identity 完全相同且 evaluator 确定性时
复用；B40/C40 不能冒充 B80/C80，费用投影按实际剩余调用数计算。

`none` 必须禁止检索和 recalled state 注入；`fixed_question` 每次使用原问题检索；`learned` 使用模型生成
query。三种模式必须由运行时 metadata 证明，而不是根据目录名猜测。

CPU 阶段分别封存 `eval40_qwen35_2b_5090.yaml` 和 `eval80_qwen35_2b_5090.yaml`。前者只接受
Base/B40/C40、32 QA/格和 C40 callback 消融；后者只接受 Base/B80/C80、64 QA/格和 C80 callback
消融。样本数、endpoint step 或 checkpoint identity 不匹配时 fail closed，不能靠运行时参数把一份
通用 eval config 改成另一层级。

两份 eval config 还必须固定 greedy decoding：`do_sample=false`、`n=1`、`temperature=0`，并封存
memory/final caps、callback 轮次策略、chat template、tokenizer revision 和 evaluator version。训练 rollout
的 `temperature=1.0` 不得泄漏到外部评测；否则同 QA 配对、Base prefix 复用和 bootstrap 都不成立。

当前实现可复用的是单 cell recurrent runner、答案指标和 paired-bootstrap primitive，不是完整矩阵
调度器。迁移必须新增 matrix dispatcher，按 dataset/doc-count/model/callback 生成不可变 cell identity，
验证跨 Base/B/C cell 的 QA ID 与顺序完全一致，去重 C learned 主结果与 callback learned 消融的同一
cell，并在所有必需 cell 验证成功后才原子发布聚合表和 verified evaluation package。

训练产物与评测证据必须形成可验证的身份链：checkpoint extra state、adapter metadata 和 merged
metadata 均嵌入 CPU handoff SHA、selected capacity-profile SHA、sealed training config ID/SHA，
以及本次 runtime-bound config SHA；verifier 逐级核对语义 SHA。B/C eval cell 再绑定 endpoint adapter metadata SHA、checkpoint
extra-state SHA、eval config SHA、manifest SHA 和 ordered QA-ID SHA。Base cell 只绑定固定 base
revision、tokenizer/template 和 eval identities，不伪称来自 capacity-selected training checkpoint。

### 7.3 指标

- 预注册 primary：四格等权 macro answer EM 的 `C-B` paired delta；
- secondary：answer EM、token F1、substring EM，以及 B-Base、C-Base、C-B 分格 delta；
- 10,000 次 paired bootstrap 95% CI；
- 合法 update/recall/final 格式率；
- callback 触发率、有效 query 率、重复/空 query 率；
- recalled supporting-document proxy、平均回看距离；
- memory/final 截断率和 processed document count；
- 每个阶段 wall time、tokens/s、峰值 VRAM、CPU RAM、磁盘和实际费用。

只有一个训练 seed 时，统计结论必须表述为该固定缩小实验上的 paired evidence，不能推广为跨 seed 稳定性。
10,000 次 bootstrap 在每个 dataset/document stratum 内按相同 QA index 对 Base/B/C 配对重采样，再将
四个 stratum 的 delta 等权平均；不得把 128/256 个 QA 扁平化成 IID 池。CI 只覆盖固定 checkpoint
下的题目变异，不覆盖训练 seed、checkpoint 选择或硬件不确定性；L1 的 32 QA/格结果明确标为探索性。
F1、substring、callback/support proxies 和案例均为 secondary/descriptive，不能在看到结果后替换
primary contrast。

### 7.4 结果解释

理想结果首先是预注册 primary `C-B` 四格等权 macro answer EM 为正；F1、800-doc 分格表现、
learned-control 和 callback support proxy 只提供支持或机制解释，不构成可事后挑选的替代成功标准。
最低可展示结果不要求所有数字为正，但必须有：

- 完整 Base/B/C 主表；
- 至少一个非平凡、可解释的 learned callback 轨迹；
- 正确的 checkpoint/resume 和 adapter/merge 证据；
- 对 C-B 为零或负的结果、方差和 2B 容量限制的完整解释；
- 不筛格子、不挑 checkpoint、不把工程成功包装成论文效果复现。

---

## 8. 时长、成本与计费门控

### 8.1 分项实测，不预填速度

不能用一个包含模型启动、resume 和 checkpoint 的 G2 job wall time乘以 86。必须从阶段化 telemetry
中分别记录，所有时间统一使用秒：

| 符号 | 定义 |
|---|---|
| `T_init` | fresh job 从进程启动到模型/worker ready 的时间 |
| `T_resume` | resume job 启动、读取并验证 checkpoint 到 ready 的时间 |
| `T_compute` | 单个 optimizer step 的 rollout/reward/log-prob/backward/update，不含启动和保存 |
| `T_save` | 一次完整 checkpoint 写入、验证和同步时间 |
| `T_artifacts40/80` | 两个 arm 合计的 adapter export、merge/reload、验证和打包时间，不含 resume probe |

R0/R1 分别计时，不能共用。G2b step2-5 给出第一版 `T_compute`；随后用 B/C pilots 和正式两边
滚动校准。任一 arm 尚不足 10 个 clean steps 时使用 observed max，不估不稳定的 p90；每个 arm
达到至少 10 steps 后分别计算 p90，正式投影取 B/C 两边较大值。一次性启动、保存和产物时间不得
塞回每个 optimizer step 重复计算。

### 8.2 训练公式与条件场景

按当前任务和 `save_freq`：

~~~text
T_L1_train_raw = 4*T_init
               + 4*T_resume
               + 86*T_compute
               + 6*T_save
               + T_artifacts40

T_L2_increment_raw = 6*T_resume
                   + 80*T_compute
                   + 4*T_save
                   + T_artifacts80
~~~

其中 L1 的 4 次 fresh init 来自 B/C pilots 和 B20/C20；4 次 resume 来自 B40/C40 的 step20 启动及
两个 step40 endpoint probes；86 steps 来自两个 3-step pilots 加 B/C40；6 次保存来自两个 pilot
step3 及 B/C 的 step20/40。L2 的 6 次 resume 来自 B60/C60、B80/C80 和两个 step80 endpoint
probes；4 次保存来自两边的 step60/80。

下表只展示 `86*T_compute` 与 `80*T_compute` 的纯 step 情景，不含启动、保存、产物和评测：

| 若实测保守 `T_compute` 为 | L1 的 86 steps | L2 增量 80 steps | 累计 166 steps |
|---:|---:|---:|---:|
| 15 分钟 | 21.5 小时 | 20 小时 | 41.5 小时 |
| 30 分钟 | 43 小时 | 40 小时 | 83 小时 |
| 60 分钟 | 86 小时 | 80 小时 | 166 小时 |

这些是条件场景，不是预期区间或性能承诺。每步有 8 条 trajectory，每条最多经历 6 个 memory state
和 final generation；HF generation micro batch 为 1，offload 与长输出可能使实际速度落在宽区间。

### 8.3 评测投影

评测不包含在训练公式中。正式评测前按 dataset、200/800 documents、model 和 callback mode 分层，
各跑预注册的少量固定样本。分别记录 context tokens/chunks 和尾部延迟，不能把所有样本混成一个 median：

~~~text
L_cell_seconds = observed_max_seconds                  if probe_count < 10
                 observed_p90_seconds                  if probe_count >= 10

E_cell_raw = L_cell_seconds * remaining_samples
             + model_load_and_adapter_switch_seconds
             + bounded_retry_seconds

E_remaining_raw = sum(all remaining cells and callback modes)
~~~

对应的 B/C cells 做成对投影时取两边 `L_cell_seconds` 较大值；Base 和 callback-only cells 使用各自
分层观测。`bounded_retry_seconds` 只覆盖预注册的瞬时基础设施重试，不覆盖 OOM、NaN、schema、科学
失败或换样本重抽。`E_cell_raw` 和 `E_remaining_raw` 与训练公式一样统一为秒；只在乘实时小时单价时除以 3600。
800-doc 评测必须完整消费数据，因此时间可能比显存更先成为瓶颈。不能为了省时只评前 30K token。

### 8.4 预算停止线

在用户未给新预算前，沿用：

- 总预算 500 元；
- GPU 消费硬停止线 450 元；
- 在 500 元总额与 450 元 GPU 硬线之间至少保留 50 元给磁盘、排障和不可预见支出；
- 训练与评测原始投影相加后统一乘一次 1.20 contingency，不重复叠加。

使用 AutoDL 创建实例时显示的实时整机价 `P`：

~~~text
C_gpu_target = C_gpu_done
             + P * 1.20 * (T_train_raw_seconds + E_remaining_raw) / 3600

C_total_target = C_gpu_target
               + C_non_gpu_done
               + C_disk_remaining
               + C_ops_reserve
~~~

`T_train_raw_seconds` 取当前决策点之后仍未执行的 L1 或 L2 分项训练时间；`E_remaining_raw` 只计剩余
评测。`C_gpu_done` 使用实际已发生的全部 GPU 费用，不再乘 contingency，并包含 G0/G1/G2、失败的
R0、所有 retry 和此前训练/评测 attempt；`C_non_gpu_done` 记录已发生 CPU/存储费用。
`C_disk_remaining + C_ops_reserve` 与已发生非 GPU 费用共同维护原计划的 50 元缓冲。任何阶段都必须同时满足 `C_gpu_target <= 450` 和
`C_total_target <= 500`。
其中 `C_ops_reserve >= max(0, 50 - C_non_gpu_done - C_disk_remaining)`，可以由操作者上调，不能为让
投影过线而下调成负数。

若 B/C40 预计超过停止线，方案在正式训练前停止；不得先跑一边再赌另一边。若 B/C80 超预算，则以
完整 L1 交付收尾，而不是牺牲主评测、callback 消融或恢复证据。

---

## 9. 云端执行拓扑

候选目标入口按付费决策拆分：

~~~text
CPU 一键 preparation
  -> 固定 commit/env/assets/data/config/handoff
  -> 成功或终态失败后关机

5090 bounded gates
  -> hardware/kernel G0
  -> 2B G1 step1/resume2/artifacts
  -> 成功或终态失败后关机

5090 2B capacity R0 launcher
  -> G2a R0
  -> G2b R0
  -> length stress R0
  -> 发布 capacity-profile.json
  -> 成功或终态失败后关机

若且仅若 R0 是容量黄色/失败：人工检查并显式批准
  -> 新 5090 capacity R1 launcher
  -> 从头 G2a -> G2b -> length stress R1
  -> 发布 R1 capacity-profile.json
  -> 成功或终态失败后关机

5090 B/C40
  -> B pilot -> C pilot
  -> B20 -> C20 -> B40 -> C40
  -> endpoint resume probe/checkpoint/export/merge
  -> 32 QA/格 + callback 消融
  -> verified package
  -> 成功或终态失败后关机

人工检查费用与结果门禁
  -> 可选 5090 B/C80
  -> B60 -> C60 -> B80 -> C80
  -> endpoint resume probe
  -> 64 QA/格 + callback 消融
  -> final verified package
  -> 成功或终态失败后关机
~~~

G2、B/C40 和 B/C80 绝不合并为一次无界启动。每段 launcher 必须持久化原始退出码、显式 predecessor、
resolved config、capacity profile、GPU ledger 和关机状态。

---

## 10. 从当前实现迁移

### 10.1 可复用的基础原语

- bootstrap、固定 commit、持久环境、flock、原始退出码和安全关机原语；
- CUDA 13 / sm_120 环境 lock 与固定 kernel source；
- Qwen3.5 text-only loader、thinking/template contract；
- PEFT/FSDP LoRA、HF recurrent rollout 和 reward/protocol；
- 数据 builder、200/800 nested manifests 和 provenance 原语；
- checkpoint/resume、adapter export、merge/reload 与语义 hash verifier；
- 单 cell Transformers recurrent runner、答案指标、callback modes 和 paired-bootstrap primitive；
- CPU handoff 的自哈希、路径 containment 和资产完整性检查框架。

这些是可复用 primitive，不等于 2B 正式交付已经“完全复用”。Active profile schema、33-config exact
inventory、G2/B/C DAG、matrix dispatcher、跨 cell 配对校验、聚合和 verified package 都需要新增；
`launch.sh`、worker、pipeline 和 stage runner 也要扩展后才能承载新 phases。

### 10.2 修改或新增

| 范围 | 候选改动 |
|---|---|
| 权威文档 | 评审通过后将本方案提升为权威，并更新 handoff/README |
| Profile namespace | 将 active profile ID 纳入 persistent root、pipeline identity、handoff 和 output 路径，隔离旧 4B state |
| Hardware profile | 严格验证 RTX 5090、单卡、>=31 GiB、sm_120/CUDA13、无其它 compute process、启动空闲显存 >=29 GiB |
| Host resources | CPU preflight 新增 >=24 cores、>=48 GiB RAM、初始 >=200 GiB 持久盘；每个 GPU phase 重验 profile RAM 和“预计写入量 + 终态 reserve”的剩余盘 |
| Asset profile | 新建仅含 0.8B、2B 和所需数据的 active manifest；旧含 4B manifest 留在 inactive profile，不能在旧 manifest 中运行时跳过 4B |
| Formal bundles | 在 2B namespace 使用固定 2B tokenizer identity 重建 train/validation/eval，并单独封存 non-scientific length-stress bundle；不能沿用 4B identity |
| Config/handoff | resolver 封存 33 份 active resolved configs；handoff 对 profile、config 和 asset exact-key 校验 |
| Capacity evidence | 新增 `capacity-profile.json`、峰值显存/RAM/step-time ledger |
| Pipeline | 在现有 G0/G1 后新增独立 G2、B/C40、B/C80 phases，并更新 phase adoption/verification 与 predecessor DAG |
| Recovery | 绑定 capacity profile、offload profile、step/predecessor 和 artifact identity；resume 实现错误不得靠切 R1 掩盖 |
| Evaluation | 新增 eval40/eval80 fail-closed configs、matrix dispatcher、跨 cell QA 顺序验证、聚合和原子发布 |
| Packaging | 输出 Base/B/C、callback 消融、资源账本、失败记录和 resume 证据 |
| Tests/audit | 更新 GPU probe；新增 2B config matrix、G2/profile、recovery、dry-run、shutdown negative tests，并更新严格 `cloud-audit-policy.json` |

### 10.3 建议的新配置文件

十四个训练/容量逻辑 source configs：

~~~text
verl/trainer/config/reproduction/g2a_qwen35_2b_5090.yaml
verl/trainer/config/reproduction/g2b_qwen35_2b_5090_step1.yaml
verl/trainer/config/reproduction/g2b_qwen35_2b_5090_resume5.yaml
verl/trainer/config/reproduction/g2_length_stress_qwen35_2b_5090.yaml
verl/trainer/config/reproduction/b_pilot_qwen35_2b_5090.yaml
verl/trainer/config/reproduction/c_pilot_qwen35_2b_5090.yaml
verl/trainer/config/reproduction/b20_qwen35_2b_5090.yaml
verl/trainer/config/reproduction/c20_qwen35_2b_5090.yaml
verl/trainer/config/reproduction/b40_qwen35_2b_5090.yaml
verl/trainer/config/reproduction/c40_qwen35_2b_5090.yaml
verl/trainer/config/reproduction/b60_qwen35_2b_5090.yaml
verl/trainer/config/reproduction/c60_qwen35_2b_5090.yaml
verl/trainer/config/reproduction/b80_qwen35_2b_5090.yaml
verl/trainer/config/reproduction/c80_qwen35_2b_5090.yaml
~~~

另新增两个 fail-closed eval source configs 和两个 compose overlays：

~~~text
verl/trainer/config/reproduction/eval40_qwen35_2b_5090.yaml
verl/trainer/config/reproduction/eval80_qwen35_2b_5090.yaml
R0 overlay: actor/ref/optimizer offload = false/false/false
R1 overlay: actor/ref/optimizer offload = false/true/false
~~~

CPU resolver 对十四个训练/容量任务分别 compose R0/R1，输出名称带明确 `_r0`/`_r1` 后缀的 28 份完整
resolved YAML；加上沿用的 G0/G1 三份和 eval40/eval80 两份，共 33 份。这里 overlay 字段顺序按
`actor param / ref param / actor optimizer` 书写，最终仍以字段名而不是位置解释。GPU 不接受
`+ref.param_offload=true` 一类临时 override。

现有 4B configs 可保留在 Git 历史或独立 inactive profile，但不进入 2B active resolver 的 source
allowlist、resolved index、asset manifest 或 handoff。新测试既要证明旧 4B 文件未被意外改写，也要
证明 2B B/C resolved configs 除预注册差异外完全相同。

### 10.4 建议的新云端入口

名称可在实施时按现有风格微调，但职责必须分离：

~~~text
scripts/cloud/start_gpu_gates.sh             # G0/G1，改为 5090 profile
scripts/cloud/start_gpu_2b_capacity.sh       # G2a/G2b/length stress + profile seal
scripts/cloud/start_gpu_2b_bc40.sh           # pilots + B/C20->40 + 中期评测/包
scripts/cloud/start_gpu_2b_bc80.sh           # B/C40->60->80 + 最终评测/包
scripts/cloud/export_2b_results.sh            # 只验证/打包，不训练
~~~

`start_gpu_2b_bc80.sh` 必须要求已验证的 B/C40 package 和显式费用批准 marker，不能因目录存在就自动续跑。
该一次性、自哈希 marker 绑定 active profile/commit、capacity profile、B/C40 package hash、两边 step40
checkpoint hashes 和本次成本投影；旧 pipeline 或内容为空的同名文件必须拒绝。

新增入口并不够；以下共享路径必须同步修改和测试：

| 文件/范围 | 必须承担的变化 |
|---|---|
| `init_cloud.sh` / `lib/runtime.sh` | 从 active profile ID 派生并验证独立 cache/data/cloud/output roots，写入持久 runtime env |
| `launch.sh` / `launcher_worker.sh` / `status.sh` / `lib/shutdown.sh` | 新 phase allowlist、request identity、三类 terminal marker、全程持锁、终态先同步再决定关机及正确状态展示 |
| `run_pipeline.sh` | G2、B/C40、B/C80 的有序 DAG、显式 predecessor、成功复验、保守 retry/adoption |
| `run_stage.sh` | 各 stage 命令、timeout、离线环境、config/capacity/checkpoint verifier 和 telemetry |
| `cloud_state.py` / `resolve_configs.py` | active profile schema、33-config exact inventory、2B asset/data namespace 与自哈希 handoff |
| `gpu_probe.py` | 5090 identity、host RAM/disk、空闲显存、其它 compute process、整卡峰值和 CUDA13 门禁 |
| artifact/eval helpers | resume5、selected profile、matrix cell、checkpoint 与 package identity 校验 |
| `cloud-audit-policy.json` / tests | 精确入口、shutdown owner、test hook，以及 success/failure/signal/lock/sync/authorization 负例 |

`scripts/cloud/README.md` 顶部必须保留最短正常路径：一个完整 CPU bootstrap block，以及当前人工批准
付费阶段的一条 GPU 命令；stage-level 命令只用于诊断。G2、B/C40、B/C80 每段都在 success、
`scientific-stop` 或 terminal failure 证据 durable sync 后才可请求关机，锁冲突、dry-run、keep-running、test mode、sync/authorization 失败
必须保持实例运行。Guest poweroff 后仍需人工确认 AutoDL 控制面已停止计费。

README 还必须要求操作者确认同一个 provider volume ID、CPU 实例已经停止并卸载后才能挂到 GPU，
且任何时刻不得有两个 host 写同一卷；guest 内 `flock` 不能防止跨主机双写。每个 GPU phase 都要在
CUDA 编译、模型加载或付费长任务前重新验证 sealed handoff、predecessor 和 active profile，并在
`HF_HUB_OFFLINE/HF_DATASETS_OFFLINE/TRANSFORMERS_OFFLINE` 下运行，缺资产时失败而不是联网补齐。

每个新入口还必须在 README 给出对应 `status`、日志 tail、显式 retry 和 `--keep-running` 诊断命令，
并解释 launcher admission 不等于 phase success。外层 bootstrap 下载失败时仓库脚本尚未运行，无法
自动关机，操作者必须立即回控制台停止实例。

当前 `run_stage.sh` 的统一 6 小时训练 timeout 不适用于正式阶段：即使 `T_compute=15` 分钟，单个
20-step segment 的纯计算也约 5 小时。G2 封存 profile 时必须根据实测分项发布每类 stage 的有界
timeout，例如训练段至少覆盖 `T_start + 20*T_compute + T_save` 再加一次预注册余量，其中 fresh
B20/C20 使用 `T_start=T_init`，其余恢复段使用 `T_start=T_resume`；pilot、artifact
和每个 eval cell 分别计算，不能共享一个魔法常量。Timeout 仍必须有限，并同时设置 AutoDL 控制面
最长运行时限和余额告警；guest 内脚本无法处理宿主掉电、SIGKILL 或 launcher 被整机 OOM 杀死。

Retry 只能由操作者检查原因后显式启动，并从最近一个重新验证成功的 20-step predecessor 开始。
Adoption 仅允许“完整 checkpoint 已原子发布，之后 cleanup/terminal publication 失败”的情形，由独立
verifier 生成新的 synthetic success attempt；partial checkpoint、OOM、NaN、schema/hash mismatch、
科学门禁失败或被信号打断的未完成 stage 永不收养，旧 failed attempt 始终保留。

### 10.5 迁移验收与负例

实施不能只跑 happy path，至少覆盖：

- 每个新 phase 的参数 allowlist、dry-run、ordered DAG、predecessor revalidation 和跨 phase 拒绝；
- tampered/missing/extra CPU handoff、capacity profile、resolved config、checkpoint、B/C40 package 和费用批准 marker；
- R1 未显式批准、R0 非容量失败、旧 R0 output 被误用，以及 `scientific-stop` retry/predecessor 拒绝；
- partial/OOM/NaN/signal attempt 不可 adopt，完整产物后 cleanup 失败才可 synthetic adoption；
- GPU 缺资产时保持 offline 并失败，compute process、VRAM、CPU/RAM/disk、mount/symlink/path escape 门禁；
- 每个入口的 success、nonzero、INT、TERM、lock conflict、keep-running、dry-run、test mode、sync、authorization 和 shutdown backend failure；
- 三类 terminal marker/exit-code/status/关机授权顺序，测试 hook 永不调用真实 shutdown/poweroff backend；
- `bash -n`/可行时 ShellCheck、Python `compileall` 与相关测试、33 份生产 config compose、静态 cloud audit、`git diff --check` 和新 shell mode `100755`。

交付 commit push 后还必须验证远端分支解析到预期 40 位 SHA，并实际读取该 SHA 对应的 raw
`bootstrap.sh`；README 的 exact commands/literals 与 `cloud-audit-policy.json` entrypoints 必须一致。

---

## 11. 实施顺序

### Phase A：文档与 profile

1. 用户确认本候选方案；
2. 更新权威 plan/handoff 和云端 README；
3. 增加 `rtx5090-32g-qwen35-2b-v1` hardware/experiment profile；
4. 明确保留旧 4B profile，不做 git revert。

### Phase B：配置和 CPU 证据

1. 新增十四个训练/容量逻辑 configs、eval40/eval80 和 R0/R1 overlays；
2. 建立独立 2B active asset manifest、profile namespace 和绑定 2B tokenizer 的 bundles；
3. CPU compose 并封存 exact 33 份 active resolved configs；
4. 增加完整 config matrix、R0/R1 和 B/C pairwise-diff tests；
5. handoff 记录 profile、2B revision、asset/data/config tree 和期望 hardware profile；
6. 验证旧 4B assets/configs/state 不进入 2B active inventory。

### Phase C：5090 G0/G1

1. 修改所有 PRO6000/90GiB 硬编码及测试，并新增单卡、空闲显存和 compute-process 门禁；
2. 在 5090 上离线编译并封存 kernel evidence；
3. 跑现有 G0/G1；
4. 确认 checkpoint、adapter、merge 和 eval 证据完整。

### Phase D：G2 capacity

1. 实现 telemetry、R0/R1 profile 和 `capacity-profile.json`；
2. 跑 G2a/G2b 和 capacity-only length-stress fixture；
3. 根据预注册显存门限选择 R0 或 R1；
4. 用实测 `T_init/T_resume/T_compute/T_save/T_artifacts` 和分层评测 probe 计算 B/C40 成本；
5. 未通过则停止，不进入正式训练。

### Phase E：B/C40 最低正式交付

1. B/C pilots；
2. 独立 B20/C20，再从 step20 显式恢复到 B40/C40；
3. checkpoint/resume、adapter/merge；
4. 用 eval40 dispatcher 完成 20 个唯一 cells、32 QA/格主表和 callback 消融；
5. verified package、成本表和 L1 报告。

### Phase F：可选 B/C80

1. 检查恢复、健康、费用和双边预算；
2. 从 step40 显式运行 B60/C60，再从 step60 运行 B80/C80；
3. 用 eval80 dispatcher 完成 20 个唯一 cells、64 QA/格主表和 callback 消融；
4. 最终 package、统计、案例和 L2 报告。

---

## 12. 风险与预注册处理

| 风险 | 早期信号 | 处理 |
|---|---|---|
| 2B 正式长度 logits 峰值 OOM | G2 容量指标红色/OOM | 仅容量失败允许 R0 -> R1，从 G2a 全量重跑 |
| R1 仍不够 | G2/length stress 仍 OOM、headroom 或 allocator 红色 | 停止，重新决策硬件；不自动 actor offload |
| Offload 极慢 | `T_compute`、PCIe 或 RAM 抖动 | 用完整分项公式重算；超预算则不进 B/C40 |
| group 4 reward 全同 | 固定 3-step pilot 未达到 2/6 非零组 | 排除实现错误后作为停止结果；不得换 prompt/seed 重抽 |
| 2B 格式/能力不足 | update/recall/final 大面积失败 | 完整报告能力边界；不挑 seed/checkpoint |
| B/C 配置漂移 | resolved leaf diff 超出 4.2 allowlist | fail closed，重新封存 configs |
| Checkpoint 无法恢复 | schema/step/RNG/optimizer/predecessor 不连续 | 在同一 profile 修复并重跑；不得切 R1 掩盖实现错误 |
| B/C40 超预算 | L1 分项训练时间 + `E_remaining_raw` 经一次 contingency 后超线 | 不启动正式双条件 |
| B/C80 超预算 | 增量投影超线 | 以完整 L1 收尾 |
| 800-doc 评测过慢 | eval probe 时间过高 | 保留四格，停在 32 QA/格；不截断文档 |
| C 未超过 B | delta<=0 或 CI 跨0 | 作为负结果报告，不选择性续训 |
| GPU 被占用或镜像非 CUDA13 | compute process/空闲显存/runtime/toolkit mismatch | preflight 落盘失败；仅在终态同步和授权复验后请求关机 |

---

## 13. 简历与最终产物

### 13.1 最低 L1 产物

- 一页项目摘要和系统架构图；
- 完整环境 lock、2B/5090 resolved configs 和 capacity profile；
- 数据 manifests/hashes；
- G0/G1/G2、B/C pilots、B20/C20->B40/C40 的日志与证据链；
- Base/B/C 四格主表；
- learned/none/fixed callback 消融；
- checkpoint/resume、adapter、merged model metadata；
- 训练曲线、callback/support proxy、案例轨迹；
- VRAM/RAM/step-time/费用 ledger；
- known limitations、失败记录和负结果解释。

### 13.2 简历表述模板

只能使用已经达到层级的模板，数字必须来自 verified package：

**仅代码/CPU preparation：**

> 为 ReMemR1 设计并实现单卡 Qwen3.5-2B 复现框架，包含固定 revision/依赖/资产、CPU-to-GPU
> 自哈希 handoff、可恢复阶段 DAG、离线 GPU admission 与失败安全关机。

此时不能写“已在 5090 跑通”、峰值显存、B/C 训练或科学结果。

**L0：**

> 在单张 RTX 5090 32GB 上完成 Qwen3.5-2B 的 CUDA/BF16、LoRA-GRPO 接口、正式长度容量与
> fresh-process checkpoint/resume 门禁，封存 R[实测] profile；峰值显存 [实测] GiB，门禁耗时/费用
> [实测]。

L0 不能写已完成 B/C40、callback 消融或正式科学评测。

**L1：**

> 在单张 RTX 5090 32GB 上将 ReMemR1 适配至 Qwen3.5-2B，设计并实现 LoRA-GRPO 的
> PEFT/FSDP、recurrent HF rollout、显存容量门禁、adapter checkpoint/merge 与长上下文评测闭环；
> 完成 B40/C40，在 HotpotQA/2WikiMultiHopQA 的 200/800-document 设置上以 32 QA/格比较
> outcome-only vs multi-level reward 及 learned/none/fixed callback，以 [实测指标] 观察到
> [实际趋势或负结果]；峰值显存 [实测] GiB、总耗时 [实测]、总成本 [实测] 元。

**L2：**

> 在 L1 闭环上成对续训 B/C 至 80 steps，并以 64 QA/格完成四格主评测与 callback 消融；报告
> [B-Base/C-Base/C-B 实测 delta 与 CI]、[实际趋势或负结果]、完整恢复证据及 [实测] 资源成本。

只有实际统计支持时才能把“观察到正向趋势”替换为更强表述；CI 跨 0 或负结果必须原样报告。

### 13.3 面试叙事

1. 为什么 4B/96GB 方案改为 2B/32GB，以及保留了哪些核心变量；
2. 如何估算 FP32 actor、BF16 ref、LoRA 和大词表 logits 的显存；
3. 为什么用 G2 预注册选择 R0/R1，而不在正式训练中临时救火；
4. 为什么 batch2/group4 会降低样本暴露和机制信号，如何诚实报告；
5. 如何保证 B/C 只差 reward alpha；
6. 如何证明 adapter 真在训练、base 未改变、checkpoint 真能恢复；
7. 如何用 callback 消融、200/800 退化和 distant-evidence 案例解释机制；
8. 哪些结果没有复现，以及为什么不能把缩小实验冒充论文结论。

---

## 14. 提升为权威方案前的检查清单

- [ ] 用户确认正式模型从 4B 改为 2B；
- [ ] 用户确认正式 GPU 固定为单张 RTX 5090 32GB；
- [ ] 用户接受 batch/mini/group `2/2/4` 和每步 8 trajectories；
- [ ] 用户接受固定 3-step pilots、2/6 非零 advantage 门禁和禁止换样本重抽；
- [ ] 用户接受正式训练按 20-step stages 切分及额外 resume 启动开销；
- [ ] 用户接受 B/C40 是最低正式交付，B/C80 是预算门控后的可选目标；
- [ ] 用户接受 R0 -> R1 的 offload 选择规则以及 R1 失败时停止；
- [ ] 用户接受 500 元计划预算和 450 元 GPU 停止线，或提供新预算；
- [ ] 用户接受 `C-B` 四格 macro answer EM 为 primary，其他指标不得事后替换；
- [ ] 用户接受 2B 结果只能声称缩小机制复现；
- [ ] 当前 4B configs 和方案保留为历史 profile，不做 revert；
- [ ] 更新权威 handoff 前再次核对工作区和远端 commit；
- [ ] 实施完成并通过 CPU tests 前不启动 5090；
- [ ] G2 通过并发布费用投影前不启动 B/C40；
- [ ] B/C40 两边均通过门控前不启动 B/C80。

---

## 15. 依据

- 当前权威方案：`docs/final_reproduction_plan_zh.md`；
- 当前实施入口：`docs/reproduction_implementation_handoff_zh.md`；
- 当前云端说明：`scripts/cloud/README.md`；
- 历史 2B/5090 规划：commit `08174dc` 中的 `docs/reproduction_plan_zh.md`；
- 当前代码与配置基线：commit `cc6c330ce1e8a1f922c28e84dc3f28b298914636`；
- 论文：`2509.23040v5.pdf`。

历史方案只提供容量、时长和降级参考；本候选方案以当前已实现的 LoRA/FSDP、HF rollout、
checkpoint、数据和评测契约为准。任何规划显存和时长均须由真实 5090 G2 证据替换。
