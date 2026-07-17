# ReMemR1 RTX 5090 / Qwen3.5-2B 项目可行性、成功率与简历路线分析

> 分析日期：2026-07-17
> 分析代码快照：commit `36f7166`，分支 `reproduction/rtx5090-2b`
> 迁移前实现基线：RTX PRO 6000 96GB + Qwen3.5-4B 适配版本
> 后续目标：单张 RTX 5090 32GB + Qwen3.5-2B
> 主要输入：`docs/rtx5090_2b_reproduction_plan_zh.md`、`docs/项目推荐.md`
> 文档性质：项目决策分析，不是 GPU 实测报告；文中的概率是基于上述代码快照、范围、预算和时间假设的主观工程区间，不是统计置信区间。
> 实施更新（2026-07-17）：本文“当前代码拒绝 5090”“尚未实现”等描述是 commit `36f7166` 的迁移前基线。后续代码已按 5090/2B active profile 适配，但尚无真实 5090 训练结果；成功率区间仍需用 G0/G1/G2 实测更新。新增 `compress_context` 已从当前主线排除。

---

## 0. 直接结论

### 0.1 这个项目值得继续，而且不需要“正向论文结果”才可以写简历

如果目标是得到一个能用于 Agent / LLM 训练工程岗位的项目，当前仓库已经不是一个空壳：Qwen3.5 text-only 适配、LoRA/FSDP、recurrent HF rollout、协议与奖励、checkpoint/resume、adapter export/merge、数据 provenance、评测 runner 和 AutoDL 状态机都已经有较大规模的实现和测试。

真正需要区分的是五种完全不同的“成功”：

| 目标 | 含义 | 从当前状态出发的估计概率 |
|---|---|---:|
| **诚实可写简历** | 有真实代码、固定测试、架构图、可回放 trajectory，最好再有真实 G1/G2 证据；不要求 C 优于 B | **80%-90%** |
| **严格 L0** | 5090 上完成 G-1/G0/G1/G2、正式长度和恢复/容量门禁 | **50%-65%** |
| **严格 L1** | 完成 pilots、B/C40、32 QA/格、callback 消融和 verified package；结果允许为负 | **28%-42%** |
| **L2** | B/C80、64 QA/格和完整最终包 | **13%-25%** |
| **L3 显著正向结果** | 预注册 `C-B` 四格 macro answer EM 为正且 paired 95% CI 不跨 0 | **3%-8%** |

注意：**commit `36f7166` 的迁移前代码不做任何修改就直接上 5090，成功概率是 0%**，因为当时的 hardware probe 会主动拒绝非 PRO 6000 / 90 GiB 环境。表中的概率都以先完成候选方案要求的 5090/2B 迁移为前提。

因此，最现实的项目定义不是“必须复现出论文提升”，而是：

> 在单张 RTX 5090 32GB 上，将一个长上下文 recurrent-memory Agent 适配到 Qwen3.5-2B，构建可恢复的 LoRA-GRPO 训练、Agent trajectory、工具行为评测和资源证据闭环；正向、无差异或负向实验结果均如实报告。

这已经足以成为项目。严格 L1 是高质量冲刺目标，L2 是可选项，L3 不应该作为项目成败标准。

### 0.2 推荐决策

1. **主线先做无新增工具的 2B/5090 baseline。**先得到真实 G1/G2，再决定是否承担 B/C40 长训练。
2. **不要现在迁移到 ROLL，也不要把 XML 协议改成 JSON。**当前仓库已经围绕 `verl`、`<update>` / `<recall>` 和完整 checkpoint/eval contract 做了大量适配；更换框架或协议会主动丢掉已有资产。
3. **当前项目排除 `compress_context` 和独立 D 组。**第 5 节只保留为早期方案比较，不进入当前实现、配置、预算或简历口径；未来如重新立项，必须使用独立分支、profile 和实验注册。
4. **不编数据。**没有正向结果可以写工程闭环、负结果、容量边界和失败分析；伪造提升不仅不必要，而且很容易被日志、配置、checkpoint、seed 和 CI 追问击穿。

---

## 1. 两份文档实际在回答不同问题

### 1.1 RTX 5090 / 2B 方案的目标

`rtx5090_2b_reproduction_plan_zh.md` 是一份偏科研复现和可审计工程的方案。它保留：

- B（outcome-only）与 C（outcome + state reward）成对训练；
- learned / none / fixed_question callback 消融；
- HotpotQA / 2WikiMultiHopQA、200 / 800 documents 四格评测；
- checkpoint/resume、adapter export/merge、配置与数据 provenance；
- 预注册 primary、预算门控和 scientific-stop。

它的优点是严谨、可解释、经得起追问；缺点是范围大；在 commit `36f7166` 快照中，33 份 active resolved config、G2 capacity、B/C40、20 个评测 cells 和 verified package 都还没有实现。

### 1.2 《项目推荐》的目标

`项目推荐.md` 更关心“面试时像一个 Agent 项目”，因此建议：

- 重点讲 Agent lifecycle、状态管理、Retriever、trajectory 和 action policy；
- RL 只是学习 memory action policy 的手段；
- 增加 `compress_context`，形成 update / recall / compress / answer 四类动作；
- 用 Base、Fixed、Prompted Agent、GRPO Agent 和 ablation 做对照。

这个方向适合简历叙事，但它对当前实现有两处关键误判：

1. 当前 Agent **不是** update / recall / answer 三选一的通用工具 Agent；
2. 当前 `<update>` 本身已经要求模型把旧 memory 与新证据重写成新的 working memory，本质上包含摘要和压缩。

因此，两份文档不能直接机械合并。正确组合方式应是：

> 以 5090/2B 方案作为可运行、可审计的 baseline；以《项目推荐》作为简历叙事和后续架构扩展方向，而不是立刻重写训练框架和动作协议。

---

## 2. 当前仓库审计

### 2.1 已经实现、可以复用的部分

| 范围 | 当前状态 | 主要证据 |
|---|---|---|
| Qwen3.5 text-only 与 thinking contract | 已实现 | `verl/models/qwen35.py`、`verl/utils/chat_template.py` |
| LoRA/FSDP 与 trainable/base 不变量 | 已实现并有测试 | `verl/models/lora_contract.py`、`tests/reproduction/test_lora_contract.py` |
| Recurrent HF rollout | 已实现 | `verl/workers/rollout/hf_rollout.py`、`recurrent/impls/memory_revisit.py` |
| `<update>` / `<recall>` parser 与 reward | 已实现并可在 CPU 测试 | `recurrent/protocol.py`、`recurrent/rewards.py` |
| 有序历史记忆与 top-1 retrieval | 已实现 | `recurrent/protocol.py:310`、`recurrent/impls/memory_revisit.py:893` |
| Checkpoint/resume、adapter export/merge | 已实现了严格 contract | `verl/utils/checkpoint/reproduction.py`、`scripts/reproduction/export_adapter.py` |
| 数据 bundle、manifest 与 provenance | 已实现 | `taskutils/data_synthesis/reproduction_builder.py`、`reproduction_manifest.py` |
| 单 cell recurrent eval 与统计 primitive | 已实现 | `taskutils/memory_eval/reproduction_runner.py`、`reproduction_metrics.py` |
| 云端 CPU preparation、G0/G1 状态机 | 已实现 | `scripts/cloud/run_pipeline.sh`、`run_stage.sh`、`launcher_worker.sh` |
| 失败安全关机、锁、恢复和 adoption | 已实现并有负例测试 | `scripts/cloud/lib/`、`tests/cloud/` |

从早期基线到当前分支，仓库约新增 3.6 万行实现和测试代码，共能检索到约 321 个 `test_` 函数。这说明项目的工程底座已经比较厚，后续不应再轻易换训练框架。

### 2.2 迁移前尚未完成的 5090 / 2B 正式部分

| 缺口 | commit `36f7166` 的迁移前事实 | 影响 |
|---|---|---|
| 5090 hardware profile | `gpu_probe.py` 和 `run_stage.sh` 仍要求 PRO 6000、至少 90 GiB | 当前 GPU 入口会直接拒绝 5090 |
| Active experiment namespace | 当前 handoff/config inventory 仍以 4B 正式 profile 为准 | 2B 与旧 4B state 尚未隔离 |
| 正式 2B G2/B/C configs | 目前只有 2B G1；G2、B/C40/80 正式配置仍是 4B | 不能直接启动 2B 正式训练 |
| G2 capacity profile | 尚无 R0/R1、length-stress、峰值显存/RAM ledger 和 `capacity-profile.json` | 32GB 是否能稳定容纳正式长度未知 |
| 正式长跑 DAG | 当前云端流水线只执行 CPU、G0、G1 | B/C pilots、B/C20/40/60/80 尚未接入 |
| 完整评测矩阵 dispatcher | 已有单 cell runner，没有 20-cell 去重、配对校验和原子聚合发布 | 尚不能生成严格 L1 verified package |
| 真实运行证据 | 仓库内没有 GPU 日志、checkpoint、capacity evidence 或正式 metrics package | 目前不能声称已在 5090 跑通或已有实验提升 |
| 权威文档切换 | handoff 和 cloud README 仍指向 4B/PRO6000 与旧分支 | 候选 5090 方案尚未成为操作入口 |

commit `36f7166` 的 `run_pipeline.sh` 只列出 CPU stages 和 `gpu-preflight -> g0 -> g1-step1 -> g1-resume2 -> g1-artifacts`；这与当时“一键流水线只到 G1”的判断一致。

### 2.3 迁移前分析的本地验证结果

本次分析没有安装 CUDA 训练依赖，也没有运行 GPU、网络下载或长训练。只做了与风险相称的静态和轻量验证：

- 云端静态审计：`errors=0 warnings=0 files_checked=29`；
- 协议、奖励、callback、评测指标和 GPU probe：`58 passed`；
- 额外 cloud 子集：`31 passed, 1 skipped`；另有 3 个失败仅由本地缺少 `pyarrow` / `hydra` 导致；
- 全量 collect 发现 324 个 test items，但本地 `.venv` 只有 pytest，因缺少 torch、datasets、tqdm、PyYAML、OmegaConf、NumPy 等依赖出现 11 个 collection errors。

这些结果说明 CPU 可测的核心 contract 和云端安全骨架有较好基础，但**不能替代** Linux/CUDA13/RTX5090、正式依赖环境、真实模型加载和长序列显存实测。

---

## 3. 成功概率的口径与假设

### 3.1 估计假设

以下概率均从当前 commit `36f7166` 出发，并假设：

- 从 2026-07-18 起有 5-6 周；
- 单人每周可以稳定投入约 15-25 小时；
- 使用已采集的单张 RTX 5090 32GB 固定机器；GPU 开启态 cgroup 为 `16 cores / 90 GiB RAM`，可准入 R0，但不满足 R1 的 128 GiB；无卡 CPU 阶段另按 `0.5 core / 2 GiB` 低资源环境准备模式设计；
- 总预算 500 元，GPU 硬停止线 450 元；
- 每个门禁最多做两次“保留完整证据、先定位原因”的重试；
- 不在训练中途静默降 dtype、换 seed、挑 checkpoint 或修改一边的配置；
- 负结果也算完成，只有缺失证据或没有闭环才算失败。

如果时间只有 2-3 周、R0 有效 RAM 低于 80 GiB、预算更低，或者同时重构工具协议/切换 ROLL，下面的 L0/L1 概率还要继续下调。

### 3.2 基础 2B 路线

| 交付层级 | 无条件概率 | 条件概率/解释 |
|---|---:|---|
| **简历工程包** | **80%-90%** | 至少争取真实 G1，尽量完成 G2；配合 trajectory、恢复/export/merge 和小规模固定 QA，不要求正式 B/C40 |
| **严格 L0** | **50%-65%** | 主要未知是正式长度大词表 logits 峰值、R0/R1 容量和 CUDA13/sm120 kernel 实机行为 |
| **严格 L1** | **28%-42%** | 条件于 L0 约 55%-65%；还需 86 个长 step、pilot 门禁、640 次 model-QA 和完整打包 |
| **L2** | **13%-25%** | 还需额外 80 个长 step、64 QA/格和更高费用；简历边际收益较小 |
| **`C-B > 0` 但不显著** | 条件于 L1 约 **35%-55%** | 方向为正不等于能证明有效 |
| **L3 显著正向** | **3%-8%** | 条件于完成 L2 约 15%-30%；单 seed、小样本和弱训练暴露使 CI 不跨 0 很难 |

这组数字看起来保守，是因为“代码能运行”与“完整预注册实验按时完成”不是一件事。2B L1 每个 arm 只有 320 条 trajectory，group 只有 4；固定 pilot 还可能因为同组 reward 全相同而触发预注册 scientific-stop。即使训练正常，800-document 评测也可能成为主要耗时。

### 3.3 为什么“可写简历”的概率明显高于 L1

对工程/Agent 岗位，能够解释并展示以下证据，已经构成完整项目：

- recurrent memory Agent 的状态流转和真实 trajectory；
- update、learned recall、none/fixed callback 的协议和消融；
- LoRA/FSDP/HF rollout 的单卡适配；
- checkpoint 在新进程中恢复，adapter export/merge 后输出一致；
- 正式长度容量门禁、峰值显存、step time 和失败边界；
- 固定小样本评测与格式/调用行为指标；
- 可恢复的云端 pipeline 和成本控制。

这些都是实际工程成果，不依赖 `C-B` 为正。相反，一个无法给出日志、配置和 checkpoint 身份的“提升 8%”，简历价值反而更低。

---

## 4. 基础路线的主要风险

| 风险 | 等级 | 为什么 | 处理方式 |
|---|---|---|---|
| 5090 32GB 正式长度 OOM | 高 | FP32 actor、BF16 reference、激活、log-prob 和约 248K 词表 logits 可能在同一窗口重叠 | 必须做 G2a/G2b/length-stress；只允许容量原因 R0 -> R1 |
| CUDA13 / sm120 kernel | 中高 | 当前仅有锁和 probe，没有本仓库的真实 5090 evidence | 先 G0，再 G1；失败保留 build log，不直接进入长训练 |
| 2B action/reward 信号弱 | 高 | group4、每步 8 trajectory、单 seed，容易出现全组同 reward | 固定 pilot，`2/6` 非零 advantage 不过就 scientific-stop |
| 正式 DAG 和配置量较大 | 中高 | 33 resolved configs、profile namespace、predecessor 与 verifier 均需新增 | 先完成 CPU config matrix，再租 GPU；不边跑边改配置 |
| 800-doc 评测过慢 | 高 | 必须消费完整 documents，20 cells 共 640/1280 model-QA | 分层 probe；L1 优先 32 QA/格，不牺牲主表追 L2 |
| 数据或 LFS 资产失败 | 中 | 当前流水线 fail closed，不允许未验证数据继续 | 在便宜 CPU 实例完成全部下载、hash 和 bundle 构建 |
| 只剩代码、没有实测 | 中 | 当前仓库没有可验证 GPU artifact | 把真实 G1 设为第一个强制简历里程碑 |
| 为追正向结果扩大范围 | 高 | 换 seed、挑 checkpoint、临时改 reward 会破坏可信度 | 预注册停止规则；负结果作为正式产出 |

---

## 5. 增加 `compress_context` 工具的历史分析（当前不实施）

### 5.1 它不是当前代码上的“一行新增工具”

当前协议由 `recurrent/protocol.py` 固定：每个中间响应必须有且只有一个非空 `<update>`，`<recall>` 只是同一响应中的可选标签。`MemoryAgent` 按 chunk cursor 固定推进，读完全部 chunk 后自动进入 final answer。

当前真实流程是：

```text
读取 chunk i
  -> 必须生成一次 update（重写 working memory）
  -> 可选同时生成 recall query
  -> runtime 检索一条历史 memory
  -> step + 1，继续读取下一 chunk
  -> 所有 chunk 用完后自动 final
```

《项目推荐》中假设的流程则更像：

```text
Agent 自由选择 update / recall / compress / answer
  -> dispatcher 执行工具
  -> 环境返回 observation
  -> Agent 再决定是否继续
```

两者不是同一个状态机。特别是当前 `<update>` 已经要求“保留旧 memory 的相关信息，并加入新证据”，它本身就是一次受生成长度限制的 working-memory 重写。若新增的 `compress(content)` 也只是把 `content` 写回 working memory，它与 update 基本同义，最终即使有差异也无法解释究竟来自新工具还是换了提示词。

当前 prompt 也不会无限累积所有历史 raw chunks：每轮主要包含当前 chunk、working memory 和至多一条 recalled memory。因此新增工具解决的是更严格的 token/成本预算，而不是修复一个已经存在的“上下文无限膨胀”故障。

完整四工具状态机还有一个隐藏风险：当前 `self.step` 同时充当 chunk cursor、rollout turn/seed 坐标、memory provenance index 和 reward 分组依据，并且每次生成后无条件递增。若插入一个“不消费新 chunk”的独立 recall/compress turn，继续递增会跳过 chunk，不递增又会破坏 seed、provenance 和 GRPO 分组。真正支持可变轮数工具调用，需要至少拆分 `decision_step`、`chunk_cursor`、逐样本 `finished_mask`、`max_agent_turns` 和 `no_progress_count`。

现有 reward 也容易被压缩动作放大利用：`memory_gain_reward` 的一部分使用当前 memory 自身词数作为 recall 分母，极短、只含答案词的 memory 可能获得高分；再叠加简单 token penalty，会进一步鼓励删除多跳证据。因此 compress reward 不能只是“原 state reward - token 数”，还要加入 supporting-fact/evidence retention、超预算和无效/重复调用惩罚。

若 compress 真正操作 raw context、history 或 context budget，则必须同步修改：

- prompt 与 action grammar；
- parser、format validity 和错误恢复；
- Agent state transition、chunk cursor 与 final 规则；
- working/history/recalled memory 的目标对象和 provenance；
- rollout token/batch contract；
- state reward、cost penalty 与 GRPO group variance；
- checkpoint identity、config matrix 和 resume compatibility；
- 训练/评测 runner、trajectory schema 和行为指标；
- Base/Prompted/RL/ablation 的实验矩阵。

因此，工具扩展是一个中等规模的 Agent 协议重构，不是轻量 UI 功能。

### 5.2 从第一天并入 GRPO 主线时的概率

| 结果 | 估计概率 | 说明 |
|---|---:|---|
| L0 + 可用 compress action | **30%-45%** | 同时承担 5090 迁移、正式容量和新状态机 |
| 自定义完整 L1 | **15%-30%** | 还需重做对照、reward、评测和 verified package |
| 自定义 L2 | **6%-15%** | 时间、费用和调试面都明显增加 |
| 显著提高回答准确率 | **2%-6%** | 工具与 update 语义重叠，2B 动作稀疏，正向信号很弱 |

这个方案不推荐。它会让 baseline 失败和工具失败互相污染，最后可能既没有可靠复现，也没有可解释的新工具结果。

### 5.3 推荐的工具路线

推荐按下面顺序做：

1. 先封存 baseline L1；如果时间不够，至少封存真实 G1/G2 和简历工程包；
2. 在独立分支和独立 D 组中实现 Context Budget Manager；
3. 先做 deterministic/prompted 版本和 16/32 QA 固定消融；
4. 行为门禁通过后，再决定是否把 action policy 交给 GRPO 学习；
5. 不改 baseline B/C 的预注册 primary，也不拿 D 组替代 B/C 主结论。

若必须新增一个“模型可调用的工具”，其语义至少要满足：

- 不采用与当前代码完全不同的 JSON 协议，继续使用可严格解析的标签；
- 明确目标对象是 working memory、recalled memory 还是 history，不能叫笼统的 context；
- 明确 hard token budget、触发阈值和压缩前后 token 数；
- 明确 update 与 compress 是互斥还是可组合，并对重复/空/冲突动作 fail closed；
- 保留 source document/chunk provenance，禁止压缩后产生无法追溯的新事实；
- 将 token 节省、证据保留和任务精度分开评测。

更稳妥的第一版不是让模型自由生成一个与 update 同义的 `<compress>content</compress>`，而是实现一个确定性的 Budget Manager：记录各 prompt block token 数，在达到阈值时按固定策略缩减指定 memory block，并把 action、前后 token、保留的 source IDs 和结果写入 trajectory。它可以证明 Agent harness 的预算管理能力，但简历上应写“实现 Context Budget Manager”，不能写“GRPO 学会调用 compress”，除非后者真的训练并评测过。

### 5.4 推荐顺序下的工具成功率

| 结果 | 估计概率 | 条件 |
|---|---:|---|
| 完成可演示、可测的工具扩展 | **60%-75%** | 条件于 baseline L1 已封存；若仅要求 G1/G2 后做 MVP，可更快但科学证据更弱 |
| 平均 context token 明显下降 | **65%-85%** | 条件于工具可用；这是机械目标，不等于回答更好 |
| 相同预算下基本保持精度 | **30%-50%** | 条件于工具可用；需要固定 QA 对照 |
| EM/F1 显著提高 | **5%-15%** | 条件于工具可用；不应作为项目硬目标 |
| baseline 与工具完整结果都拿到 | **17%-32%** | 从当前状态出发的组合概率 |

工具最可能产生的真实成果是“减少 token / 控制成本，同时精度下降可控”，而不是提高 EM。若简历目标是 Agent 架构岗，这种系统权衡同样有价值。

---

## 6. 推荐实施路线

### Phase 1：冻结 baseline，完成 5090/2B CPU 迁移（7/18-7/24）

- 将 5090/2B 候选方案提升为 active profile；
- 保留旧 4B/PRO6000 profile，不覆盖、不混用 state；
- 更新 handoff、cloud README 和 branch/commit 入口；
- 新增 2B G2、B/C20/40/60/80、eval40/eval80 与 R0/R1 overlays；
- 构建 2B tokenizer 绑定的独立 formal bundles；
- compose 并 exact-key 验证 33 份 resolved configs；
- 更新 hardware probe、config matrix 和 negative tests；
- **本阶段不做 compress。**

完成判据：CPU 测试、config compose、handoff、静态 audit 全绿，旧 4B state 不进入 2B active inventory。

### Phase 2：真实 5090 G0/G1（7/25-7/31）

- G0：0.8B kernel/BF16/20-step gate；
- G1：2B step1 -> 新进程 resume2；
- adapter-only export/reload；
- merged model reload；
- 固定 2-QA recurrent eval；
- 保存 GPU identity、VRAM、日志、checkpoint 和 hash evidence。

这是**第一个强简历 checkpoint**。到这里即使后续 G2 或长训练失败，也可以诚实写“完成 Qwen3.5-2B 单卡 Agentic RL 接口、恢复和模型产物闭环”，但不能写正式长上下文 B/C 结果。

### Phase 3：G2 capacity（8/1-8/7）

- 先跑 R0 G2a/G2b/length-stress；
- 只有显存/headroom/allocator 容量原因才允许显式切 R1；
- 封存 `capacity-profile.json`；
- 实测 `T_init/T_resume/T_compute/T_save/T_artifacts`；
- 计算 B/C40 + 评测的费用和时间投影。

完成严格 L0 后，简历项目已经具备较强的资源适配和工程证据。R0/R1 均失败时，不要静默降 dtype；直接把“32GB 容量边界和失败证据”作为项目结论。

### Phase 4：Pilots 与 B/C40（8/8-8/16）

- B/C 各固定 3-step pilot；
- 不换 prompt、seed 或 LoRA init；
- 不满足 `2/6` 非零 advantage 时发布 scientific-stop；
- 通过后运行 B20/C20，再由新进程恢复到 B40/C40；
- 两边同步决定，不只延长领先方。

### Phase 5：L1 评测和简历打包（8/17-8/23）

优先级高于 B/C80：

- 20 个唯一 eval cells、32 QA/格；
- Base/B/C 主表和 learned/none/fixed callback 消融；
- 配对 bootstrap、行为指标和资源 ledger；
- 至少一条成功 trajectory 和一条失败 trajectory；
- 一页架构图、README、known limitations、失败分析；
- verified package 的 hashes 和复现实操命令。

### Phase 6：L2 或项目收尾（8/24 以后）

- 面向科研复现且 C-B 已有可信趋势：可以选择 B/C80；
- 面向工程或 Agent 架构岗：优先整理状态机、恢复、trajectory、资源与失败证据；
- 当前主线不承担任何新增工具；只在预算允许时进入 L2，否则以完整 L1 收尾。

---

## 7. 明确止损点

| 时间/门禁 | 停止条件 | 收尾方式 |
|---|---|---|
| T+7 天 | 2B/5090 CPU config、数据或测试仍不能稳定通过 | 冻结范围，先修 baseline |
| G1 | 连续两次出现同一类失败 | 切已验证底座或缩短声明范围，不无限排查 |
| G2 | R0/R1 都失败 | 停止正式 B/C；发布容量边界，不偷改 dtype |
| 预算 | L1 投影超过 GPU 450 / 总额 500 元 | 不启动 B/C40；用 L0 工程包收尾 |
| Pilot | 固定样本未达到 `2/6` 非零 advantage | 发布 scientific-stop，不重抽 seed/样本 |
| 8 月 16 日 | 尚未得到 B/C40 | 停止追严格 L1，整理 G1/G2 + prompted/小评测简历包 |

---

## 8. 简历策略

### 8.1 现在或 CPU-only 阶段可以写什么

可以写实现内容，不写 GPU 和效果数字：

> 面向 ReMemR1 长上下文记忆 Agent，设计 Qwen3.5 text-only、LoRA/FSDP、recurrent HF rollout、严格 checkpoint/resume、数据与评测 provenance 及可恢复云端阶段 DAG，构建 update/recall 动作解析、状态奖励和 trajectory 评测闭环。

这里不能写“已在 RTX 5090 跑通”“准确率提升”或“峰值显存”。

### 8.2 完成真实 G1/G2 后可以写什么

> 在单张 RTX 5090 32GB 上完成 Qwen3.5-2B 长上下文 Memory Agent 的 LoRA-GRPO 接口与容量适配，验证新进程 checkpoint 恢复、adapter export/merge 和固定 recurrent QA；实测峰值显存、单步耗时与失败边界，并通过可恢复阶段流水线固化运行证据。

方括号数字只能填入 verified evidence 的实测值。

### 8.3 完成 L1 后可以写什么

> 在 HotpotQA / 2WikiMultiHopQA 的 200/800-document 设置上完成 outcome-only 与 multi-level reward 的 B40/C40 成对实验，以及 learned/none/fixed callback 消融；从 answer EM/F1、格式率、召回行为、平均回看距离、token 和成本等维度报告实际趋势与置信区间。

如果 C-B 为零或为负，就写“未观察到显著提升，并定位 2B/group4/单 seed 下的信号与容量边界”，不要把 secondary 指标替换成事后 primary。

### 8.4 未来独立工具项目的边界（不属于本次交付）

只有在真实对照完成后才写：

> 在 baseline Memory Agent 上增加可审计的 Context Budget Manager，记录压缩触发、前后 token、source provenance 和精度变化；在固定 QA 与相同上下文预算下，将平均输入 token 从 [实测] 降至 [实测]，回答指标变化为 [实测]。

如果只是 deterministic harness，就不要写“GRPO 学会何时压缩”；如果只做了 synthetic fixture，就必须明确标为测试/演示数据。

---

## 9. 关于“编数据”的边界

不建议，也不能把未发生的训练或评测结果编成实测数据。原因不仅是学术规范，更是现实的面试风险：这个仓库强调 commit、resolved config、manifest、seed、checkpoint、adapter、评测 cell 和 hash 身份，任何一个“提升 X%”都可能被追问原始 JSON、样本数、CI、失败 runs 和复现命令。

允许且推荐的做法：

- 在内部草稿使用 `X/Y/Z` 占位，但投递前删除或替换为实测；
- 把 `planned`、`estimated`、`measured` 分成不同列；
- synthetic fixture 明确标注为测试数据，不进入正式主表；
- 报告无提升、负提升、scientific-stop 或容量失败；
- 用固定公开样本展示 qualitative trajectory，并说明它是案例而非总体统计；
- 简历只声明实际达到的层级。

不能做的事情：

- 把论文数字当作自己的实验结果；
- 把 CPU unit test 写成 5090 已跑通；
- 把短 G1 fixture 写成正式 200/800-doc 评测；
- 挑 seed、checkpoint 或评测格制造有利结果；
- CI 跨 0 时写成“显著提升”；
- 将 projected VRAM/time/cost 写成 measured。

负结果并不等于没有成果。对工程与 Agent 岗位，能够证明“系统完整运行、实验控制变量正确、结果没有显著改善，并定位了动作信号、容量、成本和评测瓶颈”，通常比无法复核的漂亮数字更有说服力。

---

## 10. 最终建议

### 如果唯一目标是尽快得到一个简历项目

目标设为：**真实 G1 + 尽量完成 G2 + 小规模固定评测 + 完整工程包**。成功概率约 80%-90%。到这个层级就可以投递，不必等待 B/C40，更不必等待显著正向结果。

### 如果目标是得到一份完整实验报告

目标设为：**无新增工具的 baseline L1**。成功概率约 28%-42%。把负结果视为有效交付，优先完成评测、图表和 verified package，不追 L2。

### 如果目标是突出 Agent 架构

当前决策是不增加工具。若未来另行立项，应先封存 baseline 证据，再在独立 profile 中设计语义清晰、目标明确、可度量的 Context Budget Manager；不得回写或污染本次 B/C 主实验。

### 一句话决策

> 继续做；把无新增工具的 5090/2B baseline 做成能跑、能恢复、能评测、能解释的简历项目。项目成功不要求论文式正向结果，但所有数字必须来自真实、可追溯的 evidence。

---

## 11. 迁移前基线的关键代码证据索引

- 当前 5090 候选层级与声明边界：`docs/rtx5090_2b_reproduction_plan_zh.md:73`、`:102`、`:994`；
- 当前实现只到 G0/G1：`scripts/cloud/run_pipeline.sh:146`、`:714`；
- PRO6000/90GiB 硬门禁：`scripts/cloud/gpu_probe.py:170`、`scripts/cloud/run_stage.sh:357`；
- 当前 formal configs 仍为 4B：`verl/trainer/config/reproduction/b40_qwen35_4b.yaml:43`、`c40_qwen35_4b.yaml:43`；
- 2B 只已有 G1 configs：`verl/trainer/config/reproduction/g1_qwen35_2b_step1.yaml:41`；
- 中间动作必须有 update、recall 仅可选：`recurrent/protocol.py:72`、`:110`、`:214`；
- 固定逐 chunk 推进与自动 final：`recurrent/impls/memory_revisit.py:690`、`:702`、`:706`、`:878`；
- 当前 prompt 只注入 current chunk / working memory / recalled memory：`recurrent/impls/memory_revisit.py:740`、`:744`；
- `step` 同时参与 rollout 与 reward 分组：`recurrent/generation_manager.py:195`、`verl/trainer/ppo/ray_trainer.py:2153`；
- update 写入 working/history memory：`recurrent/impls/memory_revisit.py:855`、`:893`；
- state reward 当前只覆盖 memory/callback/format：`recurrent/rewards.py:97`、`:110`；
- 当前 memory reward 的 recall 方向：`recurrent/rewards.py:60`、`recurrent/protocol.py:354`；
- 当前 evaluator 直接复用同一 action contract：`taskutils/memory_eval/reproduction_runner.py:746`、`:785`、`:914`。
