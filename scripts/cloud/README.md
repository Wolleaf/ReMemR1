# ReMemR1 RTX 5090 / Qwen3.5-2B 云端复现

本目录是 `rtx5090-32g-qwen35-2b-v1` 的 AutoDL 操作入口。用户自行完成 Git checkout，
仓库就绪后正常操作只有两条命令：一条做无卡 CPU 准备，一条做 RTX 5090 GPU 训练。
内部仍保留可恢复 stage、持久日志、原始退出码和 CPU-to-GPU handoff，但不再要求用户手动选择
G0/G1/G2 脚本。

代码交付不等于 GPU 实测。只有对应 immutable attempt、terminal evidence 和 verified package
存在时，才可声明已经达到 L0、L1 或 L2。本 profile 不增加 `compress_context`，不改 JSON action，
继续使用强制 `<update>`、可选 `<recall>` 和 final answer 的既有 Agent 协议。

> Guest 内执行 `shutdown` 不等于 AutoDL 控制面已经停止计费。每个阶段结束后都必须回到
> AutoDL 控制台确认实例已停止，并核对余额和账单。

## 正常流程：Git + 两条命令

### 0. Git 由用户处理

把目标 commit 完整 checkout 到 `/root/autodl-tmp/ReMemR1`，并确保 worktree clean。CPU 命令会把
当前 40 位 `HEAD` 固定到 CPU-to-GPU 交接；CPU 准备开始后，到 GPU 训练结束前不要再 `pull`、
switch branch 或修改仓库文件。两个入口都不执行 clone、fetch、pull 或 checkout。

### 1. CPU 准备

GPU 不开启、AutoDL 临时 shell 只有 `0.5 core / 2 GiB` 时执行：

```bash
bash /root/autodl-tmp/ReMemR1/scripts/cloud/prepare_cpu.sh
```

本入口自动初始化持久目录，强制隐藏 CUDA，然后在后台串行创建 Python 3.12.2 隔离环境、
安装锁定依赖、下载模型/数据资产并预取 GPU kernel 源码。低资源模式不在 2 GiB RAM 中强行物化
32k 正式数据；成功后发布 `.cpu-env-ready` 并尝试关机。

### 2. GPU 训练

确认 CPU 阶段成功且挂载同一块 provider volume，启动这台固定 RTX 5090 后只执行：

```bash
bash /root/autodl-tmp/ReMemR1/scripts/cloud/run_gpu.sh
```

一个 detached launcher 会在同一全局锁下顺序完成：离线构建正式数据/配置并密封 CPU handoff，
编译 `sm_120` kernel，运行 G0/G1 真实训练与恢复门禁，再以固定 R0 执行 G2a/G2b/length-stress 并
发布 L0 capacity evidence。任一内部阶段失败都保留原始退出码和可恢复记录，不自动切 R1、不自动进入
B/C40 或 B/C80 长训练。这条命令的目标是交付可写简历的 L0 工程闭环。

两条命令都只会等到 launcher 取得锁并打印 `launcher_dir=...`，长任务由 `nohup + setsid`
后台执行。如果普通基础设施失败，检查日志后对同一入口增加 `--retry-failed-stage`；科学停止 42
或容量停止 43 不得重试到“通过”。

### 可选机器清单

需要重新记录主机时，可运行只读的 `collect_host_inventory.sh`。它不安装软件、不改设置、不关机。
当前固定机器的稳定规格摘要位于 `docs/rememr1_fixed_autodl_host_profile_zh.md`。原始本地清单不公开，
代码也不把短期 container ID 或 GPU UUID 当成可更换的资源门槛。

### 高级科学实验

B/C40、B/C80、费用投影、R1 一次性批准和结果导出仍保留在内部分阶段入口中，但不属于上面
两条正常命令，也不会被 `run_gpu.sh` 自动触发。只有完成 L0 并决定追求正式 L1/L2 结果时才需要使用。

## 实例硬门禁

- CPU 和 GPU 实例必须挂载同一个 provider volume，路径固定为 `/root/autodl-tmp`，且任何时刻只允许一个 host 写入。
- 系统为 Ubuntu 22.04、root、非 WSL；`/root/autodl-tmp` 必须是独立持久挂载，不能是 `/`、overlay、tmpfs、ramfs 或 squashfs。
- 无卡 CPU 准备按实时 cgroup 配额最低支持 `0.5 core / 2 GiB RAM`，并强制单线程/低并发；
  active profile 持久盘初始至少 200 GiB 可用。GPU 内部 `cpu-finalize`、gates、capacity 会按
  剩余不可变产物预算分别要求至少 128/128/80 GiB，而不是错误地每次重要求 200 GiB。
- GPU 必须恰好一张 NVIDIA GeForce RTX 5090，可见显存至少 31 GiB，启动空闲显存至少 29 GiB，无其它 compute process。
- GPU compute capability 固定 `sm_120`；CUDA runtime/toolkit 固定 13.0；镜像为 12.8 时必须停止并重新封环境，不能视为等价。
- 已记录机器的 GPU 态有效配额为 `16 cores / 90 GiB RAM`；R0 门禁为 80 GiB，因此可准入，实际峰值仍须由 G2 证明。该机器不满足 R1 的 128 GiB，正常 GPU 入口固定 R0 且不提供自动 R1。
- CPU 阶段需要 GitHub、PyPI、PyTorch wheel index 和 Hugging Face 网络；GPU 阶段强制 Hugging Face、Transformers、Datasets 离线。
- 仓库当前使用公开资源，不把 token 写入 `rememr1-cloud.env`、日志、配置或结果包。

## CPU 准备与 GPU 交接

为了在 AutoDL 无卡 `0.5 core / 2 GiB` 临时配额内可靠运行，准备工作按内存边界分开：

1. 公开 CPU 命令创建持久 Python 3.12.2 环境，安装精确依赖，执行 `pip check`，保存并哈希 freeze；
2. CPU 命令预取固定 commit 的 FLA/causal-conv1d 源码，下载并逐文件校验 0.8B、2B 模型与所需数据，然后发布 `.cpu-env-ready`；
3. GPU 公开命令的第一个内部步骤仍强制隐藏 CUDA 并开启离线模式，使用 GPU 实例的 16 CPU/90 GiB RAM 生成 gate、正式 train/validation/eval 和 length-stress bundles；
4. 该离线交接步骤执行 reproduction/cloud tests、`compileall`，compose 并 exact-key 校验 33 份 resolved configs；
5. 在任何 CUDA 编译或训练前发布 schema v3、自哈希、原子写入的 `cpu-handoff.json`，绑定 commit、profile、环境、资产、数据、kernel sources 和配置树。

任何资产 revision、LFS digest、数据 schema、配置 inventory 或路径不完整都会 fail closed，不会用占位
SHA、未验证缓存或 4B 历史资产继续。

## GPU 门禁与容量语义

GPU admission 在任何 CUDA 编译和权重加载前重验完整 CPU handoff，然后设置 offline 环境。G0 使用
0.8B 跑 20 个真实 optimizer steps；G1 使用 2B 执行 step1、由新 Python 进程恢复到 step2、adapter
export/reload、固定 2-QA recurrent eval、merge/reload 与 logits/generation 一致性验证。

容量阶段预注册两个 profile：

| Profile | Actor offload | Optimizer offload | Reference offload | 用途 |
|---|---:|---:|---:|---|
| R0 | false | false | false | 首选 |
| R1 | false | false | true | 仅 R0 容量原因的显式 fallback |

R0/R1 的 launcher、stage state、attempt、checkpoint 和日志按 phase/profile 隔离，互不 resume 或覆盖。
Length-stress 强制截断，只能贡献 VRAM/RAM/allocator 等资源峰值，不得写入 GRPO、格式或科学结论。
进程退出 0 也不自动等于数值或科学成功；所有 finite-loss/gradient、格式、advantage 和产物规则仍需
单独通过。只有 Python 能捕获的真实 PyTorch CUDA OOM 才能签发 `capacity-stop.json`；宿主 OOM、
SIGKILL 或整机掉电没有足够证据时保持 fail closed，不能据此批准 R1。

### R1 显式批准

当前固定机器只有 90 GiB 有效 RAM，因此下述 R1 流程在本机上不可用，也不会由正常
`run_gpu.sh` 触发。本节仅保留为未来换用 `>=128 GiB` 实例时的高级研究入口。

只有 R0 发布了完整、canonical、self-hashed 且标记为 R1-eligible 的容量证据，操作者检查后才能
生成 R1 target identity 和一次性 marker。典型命令如下，所有输入必须使用实际绝对路径：

```bash
python=/root/autodl-tmp/rememr1/profiles/rtx5090-32g-qwen35-2b-v1/envs/reproduction-cu130/bin/python
repo=/root/autodl-tmp/ReMemR1

"${python}" "${repo}/scripts/cloud/capacity_aggregate.py" target-identity \
  --handoff <ABSOLUTE_CPU_HANDOFF_JSON> \
  --index <ABSOLUTE_RESOLVED_CONFIG_INDEX_JSON> \
  --gpu-evidence <ABSOLUTE_R0_GPU_EVIDENCE_JSON> \
  --profile r1 \
  --output <ABSOLUTE_R1_TARGET_IDENTITY_JSON>

"${python}" "${repo}/scripts/cloud/capacity_evidence.py" approve-r1 \
  --r1-identity <ABSOLUTE_R1_TARGET_IDENTITY_JSON> \
  --r0-capacity-evidence <ABSOLUTE_R0_CAPACITY_EVIDENCE_JSON> \
  --r0-terminal-sha256 <R0_TERMINAL_SHA256> \
  --budget-projection-sha256 <BC40_PROJECTION_SELF_SHA256> \
  --approval-nonce <UNIQUE_SAFE_NONCE> \
  --output <ABSOLUTE_R1_APPROVAL_JSON>
```

然后新建 R1 GPU launcher；不得在原 R0 launcher 内继续：

```bash
bash /root/autodl-tmp/ReMemR1/scripts/cloud/start_gpu_2b_capacity.sh \
  --offload-profile r1 \
  --r1-approval <ABSOLUTE_R1_APPROVAL_JSON> \
  --budget-projection <ABSOLUTE_BC40_PROJECTION_JSON>
```

Marker 绑定 commit、handoff、R0 evidence/terminal、目标 R1 config set、GPU identity、费用投影和
唯一 nonce。持锁 pipeline 会在任何 R1 stage 前原子发布 consumption record；即使随后崩溃，已消费、
篡改或其它 pipeline 的 marker 也不能复用，必须重新核算费用并使用新 nonce。

## 状态、日志与成功判定

启动命令会打印 `launcher_dir`、PID 和日志路径。查看最新 launcher：

```bash
bash /root/autodl-tmp/ReMemR1/scripts/cloud/status.sh
```

查看指定 launcher：

```bash
bash /root/autodl-tmp/ReMemR1/scripts/cloud/status.sh <ABSOLUTE_LAUNCHER_DIR>
tail -f <ABSOLUTE_LAUNCHER_DIR>/launcher.log
```

Launcher admission 不是阶段成功。只有原始 `exit-code=0`、匹配的 `.success`、无 `.running`，且输出
重新验证通过时才算成功。`exit-code=42`、`.scientific-stop` 和 `retryable=false` 表示有效的非重试
科学停止；`exit-code=43`、`.capacity-stop` 和 `retryable=false` 表示有可信证据的容量停止。普通
`.failed` 必须先检查 `failed-stage`、run log 和 terminal evidence。手工运行 `cost_gate.py project`
时返回 42 则单独表示有效的超预算停止证据。

Profile 状态位于：

```text
/root/autodl-tmp/ReMemR1
    用户完成的 clean checkout；CPU 命令固定其 40 位 HEAD
/root/autodl-tmp/rememr1-cloud.env
    root:0600，保存非秘密路径与 commit
/root/autodl-tmp/rememr1/profiles/rtx5090-32g-qwen35-2b-v1/
    envs/       Python 环境
    cache/      HF、PyTorch、pip cache
    sources/    固定 kernel git objects
    data/       gate、formal、eval、length-stress bundles
    evidence/   freeze、GPU 与 build evidence
    cloud/      locks、launchers、phase/profile pipelines、immutable runs
    outputs/    verified packages 与 exports
```

不要手工编辑 handoff、resolved config、capacity profile、stage pointer、terminal、费用投影或评测
binding。流水线使用 canonical JSON、自哈希和 create-if-absent/原子发布，手改会在下一次验证中失败。

## 失败恢复

普通基础设施失败或中断只能在操作者检查并修复原因后显式重试。旧 attempt 永不改写：

```bash
bash /root/autodl-tmp/ReMemR1/scripts/cloud/prepare_cpu.sh --retry-failed-stage
bash /root/autodl-tmp/ReMemR1/scripts/cloud/run_gpu.sh --retry-failed-stage
```

R1 的一次性 marker 不可重试；基础设施失败后必须重新核算投影并生成新 nonce。B/C40/B/C80 的新增
支出也必须生成新投影和新 generation，旧 projection-bound stage 不会复用。CUDA OOM、NaN、
schema/hash mismatch、resume drift 和 scientific-stop 不会自动重试。只有“完整训练制品已原子发布，但后续 cleanup/terminal
publication 失败”才可能由严格 verifier 生成新的 synthetic adoption success；原失败 attempt 保留。

排障时保留机器：

```bash
bash /root/autodl-tmp/ReMemR1/scripts/cloud/run_gpu.sh --retry-failed-stage --keep-running
```

`--dry-run` 只验证并打印阶段顺序；`--keep-running` 和 `--dry-run` 都不会自动关机。测试模式永不调用
真实 shutdown backend，也拒绝使用生产持久路径。

## 关机安全边界

自动关机顺序固定为：stage/pipeline 终态 -> launcher log sentinel/terminal/original exit code -> durable
sync -> `shutdown-safe` -> 复验 Linux/root/mount/path/commit/capability/flock/state ->
`shutdown-requested` -> 绝对路径 shutdown backend。

以下情况故意保持实例运行：参数校验失败、锁冲突、终态或日志无法持久化、sync 失败、mount/path/
symlink/capability/commit/flock 复验失败、test mode、`--keep-running` 或 `--dry-run`。SIGKILL、宿主掉电
或整机 OOM 无法由 guest 脚本补救，必须同时设置 AutoDL 最长运行时和余额告警。

## 结果与简历口径

- 只有代码与 CPU preparation：可写 5090/2B 可恢复复现框架，不能写“已在 5090 跑通”。
- L0：真实 G0/G1/G2 全部通过，可写 5090 容量、恢复和制品闭环，并引用实测峰值/耗时。
- L1：双方 pilot 与 B/C40、32 QA/格和 verified package 完成，可写成对实验；C-B 为零或负仍是有效结果。
- L2：双方同步续训到 B/C80、64 QA/格和最终包完成，才可写 L2 数字。
- 不得把规划显存、预计费用、CPU fixture、论文指标或 synthetic data 写成实测；不得筛 seed、checkpoint 或评测格制造有利结果。
