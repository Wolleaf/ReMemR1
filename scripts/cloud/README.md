# ReMemR1 RTX 5090 / Qwen3.5-2B 云端复现

本目录是 `rtx5090-32g-qwen35-2b-v1` 的 AutoDL 操作入口。正常流程使用同一块持久卷，先由
CPU 实例完成联网准备和密封交接，再由单张 RTX 5090 实例分阶段执行门禁、容量、B/C40、可选
B/C80 与导出。每个 GPU 阶段都是独立的付费决策，不会由一条命令无门控地跑完整个实验。

代码交付不等于 GPU 实测。只有对应 immutable attempt、terminal evidence 和 verified package
存在时，才可声明已经达到 L0、L1 或 L2。本 profile 不增加 `compress_context`，不改 JSON action，
继续使用强制 `<update>`、可选 `<recall>` 和 final answer 的既有 Agent 协议。

> Guest 内执行 `shutdown` 不等于 AutoDL 控制面已经停止计费。每个阶段结束后都必须回到
> AutoDL 控制台确认实例已停止，并核对余额和账单。

## 最短操作路径

### 1. CPU：一次 bootstrap

公开仓库按当前交付分支启动：

```bash
bash <<'REMEMR1_BOOTSTRAP'
set -euo pipefail
bootstrap="$(mktemp /tmp/rememr1-bootstrap.XXXXXX)"
trap 'rm -f -- "${bootstrap}"' EXIT
curl --fail --show-error --silent --location --connect-timeout 30 --max-time 300 \
  --speed-limit 1024 --speed-time 60 --retry 3 --output "${bootstrap}" \
  https://raw.githubusercontent.com/Wolleaf/ReMemR1/reproduction/rtx5090-2b/scripts/cloud/bootstrap.sh
bash "${bootstrap}" --phase cpu --allow-guest-shutdown
REMEMR1_BOOTSTRAP
```

外层命令先完整下载脚本再执行，不使用 `curl | bash`。如果下载在仓库代码启动前失败，云脚本无法
替你关机，必须立即在控制台停止实例。命令返回只表示 detached launcher 已建立并取得锁，不表示
CPU preparation 成功。

更高保证的做法是从可信渠道取得 40 位 commit，并固定 bootstrap 与 checkout：

```bash
bash <<'REMEMR1_BOOTSTRAP'
set -euo pipefail
commit='<TRUSTED_40_HEX_COMMIT>'
bootstrap="$(mktemp /tmp/rememr1-bootstrap.XXXXXX)"
trap 'rm -f -- "${bootstrap}"' EXIT
curl --fail --show-error --silent --location --connect-timeout 30 --max-time 300 \
  --speed-limit 1024 --speed-time 60 --retry 3 --output "${bootstrap}" \
  "https://raw.githubusercontent.com/Wolleaf/ReMemR1/${commit}/scripts/cloud/bootstrap.sh"
bash "${bootstrap}" --expected-commit "${commit}" --phase cpu --allow-guest-shutdown
REMEMR1_BOOTSTRAP
```

### 2. GPU：先跑有界 G0/G1 门禁

CPU launcher 必须是 `state=success` 且 `exit-code=0`。确认 CPU 实例已经停止，再把同一 provider
volume 挂到满足下文硬门禁的 5090 实例，然后执行：

```bash
bash /root/autodl-tmp/ReMemR1/scripts/cloud/start_gpu_gates.sh
```

本入口只跑离线 kernel/BF16、G0 和 G1，不会进入正式长度或 B/C 长训练。

### 3. GPU：单独启动 R0 容量阶段

G0/G1 成功、控制台确认上一实例已停止后，新开一个符合 R0 RAM 门禁的实例：

```bash
bash /root/autodl-tmp/ReMemR1/scripts/cloud/start_gpu_2b_capacity.sh --offload-profile r0
```

R0 只在 G2a、fresh-process resume、length-stress、制品和资源证据全部满足预注册规则时发布
`capacity-profile.json`。R0 的黄色/OOM 证据不会在同一个 launcher 内自动切换 R1；见“R1 显式
批准”。可信容量停止使用 `exit-code=43`、`.capacity-stop` 和 `retryable=false`；schema、resume、
hash、NaN 或科学失败不具备 R1 资格。

### 4. GPU：费用批准后运行 B/C40

用 G2 实测时间、剩余评测时间、当前支出和租卡单价生成并验证 `bc40` 费用投影。投影必须位于
profile 持久根内，并同时满足 GPU 450 元、总额 500 元硬停止线：

输入 JSON 必须严格使用下列字段；所有秒数和金额都填实际证据，不得填规划值。`evidence` 至少
包含一份 G2 telemetry/capacity 或评测 probe 文件的绝对路径与当前 SHA-256：

```json
{
  "decision": "bc40",
  "disk_remaining_rmb": 0.0,
  "evidence": [
    {
      "path": "/absolute/path/to/measured-evidence.json",
      "sha256": "<64_HEX_SHA256>"
    }
  ],
  "gpu_cost_done_rmb": 0.0,
  "gpu_hourly_rate_rmb": 0.0,
  "measurements": {
    "evaluation_remaining_seconds": 0.0,
    "t_artifacts_seconds": 0.0,
    "t_compute_seconds": 0.0,
    "t_init_seconds": 0.0,
    "t_resume_seconds": 0.0,
    "t_save_seconds": 0.0
  },
  "non_gpu_cost_done_rmb": 0.0,
  "ops_reserve_rmb": 50.0,
  "schema_version": 1
}
```

```bash
/root/autodl-tmp/rememr1/profiles/rtx5090-32g-qwen35-2b-v1/envs/reproduction-cu130/bin/python \
  /root/autodl-tmp/ReMemR1/scripts/cloud/cost_gate.py project \
  --input <ABSOLUTE_BC40_COST_INPUT_JSON> \
  --output <ABSOLUTE_BC40_PROJECTION_JSON>
```

工具返回 42 表示一份有效的“超预算停止”证据，不得重试到通过。只有返回 0 后才启动：

```bash
bash /root/autodl-tmp/ReMemR1/scripts/cloud/start_gpu_2b_bc40.sh --budget-projection <ABSOLUTE_BC40_PROJECTION_JSON>
```

该阶段固定执行 B/C pilots、pilot scientific gate、B20/C20、从 step20 恢复到 B40/C40、20-cell
L1 评测和 verified package。Pilot 达不到双方各 `2/6` 个 nonzero-advantage groups 时返回 42，
这是有效科学停止，不允许换 seed、样本或重复重抽。

评测绑定前，B/C endpoint 必须各自产生 fresh-process、deserialize-only 的 full-state resume probe；
cell 复用与 verified package 复验会从 `raw_final_output + gold_answers` 独立重算答案指标、summary 和
10,000 次 bootstrap aggregate，不信任可被重新封装的预汇总分数或仅有自哈希的 package。

### 5. 可选：B/C80 与导出

L1 成功后，用实际剩余调用数重新生成 `bc80` 投影，再单独启动：

```bash
bash /root/autodl-tmp/ReMemR1/scripts/cloud/start_gpu_2b_bc80.sh --budget-projection <ABSOLUTE_BC80_PROJECTION_JSON>
```

无论最终停在 L1 还是 L2，都可在独立 launcher 中重新验证并导出最新结果：

```bash
bash /root/autodl-tmp/ReMemR1/scripts/cloud/export_2b_results.sh
```

## 实例硬门禁

- CPU 和 GPU 实例必须挂载同一个 provider volume，路径固定为 `/root/autodl-tmp`，且任何时刻只允许一个 host 写入。
- 系统为 Ubuntu 22.04、root、非 WSL；`/root/autodl-tmp` 必须是独立持久挂载，不能是 `/`、overlay、tmpfs、ramfs 或 squashfs。
- CPU preparation 至少 48 GiB RAM；CPU/GPU host 至少 24 cores；active profile 持久盘至少 200 GiB 可用。
- GPU 必须恰好一张 NVIDIA GeForce RTX 5090，可见显存至少 31 GiB，启动空闲显存至少 29 GiB，无其它 compute process。
- GPU compute capability 固定 `sm_120`；CUDA runtime/toolkit 固定 13.0；镜像为 12.8 时必须停止并重新封环境，不能视为等价。
- R0 host RAM 至少 96 GiB；R1 至少 128 GiB。R1 还要求完整 R0 容量证据、一次性审批和 B/C40 费用投影。
- CPU 阶段需要 GitHub、PyPI、PyTorch wheel index 和 Hugging Face 网络；GPU 阶段强制 Hugging Face、Transformers、Datasets 离线。
- 仓库当前使用公开资源，不把 token 写入 `rememr1-cloud.env`、日志、配置或结果包。

## CPU 阶段封存内容

CPU 流水线固定 clean detached checkout，并完成：

1. 创建持久 Python 3.12.2 环境，安装精确依赖，执行 `pip check`，保存并哈希 freeze；
2. 预取固定 commit 的 FLA 和 causal-conv1d 源码，供 GPU 离线编译；
3. 下载并逐文件校验 Qwen3.5-0.8B、Qwen3.5-2B、HotpotQA、2WikiMultiHopQA 和 Byted 正式数据；active manifest 不含 4B；
4. 生成 gate、正式 train/validation/eval 和独立 non-scientific length-stress bundles，全部绑定 2B tokenizer；
5. 执行 reproduction/cloud tests、`compileall`，真实 compose 并 exact-key 校验 33 份 resolved configs；
6. 发布 schema v3、自哈希、原子写入的 `cpu-handoff.json`，绑定 commit、profile、环境、资产、数据、kernel sources 和配置树。

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
    固定 commit、clean、detached checkout
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
bash /root/autodl-tmp/ReMemR1/scripts/cloud/start_cpu_prep.sh --retry-failed-stage
bash /root/autodl-tmp/ReMemR1/scripts/cloud/start_gpu_gates.sh --retry-failed-stage
bash /root/autodl-tmp/ReMemR1/scripts/cloud/start_gpu_2b_capacity.sh --offload-profile r0 --retry-failed-stage
bash /root/autodl-tmp/ReMemR1/scripts/cloud/start_gpu_2b_bc40.sh --budget-projection <ABSOLUTE_BC40_PROJECTION_JSON> --retry-failed-stage
```

R1 的一次性 marker 不可重试；基础设施失败后必须重新核算投影并生成新 nonce。B/C40/B/C80 的新增
支出也必须生成新投影和新 generation，旧 projection-bound stage 不会复用。CUDA OOM、NaN、
schema/hash mismatch、resume drift 和 scientific-stop 不会自动重试。只有“完整训练制品已原子发布，但后续 cleanup/terminal
publication 失败”才可能由严格 verifier 生成新的 synthetic adoption success；原失败 attempt 保留。

排障时保留机器：

```bash
bash /root/autodl-tmp/ReMemR1/scripts/cloud/start_gpu_gates.sh --retry-failed-stage --keep-running
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
