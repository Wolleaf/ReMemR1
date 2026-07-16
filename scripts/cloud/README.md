# ReMemR1 云端一键复现

这套脚本面向 AutoDL 的“CPU 实例准备持久卷，再切换到单卡 GPU 实例”流程。正常使用时，每台实例只执行一条命令；环境安装、资产下载、数据构建、测试、GPU 门禁、日志、退出码和关机都由后台流水线完成。

脚本默认采用严格策略：后台任务一旦获得全局锁，无论成功还是失败，都会先把终态和原始退出码同步到持久卷，然后请求 guest 关机。显式传 `--keep-running` 或 `--dry-run` 时不关机；锁冲突、终态无法持久化、同步失败或关机授权复验失败时也会保持运行，避免在证据未落盘时关机。

> Guest 内执行 `shutdown` 不等于 AutoDL 控制面一定已经停止计费。每次运行后都必须回到 AutoDL 控制台，确认实例状态为“已关机/已停止”。

## 前提

- CPU 和 GPU 实例挂载同一个持久卷，并保持 `/root/autodl-tmp` 路径不变；不要同时启动两个写入该卷的实例。
- 使用 Ubuntu 22.04、root 用户和真实的 `/root/autodl-tmp` 独立挂载；脚本明确拒绝 WSL 和落在系统盘 `/` 上的伪持久目录。
- CPU 实例至少需要 48 GiB RAM，持久卷启动时至少需要 80 GiB 可用空间；这是脚本的硬门槛，不满足会在下载前失败。
- CPU 镜像必须预装 conda（或通过 `REMEMR1_CONDA_BIN` 指定）；脚本会创建持久 Python 3.12.2 env，不会静默改用系统 Python。
- CPU 阶段需要访问 GitHub、PyPI、PyTorch wheel index 和 Hugging Face；GPU 阶段强制 Hugging Face/Transformers/Datasets 离线。
- GPU 阶段固定要求一张 RTX PRO 6000 96GB、compute capability 12.0、CUDA runtime/toolkit 13.0。
- 仓库和数据集当前均为公开资源，不需要持久化 token。若以后需要 `HF_TOKEN`，只在启动 shell 的环境中提供；`rememr1-cloud.env` 永远不保存 token。

## 第一步：CPU 实例一条命令

公开仓库的直接启动命令如下。它会把分支 tip 解析一次为完整 SHA、以 detached HEAD 拉取，然后把这个 SHA 固定到整个 CPU→GPU handoff：

```bash
bash <<'REMEMR1_BOOTSTRAP'
set -euo pipefail
bootstrap="$(mktemp /tmp/rememr1-bootstrap.XXXXXX)"
trap 'rm -f -- "${bootstrap}"' EXIT
curl --fail --show-error --silent --location --connect-timeout 30 --max-time 300 \
  --speed-limit 1024 --speed-time 60 --retry 3 --output "${bootstrap}" \
  https://raw.githubusercontent.com/Wolleaf/ReMemR1/reproduction/qwen35-plan/scripts/cloud/bootstrap.sh
bash "${bootstrap}" --phase cpu --allow-guest-shutdown
REMEMR1_BOOTSTRAP
```

外层 wrapper 会先把 bootstrap 完整下载到临时文件，只有 `curl` 成功后才执行，避免流式下载中断时运行半截脚本。下载失败会传播成非零退出；此时仓库脚本还没有开始运行，无法代替你关机，必须立即在控制台停止实例。`--allow-guest-shutdown` 是操作者对这次持久卷的显式授权，不是 AutoDL 提供商 attestation。脚本仍会复验 Linux/root/mount/capability/commit/flock，但无法从 guest 内密码学证明它一定运行在 AutoDL；不要在本地 root Linux 传入这个参数。

对供应链身份有更严格要求时，从可信渠道取得 40 位 commit 后，用 commit 固定 bootstrap 脚本和 checkout。整段仍然只执行一次：

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

完成 branch resolve、首次 clone/fetch、初始化和 launcher 锁握手后命令才返回；已有 checkout 时通常很快，首次 clone 可能需要数分钟。输出例如：

```text
launcher_dir=/root/autodl-tmp/rememr1/cloud/launchers/20260716T120000Z-cpu-ab12cd34ef56-1234-5678-1
launcher_pid=4321
launcher_log=/root/autodl-tmp/rememr1/cloud/launchers/<LAUNCHER>/launcher.log
```

返回只表示后台 launcher 已建立并取得写锁，不表示 CPU 阶段已经成功。无需继续粘贴安装或下载命令。可选地查看日志：

```bash
bash /root/autodl-tmp/ReMemR1/scripts/cloud/status.sh
```

```bash
tail -f /root/autodl-tmp/rememr1/cloud/launchers/<LAUNCHER>/launcher.log
```

CPU 流水线依次完成：

1. 验证无 GPU、固定 commit、clean checkout、持久卷和基础工具；
2. 创建持久 Python 3.12.2 环境，安装精确依赖，执行 `pip check`，保存并哈希 CPU freeze；
3. 在 CPU 实例拉取两个固定 commit 的 CUDA kernel 源码，供 GPU 离线编译；
4. 动态解析资产清单中尚未发布的正式 Parquet LFS size/SHA256，生成仓库外 runtime manifest；
5. 下载并逐文件验证 0.8B/2B/4B 模型、HotpotQA、2Wiki 和正式训练源；
6. 生成 G0/G1 的确定性 20-QA 训练 fixture、G1 专用 2-QA recurrent-eval fixture，以及正式 train/validation/eval bundles；
7. 执行 `tests/reproduction`、`tests/cloud`、`compileall`，并真实 compose/resolve 13 份 Hydra 配置；
8. 发布自哈希 `cpu-handoff.json`，记录 commit、lock、资产、依赖、kernel source、配置树和全部数据 manifest SHA。

正式数据始终 fail closed。若 Hugging Face 仍无法返回 Byted 数据的 LFS SHA，或旧 Parquet 没有 builder 要求的结构化 documents/supporting facts，CPU 阶段会保留错误日志、写非零退出码并关机；不会用未验证下载、平铺 context 或占位 SHA 继续训练。

CPU 成功和失败都会关机，不能只凭控制台显示 stopped 判断结果。先对启动命令打印的 launcher 路径执行：

```bash
bash /root/autodl-tmp/ReMemR1/scripts/cloud/status.sh <CPU_LAUNCHER_DIR>
```

只有输出同时包含 `state=success` 和 `exit-code=0`，才能在 AutoDL 控制台确认实例已停止，再把同一数据盘切换到 GPU 实例。否则先按“失败恢复”处理。不要移动或重新 clone 仓库；GPU handoff 同时绑定绝对路径和 commit。

## 第二步：GPU 门禁一条命令

GPU 实例启动后执行：

```bash
bash /root/autodl-tmp/ReMemR1/scripts/cloud/start_gpu_gates.sh
```

GPU 流水线在任何 CUDA 编译或权重加载前重新验证 CPU handoff；随后全程离线、严格串行执行：

1. 从 CPU 预取的 git 对象离线编译固定 FLA/causal-conv1d commit；
2. 在 sm_120 上分别生成 BF16 forward、backward 和 20 次 optimizer-loop 证据，封装并复验 `build-info.json`；
3. G0：Qwen3.5-0.8B，20 次真实 optimizer step；
4. G1 step 1：Qwen3.5-2B 保存完整可恢复 checkpoint；
5. 停止旧 Ray 进程后，用新 Python 进程从 step 1 显式恢复到 step 2；
6. 校验 adapter-only export，用 2B tokenizer 绑定的 2-QA fixture 真实 reload PEFT adapter，并通过 HF Transformers recurrent evaluator 完成两个样本；
7. merge adapter、reload merged model，对固定 prompt 比较 adapter/merged 的 logits 与 greedy generation token。

G1 eval 明确是接口门禁而非正式科学评测：它固定 `profile=fixture`、2 个样本、16 documents、`chunk_size=1024`，并强制每条记录恰好 2 个 recurrent chunks；不会冒充正式 runner 的 32/64 样本结果。任一 stage 非零都会停止后续 stage。G-1/G0/G1 全部通过前，不会进入 G2 或 B/C 长训练。本入口有意停在有证据的 bounded gate；长训练不会因为一次启动命令被意外触发并持续计费。

## 失败恢复

launcher 目录保存：

```text
launcher.log       后台完整日志
launcher-pid       launcher 进程号
lock-acquired      已取得全局持久卷写锁
pipeline-result    对应 pipeline 状态目录
.success/.failed   launcher 终态
exit-code          原始 pipeline 退出码
shutdown-safe      退出码和终态已同步，允许请求关机
shutdown-failed    guest 关机命令失败（若存在）
```

pipeline 目录另有每个 stage 的独立 attempt、日志、成功/失败 marker 和前驱记录。重跑不会覆盖旧 attempt；已经成功且重新验证通过的 stage 会复用。

若 G0/G1 训练已经完整发布 checkpoint 和 adapter，但随后 Ray 清理或 stage 状态同步失败，下一次 `--retry-failed-stage` 会先重新执行 GPU preflight、清理遗留 Ray，再严格核对 checkpoint、adapter、base revision、train/validation manifest 和 resume 前驱。全部一致时脚本发布一个新的 adoption success run 并继续，不会重训；原 failed run、日志和原始退出码保持不变。CPU stage、GPU preflight、部分训练产物或未通过科学/接口门禁的产物绝不会被这种机制收养。

普通失败、超时、OOM 或中断后，先查看最新状态和对应 `failed-stage`，确认原因已经修复，再只执行一条恢复命令：

```bash
bash /root/autodl-tmp/ReMemR1/scripts/cloud/start_cpu_prep.sh --retry-failed-stage
```

或：

```bash
bash /root/autodl-tmp/ReMemR1/scripts/cloud/start_gpu_gates.sh --retry-failed-stage
```

训练、OOM、NaN、schema/hash mismatch 和科学门禁不会自动重试。只有 Hugging Face 单文件下载在阶段内部做有限次数、最终受 SHA 约束的重试。

若要保留机器现场，不自动关机：

```bash
bash /root/autodl-tmp/ReMemR1/scripts/cloud/start_gpu_gates.sh --retry-failed-stage --keep-running
```

排障结束后由操作者在控制台关机。`--keep-running` 不改变执行顺序、日志、marker 或退出码。

## 关机安全边界

自动关机不是一个普通 EXIT trap。顺序固定为：

```text
stage/pipeline 终态落盘
        -> launcher 写日志 sentinel、terminal、exit-code 和终态 marker
        -> sync 成功
        -> 发布 shutdown-safe
        -> 再次验证 Linux/root/mount/commit/capability/flock
        -> 请求 guest shutdown
```

只有同时满足 Linux 非 WSL、root、真实持久挂载、固定 clean checkout、root-owned `0600` capability、正确全局锁仍由 launcher 持有、状态文件位于持久根等条件，才可能执行绝对路径下的 `shutdown`/`systemctl`。以下情况故意不自动关机：

- 命令行参数错误，以及 bootstrap 尚未取得持久锁时的失败；
- 全局锁已被另一任务占用；
- 终态、日志或 `sync` 无法可靠写入持久卷；
- capability、commit、路径、mount 或锁身份复验失败；
- `--keep-running` 或 `--dry-run`。

bootstrap 在取得持久锁后若 clone/fetch/init/launcher 建立失败，也会先写自己的日志和退出码；只有退出时的 mount/lock 复验（初始化后还包括 capability/commit/clean checkout 复验）通过，才按显式授权请求关机。日志位于 `/root/autodl-tmp/rememr1/cloud/bootstrap/`。原始 `curl` 下载尚未启动脚本、参数校验尚未建立持久状态、锁冲突或状态无法落盘时则不会冒险关机。查看最近一次 bootstrap 失败可执行：

```bash
run="$(find /root/autodl-tmp/rememr1/cloud/bootstrap -mindepth 1 -maxdepth 1 -type d -printf '%T@ %p\n' | sort -nr | head -n 1 | cut -d' ' -f2-)"
printf 'bootstrap_run=%s\n' "${run}"
cat "${run}/terminal" "${run}/exit-code"
tail -n 100 "${run}/bootstrap.log"
```

`SIGKILL`、宿主掉电或 OOM 同时杀掉 launcher 与子进程时，guest 内任何脚本都无法保证关机，因此仍需配置 AutoDL 自身的最长运行时限/余额告警，并在控制台确认 stopped。

## 持久目录

```text
/root/autodl-tmp/ReMemR1                    固定、clean、detached checkout
/root/autodl-tmp/rememr1-cloud.env          root:0600，路径与 commit（无 token）
/root/autodl-tmp/rememr1/envs                Python 环境
/root/autodl-tmp/rememr1/cache               HF/PyTorch/pip cache
/root/autodl-tmp/rememr1/sources             固定 kernel git 对象
/root/autodl-tmp/rememr1/data                gate/formal bundles
/root/autodl-tmp/rememr1/cloud/launchers     每次启动的日志和退出码
/root/autodl-tmp/rememr1/cloud/pipelines     可恢复 DAG、handoff、checkpoint 和证据
```

不要手工编辑 `cpu-handoff.json`、pipeline identity、stage marker 或仓库内 YAML 的 0/1/2/3 占位 SHA。CPU 阶段生成的真实 SHA 只通过 runtime overrides 和 immutable resolved snapshots 注入，checkout 必须始终保持 clean。
