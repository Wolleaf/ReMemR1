# ReMemR1 固定 AutoDL 主机规格

> 采集日期：2026-07-17
> 原始本地清单：`docs/rememr1-host-inventory.txt`（不纳入公开仓库）

本文只记录稳定资源规格。container ID、GPU UUID、当前已用内存和瞬时空闲显存不作为
固定配置，这些动态值仍由 GPU 阶段在启动时重新检查。

## 稳定规格

| 项目 | 记录值 |
|---|---|
| 系统 | Ubuntu 22.04.5 LTS，Linux 5.15，x86_64 |
| CPU | Intel Xeon Gold 6459C；GPU 开启态 cgroup quota 为 16 cores |
| CPU 无卡临时配额 | 0.5 core / 2 GiB RAM（由操作者确认） |
| GPU 开启态 RAM | cgroup v2 `memory.max=96636764160`，即 90 GiB |
| Swap | cgroup swap max/current 均为 0 |
| 持久盘 | `/root/autodl-tmp`，XFS，250 GiB，采集时基本空闲 |
| GPU | 1 x NVIDIA GeForce RTX 5090，32607 MiB VRAM |
| Compute capability | 12.0 (`sm_120`) |
| Driver | 580.105.08 |
| CUDA toolkit | 13.0，nvcc 13.0.88 |
| 基础镜像 | Python 3.12.3，torch 2.12.1+cu130 |
| 项目隔离环境 | CPU 准备另建 Python 3.12.2 锁定环境，不直接复用基础镜像 torch |

该 250 GiB 盘按“CPU 初始 200 GiB、GPU `cpu-finalize`/gates 128 GiB、R0 capacity 80 GiB”
的阶段剩余空间门槛使用；这些值是不可变产物预算，不是设备身份校验。

## 固定运行决策

- Git clone/fetch/checkout 由操作者完成；CPU 命令固定当前 clean `HEAD`。
- CPU 无卡阶段只做低内存可完成的环境、源码和资产准备，并强制隐藏 CUDA。
- GPU 命令先在 CUDA 隐藏/离线状态下完成高内存 CPU 数据封存，再编译 kernel 并训练。
- 正常 GPU 入口固定 R0。本机 90 GiB 不满足 R1 的 128 GiB RAM 门禁，不自动切 R1。
- 开始训练前仍必须动态确认无其它 GPU compute process、显存空闲、持久盘余量和当前 cgroup 配额。
