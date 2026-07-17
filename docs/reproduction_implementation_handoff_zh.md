# ReMemR1 实施交接入口

> 本文件不维护独立参数，避免与权威方案冲突。
> 唯一权威方案：[rtx5090_2b_reproduction_plan_zh.md](./rtx5090_2b_reproduction_plan_zh.md)
> 论文：[2509.23040v5.pdf](../2509.23040v5.pdf)
> 工作区：`D:\codespace\python\ReMemR1`
> 分支：`reproduction/rtx5090-2b`
> Active profile：`rtx5090-32g-qwen35-2b-v1`

新会话应直接阅读权威方案，特别是：

- 第 0-4 节：2B/5090 固定决策、容量策略和配置契约；
- 第 5-8 节：容量门禁、成对训练、评测与费用边界；
- 第 9-10 节：可恢复状态机和负例契约；
- 第 11 节：G-1 -> G0/G1 -> G2 -> B/C40 -> B/C80 的执行顺序；
- 第 13-14 节：可声明范围和验收清单。

当前项目定位固定为：

> **基于单张 RTX 5090 和 Qwen3.5-2B LoRA-GRPO 的 ReMemR1 缩小机制复现**

正式模型是固定 revision 的 Qwen3.5-2B，训练为 LoRA-GRPO，硬件为单张 GeForce RTX 5090
32GB。旧 4B / PRO 6000 文档只作历史参考，不进入 active profile。

本主线不增加 `compress_context` 或其它新工具，不改 JSON action，不改变 `<update>` / `<recall>` /
final answer 协议。无真实 GPU evidence 时只能声明代码与 CPU 验证完成；不得声称已跑通 5090，
不得填写虚构指标。

云端正常操作、恢复、费用证据和导出命令统一见
[scripts/cloud/README.md](../scripts/cloud/README.md)。开始工作前先执行 `git status`，保留工作区已有变更。
