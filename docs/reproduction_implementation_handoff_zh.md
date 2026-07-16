# ReMemR1 实施交接入口

> 本文件不再维护一套独立参数，避免与最终方案冲突。
> 唯一权威方案：[final_reproduction_plan_zh.md](./final_reproduction_plan_zh.md)
> 论文：[2509.23040v5.pdf](../2509.23040v5.pdf)
> 工作区：`D:\codespace\python\ReMemR1`
> 分支：`reproduction/qwen35-plan`

新会话应直接阅读最终方案，特别是：

- 第 0 节：最终固定决策；
- 第 5 节：LoRA/FSDP/rollout/checkpoint 契约；
- 第 8 节：正式超参数；
- 第 9 节：G-1 → G0 → G1 → G2 → B/C40 → B/C80 的执行顺序；
- 第 14 节：验收清单；
- 第 16 节：可直接使用的新会话启动指令。

当前项目定位固定为：

> **基于 Qwen3.5-4B LoRA-GRPO 的 ReMemR1 缩小机制复现**

不要再按旧交接中的 2B 全参数主线实施。正式模型是 Qwen3.5-4B，
训练是 LoRA-GRPO，硬件是单张 RTX PRO 6000 96GB。

开始实现前先执行 `git status`，保留工作区已有变更。除非用户明确要求，
不要自动提交或推送。
