# Diffusion Policy 入门：从 PushT 开始

目标：在一个简单任务中，理解“示范数据 → 训练策略 → 生成动作序列 → 闭环执行”，再比较 BC、Diffusion Policy 和 Flow Matching。

## 1. 先选低维 PushT

PushT 的任务是控制二维推手，把 T 形物块推到目标区域。

| 版本 | 策略输入 | 策略输出 | 学习重点 |
| --- | --- | --- | --- |
| 低维版 | 位置、关键点等数值 | 推手的未来动作序列 | 动作生成和闭环控制 |
| 图像版 | 场景图片和推手位置 | 推手的未来动作序列 | 加上视觉特征提取 |

官方仓库的低维配置使用 9 个物块关键点的二维坐标，加上推手二维位置，共 20 个观测数值；图像配置使用 96×96 RGB 图片和推手二维位置。不同示例可能使用不同观测表示，以实际配置和数据为准。

两者扩散的对象都是动作序列。图像版的图片是条件输入，不是要生成的结果。官方仿真中的动作是推手的二维目标位置，不是位置增量。

## 2. 推荐顺序

### 第一步：运行官方低维示例

- 优先从官方低维 Colab 入手，先看懂精简流程，再读完整仓库。
- 查看环境画面和示范轨迹，打印 observation、action 的形状与取值范围。
- 跑通策略 rollout：让模型实际控制环境，并保存视频。

完成标志：能解释“一条示范轨迹是什么、模型看到什么、输出什么”。

### 第二步：补一个 BC 基线

- 使用相同的观测和示范数据，用 MLP 预测动作，以 MSE 训练。
- 按完整 episode 划分训练集和验证集，避免相邻时间步泄漏。
- 把策略放回环境闭环执行，记录得分和失败视频。

完成标志：能说明为什么离线 loss 低，实际操作仍可能失败。

### 第三步：理解 Action Chunk

- 单步策略：每次预测一个动作。
- 动作块策略：每次预测一段动作，执行其中一部分，再获取新观测并重新预测。
- 分清观测长度、预测长度和每次执行的动作数量。

完成标志：能沿着时间轴解释“看几帧、预测几步、执行几步”。不要求先完整复现 ACT。

### 第四步：训练低维 Diffusion Policy

- 先用条件 1D U-Net 和示例默认参数，训练一个随机种子。
- 训练：从示范中取动作序列，随机选择扩散时间步并加噪，让网络根据观测、带噪动作和时间步预测噪声。
- 推理：从随机噪声开始多次去噪，得到动作序列，执行一部分后重新规划。
- 先理解数据归一化、时间窗口、条件输入和采样过程，再调整网络。

完成标志：能从一批数据追踪到 loss，也能从一次观测追踪到环境实际执行的动作。

### 第五步：做一次小实验

- 保持模型、数据和评估初始条件一致，比较少量不同采样步数。
- 记录环境得分、按明确判据统计的成功率、动作块生成延迟和 rollout 视频。
- 按实际 scheduler 支持的方式修改采样步数，并记录采样器类型。
- 注意：官方 `test/mean_score` 是得分指标，不能直接当作成功率百分比。

完成标志：能用数据说明采样速度与任务表现之间的关系。正式比较时再增加训练种子。

### 第六步：扩展

低维 PushT 跑通后，再做图像版 PushT；随后在相同数据与评估接口上接入 Flow Matching，最后迁移到机械臂任务。

## 3. 虚拟环境怎么划分

本工作区按当前约定统一使用 `E:\_WORKING\.venv`（Python 3.14.5），MLP、CNN、PushT 共用它；不在 DP 下另建环境。

官方完整仓库使用 Python 3.9、PyTorch 1.12.1 等旧版本。本地 `v0.1_pushT` 保留当前 PyTorch / NumPy，适配 PushT 所需接口，并提供独立入口；不要直接安装上游旧版 Conda 环境文件。

| 项目组 | 环境建议 |
| --- | --- |
| MLP / CNN 基础练习 | 继续共用根目录 `.venv` |
| PushT / Diffusion Policy | 使用根目录 `.venv`，按本地入口验证兼容性 |
| 同一实验中的 BC / DP / Flow Matching | 沿用同一个已验证环境，方便公平比较 |

- 运行脚本或 notebook 前确认解释器 / kernel 指向根目录 `.venv`。
- 补装依赖时约束现有版本，避免影响 MLP / CNN；根目录 `requirements.txt` 记录共享环境。
- Colab 仍可用作参考，但本轮以 Windows 本地入口为准。
- Windows 与 WSL 不能复用同一份 `.venv`；未来切换平台时再单独处理环境迁移。
- 数据和 checkpoint 不提交到 Git。

本地入口、运行命令和兼容范围见 [v0.1_pushT/README_LOCAL.md](v0.1_pushT/README_LOCAL.md)。

## 4. 第一个小目标

- [x] 跑通低维 PushT 环境。
- [x] 展示一条示范轨迹。
- [x] 打印一批观测与动作的形状。
- [x] 跑通一次策略 rollout 并保存视频。
- [ ] 画出“观测 → 策略 → 动作 → 环境 → 新观测”的闭环。

2026-10-19：以上技术检查已在 `v0.1_pushT` 完成。模型仅做了 10 次更新来验证训练与回放流程，尚未训练到能完成任务；下一步先阅读数据与模型入口，再进行正式训练。

## 官方资料

- [官方仓库与运行说明](https://github.com/real-stanford/diffusion_policy)
- [低维 Colab](https://colab.research.google.com/drive/1gxdkgRVfM55zihY9TFLja97cSVZOZq2B?usp=sharing)
- [图像 Colab](https://colab.research.google.com/drive/18GIHeOQ5DyjMN8iIRZL2EKZ0745NLIpg?usp=sharing)
- [低维观测配置](https://github.com/real-stanford/diffusion_policy/blob/main/diffusion_policy/config/task/pusht_lowdim.yaml)
- [图像观测配置](https://github.com/real-stanford/diffusion_policy/blob/main/diffusion_policy/config/task/pusht_image.yaml)
- [官方完整环境文件](https://github.com/real-stanford/diffusion_policy/blob/main/conda_environment.yaml)
