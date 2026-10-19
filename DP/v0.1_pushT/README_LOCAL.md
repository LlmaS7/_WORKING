# v0.1 PushT：本机入门版

在官方 [real-stanford/diffusion_policy](https://github.com/real-stanford/diffusion_policy) 克隆上适配。
上游提交：`5ba07ac6661db573af695b419a7947ecb704690f`。

本版使用 `E:\_WORKING\.venv`，保留已有 Python 3.14、PyTorch 和 CUDA，提供低维 PushT 的独立运行入口。官方完整仓库的其他任务、视觉训练和实机入口仍保留原始依赖要求；本版的验证范围是下面的 `run_pusht.py`。

## 从这里开始

以下命令均在 PowerShell 的 `E:\_WORKING` 目录执行，无需激活环境。

```powershell
# 1. 检查环境渲染、GPU 训练一步、动作去噪和环境执行
.\.venv\Scripts\python.exe DP\v0.1_pushT\run_pusht.py smoke

# 2. 下载官方示范数据（已存在时直接跳过）
.\.venv\Scripts\python.exe DP\v0.1_pushT\run_pusht.py download

# 3. 查看轨迹数量、数据形状，并生成第一条示范的视频
.\.venv\Scripts\python.exe DP\v0.1_pushT\run_pusht.py inspect

# 4. 手动体验：鼠标控制推手，R 换一个初始状态，Esc 退出
.\.venv\Scripts\python.exe DP\v0.1_pushT\run_pusht.py demo

# 5. 短训练：只确认真实数据上的训练与保存流程，不能据此判断策略效果
.\.venv\Scripts\python.exe DP\v0.1_pushT\run_pusht.py train --small --epochs 1 --max-train-steps 10

# 6. 正式训练：自动选择 CUDA，使用中等大小的 U-Net
.\.venv\Scripts\python.exe DP\v0.1_pushT\run_pusht.py train --epochs 100
```

训练结束会打印 `best.pt` 的完整路径。评估时，将下面的路径替换为该实际路径：

```powershell
.\.venv\Scripts\python.exe DP\v0.1_pushT\run_pusht.py eval --checkpoint "实际训练目录\best.pt" --episodes 5
```

评估默认使用与训练对应的 DDPM，采样 100 步；可以增加 `--sampling-steps 20` 探索速度与效果变化。训练和评估随机种子默认均可用 `--seed` 指定；评估建议使用训练时未用于检查表现的固定种子集合，例如 `--seed 100000`。这里只按观测归一化后生成动作，没有语言输入。

## 数据与模型

- 观测：9 个物块关键点的二维坐标 + 推手二维位置，共 20 维。
- 环境原始输出另含 20 个可见性标志；本版设置全部可见，仅取前 20 维输入模型。
- 从示范数据反算并保存物块局部关键点，评估使用同一组点，避免新版渲染库改变关键点采样位置。
- 动作：推手二维目标位置，坐标约在 `[0, 512]`。
- 观测长度 2、动作预测长度 16、每次执行 8 步，然后重新观察。
- episode 单独划分训练 / 验证集，比例约 9:1；序列不会跨 episode。
- 仅用训练 episode 拟合归一化范围；短序列边缘重复端点。
- 使用官方 `ConditionalUnet1D`、现代 Diffusers DDPM 调度器、AdamW、学习率预热和余弦衰减，以及 EMA 参数。
- 本地默认网络宽度 `[128, 256, 512]`、时间嵌入 128；`--small` 使用 `[32, 64, 128]`。它们比官方完整配置小，方便学习，不作为论文指标的原样复现。
- 本版 checkpoint 是精简入口自己的 `.pt` 格式，不直接兼容上游 `.ckpt`；当前训练命令每次开启新实验，没有断点续训功能。

## 输出在哪里

所有默认路径相对于项目目录解析，不受启动位置影响。

```text
data/pusht/pusht_cchi_v7_replay.zarr/   官方示范数据
outputs/smoke/                        环境截图与基础验证结果
outputs/inspect/                      示范视频和数据摘要
outputs/train_时间/                   配置、训练日志、best.pt、latest.pt
outputs/eval_时间/                    策略视频与 eval.json
```

`eval.json` 分别记录平均最高得分、成功率和每次动作块生成延迟。成功指某一步的物块目标覆盖率超过 95%；得分不能直接换算为成功率。延迟包含归一化、去噪和动作传回 CPU，不包含环境仿真和视频编码。

## 兼容修改

1. Gym 的基础类与空间改用 Gymnasium；为兼容原仓库包装器，PushT 保留原有 `reset -> obs`、`step -> obs, reward, done, info` 接口，不把它当作标准 Gymnasium 五返回值环境使用。
2. 碰撞回调适配 Pymunk 7 的 `on_collision`。
3. 关键点颜色表适配新版 Matplotlib，动作标记坐标适配实际渲染尺寸。
4. 用 Zarr 3 直接读取官方 Zarr v2 数据，增加轻量训练 / 评估入口，避免加载本任务不需要的旧版 NumPy、Numba、Ray、MuJoCo 和 Linux 视频处理模块。
5. Windows 数据加载使用 `num_workers=0`；checkpoint 显式使用 `weights_only=True` 加载本入口保存的张量与基础配置。

依赖直接安装到根目录虚拟环境。根目录 `requirements.txt` 记录整个共享环境的固定版本，项目 `requirements-pusht.txt` 记录本入口直接依赖。

若以后在同样的共享环境补装，可在根目录执行：

```powershell
.\.venv\Scripts\python.exe -m pip install -c requirements.txt -r DP\v0.1_pushT\requirements-pusht.txt
.\.venv\Scripts\python.exe -m pip check
```

原始训练说明仍见 [README.md](README.md)，学习路线见 [../Guide.md](../Guide.md)。

## 本轮已验证（2026-10-19）

- 根目录共享环境安装依赖完成，`pip check` 通过，原有 22 个固定依赖版本全部保留。
- RTX 5070 Ti Laptop GPU 上通过环境渲染、动作去噪、梯度更新和环境执行检查。
- 6 项测试通过：episode 划分、序列边界、训练集归一化、关键点标定、接触推动和环境复现。
- 官方数据共 206 条轨迹、25,650 帧；按种子 42 划分为 185 条训练轨迹和 21 条验证轨迹。
- 已生成 `outputs/inspect/demonstration.mp4` 和 `first_frame.png`。
- 默认 16,605,954 参数网络完成 10 次真实数据训练更新，并保存到 `outputs/verification_train/best.pt`。
- 已加载该模型，使用种子 100000、100 步 DDPM 采样，执行 300 个环境步并保存 `outputs/verification_eval/rollout_00.mp4` 与 `eval.json`。
- 这次短训练只验证流程；回放得分 0，未成功完成任务。需要正式训练后再评估效果。

在根目录重跑针对性测试：

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s DP\v0.1_pushT -p test_pusht_local.py -v
```
