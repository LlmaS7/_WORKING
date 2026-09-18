# 机械臂生成式策略学习路径

> 研究主线：Behavior Cloning → Action Chunk → Diffusion Policy → Flow Matching → 算法对比与改进  
> 当前起点：正在建立 Python、PyTorch、神经网络训练流程等基础能力  
> 原始方向说明：[temp.md](./temp.md)

## 1. 最终目标

围绕一个机械臂操作任务，独立完成以下完整流程：

1. 理解并处理机器人示范数据；
2. 训练一个 Behavior Cloning 基线；
3. 复现 Diffusion Policy；
4. 实现或复现基于 Flow Matching 的动作生成策略；
5. 在相同任务、数据和硬件条件下比较成功率与推理速度；
6. 根据实验结果提出一个范围明确的改进方法。

最终应能清楚解释：

```text
Observation
    ↓
视觉与状态编码
    ↓
策略模型生成未来动作序列
    ↓
执行部分动作
    ↓
获得新的 Observation
    ↓
继续闭环控制
```

## 2. 环境路线

### 前期：Windows + WSL2 Ubuntu

前期在 WSL2 中完成：

- Linux 命令行与 Git；
- Python 和 PyTorch；
- Behavior Cloning；
- Diffusion、Flow Matching 的小型实验；
- MuJoCo 中的基础任务；
- 论文代码阅读与修改。

环境使用原则：

- 项目放在 WSL2 的 Linux 文件系统中，例如 `~/projects`；
- 不在 `/mnt/c`、`/mnt/d` 等 Windows 挂载目录中进行长期训练；
- 保留 Ubuntu 的系统 Python，不执行 `sudo pip install`；
- 每个主要项目建立独立的 Conda 或虚拟环境；
- NVIDIA 显卡只安装 Windows 驱动，不在 WSL2 中安装 Linux 显卡驱动；
- GitHub 同步代码、配置和笔记，不同步虚拟环境、数据集和 checkpoint。

### 后期：原生 Ubuntu

出现以下需求后，再迁移到原生 Ubuntu：

- 使用 Isaac Sim 或 Isaac Lab；
- 连接机械臂、相机及其他 USB 设备；
- 使用 ROS 2 进行实机通信；
- WSL2 出现无法绕开的图形、网络或硬件兼容问题。

## 3. 学习阶段

### 阶段 0：Python 与 Linux 基础

#### 学习内容

- Python 基本语法、函数、类、模块和异常；
- NumPy 数组、索引、广播和矩阵运算；
- Linux 目录、文件、权限和进程的基本概念；
- Git 的 clone、status、add、commit、pull、push 和 branch；
- Conda 环境的创建、激活、导出与恢复。

#### 完成标志

- 能在 WSL2 中创建并运行 Python 项目；
- 能独立安装依赖并处理常见路径问题；
- 能用 GitHub 管理代码和 Markdown 笔记；
- 能说清 Python 环境、包和项目之间的关系。

---

### 阶段 1：PyTorch 训练闭环

#### 学习内容

- Tensor、形状、数据类型和设备；
- Dataset、DataLoader、Batch 和数据划分；
- `nn.Module`、`forward()` 和常见网络层；
- Loss、Optimizer、Learning Rate；
- Forward、Backward、梯度清零和参数更新；
- Epoch、Checkpoint、Inference 和验证集；
- MLP、CNN 的基本结构；
- 训练曲线、过拟合与欠拟合。

#### 完成标志

- 能沿着“数据 → 模型 → 损失 → 梯度 → 参数更新 → 评估”解释训练过程；
- 能判断主要张量在网络各层的形状；
- 能独立训练并保存一个 MNIST 分类模型；
- 能加载 checkpoint 并进行推理；
- 能定位维度错误、设备错误和数据类型错误。

---

### 阶段 2：机器人与机械臂基础

#### 学习内容

- 关节空间与笛卡尔空间；
- 基坐标系、末端坐标系和相机坐标系；
- 位置、姿态、旋转矩阵和四元数；
- 正运动学 FK 与逆运动学 IK 的输入、输出和用途；
- 位置、速度、增量和力矩等动作表示；
- 控制频率、轨迹、夹爪和末端执行器；
- Observation、Action、Reward、Episode；
- 开环控制与闭环控制。

#### 完成标志

- 能说明一个机械臂控制闭环；
- 能区分关节动作和末端动作；
- 能解释一条机器人示范轨迹包含哪些数据；
- 能判断策略输出如何转换为实际控制命令。

此阶段先理解物理含义与数据流，不要求完整推导复杂运动学公式。

---

### 阶段 3：MuJoCo 与简单机器人任务

#### 学习内容

- 启动仿真环境和加载机械臂；
- 获取关节状态、末端状态和相机图像；
- 发送关节或末端控制命令；
- 控制夹爪；
- Episode 重置、终止和成功条件；
- 控制频率与仿真步长；
- Reach、Push、Pick、Place 等简单任务。

#### 完成标志

- 能运行一个现有 MuJoCo 任务；
- 能打印并解释 Observation 与 Action 的形状和单位；
- 能用预设动作完成 Reach 或 Push；
- 能保存一次 rollout 的状态、动作和结果。

前期只选择一个仿真平台。暂不同时学习 MuJoCo 和 Isaac Lab。

---

### 阶段 4：模仿学习与 Behavior Cloning

#### 学习内容

- Imitation Learning 与 Behavior Cloning；
- 示范数据的轨迹、时间步和 Episode 结构；
- 状态输入、视觉输入和多模态输入；
- 动作的绝对量、增量与归一化；
- 训练集、验证集按完整轨迹划分；
- 单步动作预测；
- 闭环误差累积与分布偏移；
- 离线 Loss 与在线任务成功率的区别。

#### 完成标志

- 能从示范轨迹构造训练样本；
- 能训练一个状态输入的 BC 策略；
- 能在仿真中闭环运行策略；
- 能记录成功率、失败类型和动作曲线；
- 能解释为什么验证集 Loss 较低仍可能任务失败。

BC 是后续所有生成式策略实验的基线，不应跳过。

---

### 阶段 5：Transformer 与 Action Chunk

#### 学习内容

- Token、Embedding 和位置编码；
- Q、K、V 与 Self-Attention；
- Cross-Attention 与条件信息；
- Transformer Encoder 与 Decoder 的基本数据流；
- 单步动作与 Action Chunk；
- ACT 的核心思想；
- 动作块长度与重新规划频率。

#### 完成标志

- 能画出 Transformer 策略的数据流；
- 能说明 Action Chunk 相比单步预测的作用；
- 能看懂张量维度在 Attention 中如何变化；
- 能在代码中找到观测编码、动作查询和动作输出位置。

ACT 用于理解 Transformer 策略与 Action Chunk。研究主线不要求先完整复现 ACT，再进入 Diffusion Policy。

---

### 阶段 6：扩散模型基础

#### 学习内容

- 数据分布与随机采样；
- 高斯噪声；
- 前向加噪与反向去噪；
- 时间步编码；
- 噪声预测网络；
- 训练过程与采样过程的区别；
- DDPM 的基本目标；
- 采样步数、速度与质量之间的关系。

#### 完成标志

- 能在二维数据或 MNIST 上运行一个小型扩散模型；
- 能解释训练时模型的输入、目标和 Loss；
- 能解释为什么推理需要多次网络前向计算；
- 能修改采样步数并观察速度与结果变化。

先在简单数据上理解扩散过程，再进入机器人动作扩散。

---

### 阶段 7：Diffusion Policy 复现

#### 学习内容

- 条件动作生成；
- Observation Encoder；
- 状态与视觉条件；
- Action Horizon、Observation Horizon 和 Prediction Horizon；
- 动作序列加噪与去噪；
- Receding Horizon Control；
- CNN/U-Net 策略与 Transformer 策略；
- 推理步数与实时控制延迟。

#### 推荐顺序

1. 状态输入的 Push-T；
2. 运行官方预训练模型；
3. 运行官方评估；
4. 使用官方数据训练一个种子；
5. 阅读 Policy、Dataset、Workspace 和 Sampling 代码；
6. 再进入视觉输入版本；
7. 最后迁移到机械臂任务。

#### 完成标志

- 能跑通官方状态输入示例；
- 能从数据加载追踪到动作输出；
- 能解释动作块如何生成和执行；
- 能定位条件编码、网络主体和 Sampling 部分；
- 能记录成功率、单次决策延迟和网络前向次数；
- 能改变一个变量并完成受控实验。

---

### 阶段 8：Flow Matching

#### 学习内容

- Continuous Normalizing Flow 的基本概念；
- 概率路径与时间变量；
- 向量场和速度预测；
- Flow Matching 的训练目标；
- 条件 Flow Matching；
- Euler 等数值积分方法；
- ODE 求解步数与 Network Function Evaluations；
- Rectified Flow 与路径变直思想。

#### 实现顺序

1. 二维点集上的无条件 Flow Matching；
2. 带类别或状态条件的 Flow Matching；
3. 生成低维动作序列；
4. 在与 Diffusion Policy 相同的数据接口上实现策略；
5. 接入相同仿真任务；
6. 比较生成速度和闭环任务表现。

#### 完成标志

- 能写出并解释 Flow Matching 的训练目标；
- 能实现一个小型条件速度场；
- 能用数值积分从噪声生成动作；
- 能解释 Flow Matching、Diffusion 和 Rectified Flow 的联系与区别；
- 能在同一任务上与 BC、Diffusion Policy 公平比较。

---

### 阶段 9：算法研究

先聚焦一个可验证的问题：

> 在相同任务、数据、模型规模和训练预算下，减少动作生成所需的网络计算次数，会怎样影响推理延迟与任务成功率？

#### 基线

- Behavior Cloning；
- Diffusion Policy；
- Flow Matching Policy。

#### 主要自变量

- Diffusion Sampling Steps；
- ODE Solver Steps；
- Action Chunk 长度；
- 每次重新规划后执行的动作数量；
- 网络宽度或层数；
- 视觉编码器是否冻结。

#### 主要指标

- 任务成功率或环境规定得分；
- 单次动作块生成时间；
- 从 Observation 到 Action 的完整延迟；
- Network Function Evaluations；
- 参数量与显存占用；
- 多个训练种子的均值与波动；
- 典型失败模式。

#### 实验原则

- 使用相同的数据划分；
- 使用相同的 Observation 和 Action 定义；
- 保持训练预算尽量一致；
- 不用单次最好结果代替稳定结果；
- 同时报告成功率和推理速度；
- 修改网络结构前先完成耗时分析，确认真正瓶颈。

---

### 阶段 10：ROS 2 与实机扩展

进入实机阶段后再学习：

- ROS 2 的 Node、Topic、Service、Action 和 TF；
- 机械臂 SDK；
- 相机驱动与时间同步；
- 手眼标定和坐标变换；
- 实机控制频率与通信延迟；
- 动作限幅、碰撞保护和急停；
- 示范采集与数据回放；
- 仿真到实机的接口差异。

#### 完成标志

- 能读取机械臂和相机数据；
- 能在安全限制下发送控制命令；
- 能采集、保存并回放一条示范轨迹；
- 能让仿真策略接口与实机控制接口保持一致。

## 4. 论文阅读顺序

论文阅读与代码复现同步进行，不需要在前期一次读完所有论文。

1. **Attention Is All You Need**  
   重点理解 Token、位置编码和 Attention 数据流。

2. **Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware**  
   重点理解 ACT、Action Chunk 和误差累积。

3. **Denoising Diffusion Probabilistic Models**  
   重点理解训练目标、加噪和采样。

4. **Diffusion Policy: Visuomotor Policy Learning via Action Diffusion**  
   重点理解条件动作序列、滚动时域控制和机器人评估。

5. **Flow Matching for Generative Modeling**  
   重点理解概率路径、目标向量场和训练目标。

6. **Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow**  
   重点理解直线路径、推理步数和 Rectification。

## 5. 数据处理清单

每次开始复现或新任务时，先明确：

- Observation 包含哪些量；
- Action 表示关节、末端、速度、增量还是力矩；
- Observation 与 Action 的采样频率；
- 图像、机器人状态和动作如何进行时间对齐；
- 动作和状态怎样归一化、反归一化；
- Action Chunk 在轨迹末尾如何补齐；
- 训练集与验证集是否按完整 Episode 划分；
- 闭环执行时预测多少步、实际执行多少步；
- 成功条件、终止条件和超时条件是什么。

## 6. GitHub 与实验文件管理

建议提交：

- 源代码；
- Markdown 笔记；
- 环境描述文件；
- 实验配置；
- 少量结果图和汇总表。

建议忽略：

```text
.env
.venv/
__pycache__/
data/
datasets/
outputs/
checkpoints/
wandb/
*.ckpt
*.pth
*.pt
```

每个可复现实验至少记录：

- Git commit；
- 环境与依赖版本；
- 数据集版本与划分；
- 随机种子；
- 完整配置；
- 训练日志；
- checkpoint；
- 评估结果与失败案例。

## 7. 当前优先顺序

当前只推进以下内容：

1. 完成 MNIST PyTorch 训练流程的理解；
2. 掌握 Tensor、Dataset、DataLoader、模型、Loss 和 Optimizer；
3. 独立运行、保存和加载一个 PyTorch 模型；
4. 建立 WSL2、GitHub 和独立 Python 环境的基本工作流；
5. 随后进入机械臂 Observation、Action 和闭环控制概念。

以下内容暂缓：

- Isaac Lab；
- ROS 2 底层机制；
- 实机驱动与标定；
- 强化学习深入内容；
- 同时复现多个大型策略；
- 在完成可靠基线前修改 Attention 或设计复杂新网络。

