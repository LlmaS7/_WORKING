## 基于生成式模型（Flow Matching/Diffision Policy）的机械臂操作任务的策略研究
1. 方向介绍
    方向与研究目的：
    本方向属于机器人学习（Robot Learning）中的机械臂操作与动作生成研究，主要目标是让机械臂根据相机图像、机器人自身状态以及任务指令，自主生成后续一段时间的连续动作，从而完成抓取、放置、推拉、物体操作等任务。与传统通过人工规划每一步运动轨迹的方法不同，该方向主要利用人类示范或机器人采集的数据训练策略模型，使机械臂从数据中学习“看到当前场景后下一步应该如何运动”。最终希望机械臂能够面对不同物体、位置和任务，自主生成合理、连续且稳定的动作轨迹，并具有一定的泛化能力。
    本课题本方向主要研究基于生成模型（Generative Model）的机器人策略，重点是 Diffusion Policy Model（扩散策略模型）和 Flow Matching（流匹配）在机械臂动作生成中的应用。模型通常以视觉图像、机器人关节/末端状态以及语言任务等作为条件，通过 Transformer等网络建立环境信息与未来动作序列之间的关系，再利用 Diffusion 或 Flow Matching 从随机分布逐步生成机械臂未来一段时间的 Action Chunk。主要研究现有模型存在的注意力冗余、模型计算量大、训练时间长、生成动作需要多次网络前向计算、实时控制速度不足等问题，通过优化 Attention、网络结构、采样过程和 Flow Matching 推理方式，提高动作生成速度和任务成功率。
2. 该技术需要的前期必备知识
    需要建立机械臂工作的整体概念，理解机械臂的组成，了解关节空间、末端位姿、坐标系、旋转矩阵、四元数、正运动学 FK、逆运动学 IK、轨迹和控制频率等基本概念（智能机器人原理课程）。并搞清楚一个机器人任务从“相机/传感器获得观测 → 策略模型根据当前观测与本体状态等信息产生 动作Action → Action 转换成关节或末端控制命令 → 控制器执行 → 得到新的 Observation”的一个工作闭环。
    机器人软件与仿真基础： 软件方面需要熟悉 Linux/Ubuntu 的基本操作，能够在电脑上创建 Python 环境、安装依赖、运行程序和查看 GPU 状态；掌握 Git/GitHub，能够 clone 项目、切换分支、提交和管理代码；掌握至少一个机器人仿真环境，例如MuJoCo或NVIDIA Isaac Lab。前期达到能够在仿真中加载机械臂、读取相机和机器人状态、控制机械臂末端或关节、控制夹爪、执行一段动作并完成 Reach/Pick/Place 等简单任务即可。后续如果进入实机实验，再学习 ROS/ROS2、机械臂 SDK、相机标定、坐标系转换以及实机通信，不需要在最开始投入大量时间学习 ROS 底层原理。
    机器学习与模仿学习基础：机器学习方面首先需要掌握 Python和 PyTorch，理解数据集、训练集/验证集、Batch、Epoch、Loss、Optimizer、Learning Rate、Forward、Backpropagation、Checkpoint、Inference 等神经网络训练的概念与流程。理论上需要掌握 MLP、CNN 的基本流程，重点学习 Transformer 和 Attention。机器人学习方面首先理解 Behavior Cloning（行为克隆）和 Imitation Learning（模仿学习），也可以了解reinforcement learning（强化学习）。
3. 该方向进行算法研究时需要掌握的深度知识
    Transformer 是本方向非常重要的底层网络结构，需要理解 Token、Embedding、Position/Time Embedding、Self-Attention、Cross-Attention、Q/K/V等概念。最重要的起点论文是 Vaswani 等人的 《Attention Is All You Need》。
    Diffusion Policy 扩散模型需要理解训练和推理流程，能够看懂机器人 Diffusion Policy 代码并进行修改。需要理解数据逐渐加噪和模型逐渐去噪的基本过程。生成扩散模型的重要基础论文是《Denoising Diffusion Probabilistic Models》和《Diffusion Policy: Visuomotor Policy Learning via Action Diffusion》。
    Flow Matching 是本课题后期需要重点掌握的核心算法，需要能够解释基本数学原理、理解训练目标、实现Flow Matching、看懂并修改 Flow Policy。《Flow Matching for Generative Modeling》、《Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow》。
4. 本科生前期了解—中期复现—后期开发的具体路线
    前期：建立基本知识和代码能力。 第一阶段建议本科生先了解高等数学、线性代数、概率论的数学内容，学习 Python。深度学习课程重点掌握神经网络、反向传播、CNN、Transformer 和 Attention；机器人课程重点掌握坐标系、关节空间、笛卡尔空间、FK/IK、机械臂 Observation/Action 和基本控制流程。这个阶段不要求阅读大量论文，可以先阅读《Attention Is All You Need》，然后自己在GitHub中完成一个简单的 MLP/CNN/Transformer 训练任务。机器人方面使用 MuJoCo 或 Isaac Lab，能够完成“启动环境—读取 Observation—给机械臂发送 Action—控制夹爪—完成 Reach/Pick/Place”。
    中期：复现经典机器人生成策略。 第二阶段正式进入 Imitation Learning 和机器人 Action Generation。建议按照 Behavior Cloning → ACT → Diffusion Policy → Flow Matching 的顺序进行。理解机器人数据集如何组织；随后阅读 ACT，理解 Transformer 和 Action Chunk；之后重点阅读并运行 Diffusion Policy，理解 Observation Encoder、Condition、Action Sequence、Diffusion Training 和 Sampling 的完整流程；最后复现 Flow Matching。

- 中期推荐复现的项目来源。 项目原则上优先找论文作者官方 GitHub，而不是直接复现第三方修改版本。需要能够根据论文找到官方代码 → 配置环境 → 跑通预训练模型 → 跑训练 → 修改配置 → 复现实验指标 → 找到 Policy 网络和 Sampling 部分的代码位置。
- 后期：从复现进入算法开发。即将 Diffusion/Flow Matching 与 Transformer Robot Policy 结合起来，并研究模型存在的问题。


