# PyTorch MNIST 手写数字识别项目 Handoff

## 1. 项目背景

当前项目是一个基于 **PyTorch** 的 MNIST 手写数字识别小项目。

当前学习目标不是单纯追求 MNIST 的最高准确率，而是通过一个足够简单、可以完整跑通的项目，理解 PyTorch 中神经网络训练的完整流程，包括：

- Dataset / DataLoader
- `nn.Module`
- 前向传播（forward）
- 损失函数（loss）
- 反向传播（backward）
- optimizer 参数更新
- epoch / batch
- 训练集与测试集准确率
- 过拟合
- 后续从 MLP 过渡到 CNN

---

## 2. 当前代码概况

当前文件：`test.py`

### 网络结构

当前模型是一个全连接神经网络（MLP）：

```text
28×28 image
   ↓ flatten
784
   ↓
Linear(784, 64)
   ↓ ReLU
Linear(64, 64)
   ↓ ReLU
Linear(64, 64)
   ↓ ReLU
Linear(64, 10)
   ↓
log_softmax
```

即：

```text
784 → 64 → 64 → 64 → 10
```

### 当前训练设置

```text
Optimizer: Adam
Learning rate: 0.001
Loss: NLL Loss
Batch size: 15
Epochs: 2
Dataset: MNIST
```

测试阶段使用 `argmax` 得到预测数字，并计算分类准确率。

---

## 3. 当前代码的训练流程

```text
加载 MNIST
    ↓
构建 DataLoader
    ↓
创建 Net
    ↓
测试随机初始化模型的 accuracy
    ↓
for epoch:
    ↓
    读取一个 batch
    ↓
    flatten: 28×28 → 784
    ↓
    forward
    ↓
    计算 NLL loss
    ↓
    backward
    ↓
    optimizer.step()
    ↓
每个 epoch 后测试 accuracy
    ↓
显示若干测试图片和预测结果
```

---

## 4. 已讨论的主要修改方案

### 第一优先级：增加 Epoch

当前只有：

```python
for epoch in range(2):
```

首先建议改成：

```python
for epoch in range(5):
```

然后尝试：

```python
for epoch in range(10):
```

目的不是简单认为“epoch 越大越好”，而是观察：

- loss 是否继续下降
- test accuracy 是否继续上升
- 是否出现过拟合

如果：

```text
training accuracy ↑
test accuracy ↑
```

说明仍然在正常学习。

如果：

```text
training accuracy ↑
test accuracy ↓
```

则可能出现过拟合。

---

### 第二优先级：记录 Loss

当前代码虽然计算了：

```python
loss = torch.nn.functional.nll_loss(output, y)
```

但没有保存或显示训练过程中的 loss。

建议记录 **每个 epoch 的平均 loss**。

期望输出类似：

```text
epoch 0 | loss: 0.52 | test accuracy: 0.91
epoch 1 | loss: 0.25 | test accuracy: 0.94
epoch 2 | loss: 0.18 | test accuracy: 0.96
...
```

后续建议绘制：

```text
Loss
│\
│ \
│  \__
│     \____
└────────── Epoch
```

这样可以直观看到：

```text
Forward
  ↓
Loss
  ↓
Backward
  ↓
参数更新
  ↓
Loss 逐渐下降
```

注意：单个 mini-batch 的 loss 不要求每一次都严格下降，更重要的是观察一个 epoch 的平均 loss 与整体趋势。

---

### 第三优先级：修改 Batch Size

当前：

```python
batch_size=15
```

可以对比测试：

```text
15
32
64
128
```

推荐首先尝试：

```python
batch_size=64
```

MNIST 训练集约有 60000 张图。

当：

```text
batch = 15
```

每个 epoch 大约需要 4000 次参数更新。

当：

```text
batch = 64
```

则约 938 次更新。

可以比较：

- 一个 epoch 的训练时间
- loss 曲线
- test accuracy
- 训练是否更稳定

建议使用 **控制变量法**，不要一次改很多参数。

---

## 5. 重要结构升级：MLP → CNN

当前模型会先：

```python
x.view(-1, 28*28)
```

将二维图像：

```text
28 × 28
```

拉平成：

```text
784
```

这样没有显式利用图像原本的二维空间关系。

因此，下一阶段适合升级成 CNN。

推荐结构示例：

```text
MNIST image
   ↓
Conv2d
   ↓
ReLU
   ↓
MaxPool
   ↓
Conv2d
   ↓
ReLU
   ↓
MaxPool
   ↓
Flatten
   ↓
Linear
   ↓
10 classes
```

CNN 是该项目最值得做的结构性升级之一。

---

## 6. 暂时不需要优先加入的内容

### Dropout

用于降低过拟合。建议在观察到明显过拟合后再引入。

### Weight Decay

属于正则化手段，可在后续对比。

### 数据增强

例如：

- 小角度旋转
- 平移
- 仿射变换

MNIST 很适合做简单的数据增强实验，但当前应先理解最基本训练流程。

### Learning Rate Scheduler

例如逐渐降低学习率。属于后续优化项，不应作为当前第一步。

---

## 7. 推荐实验路线

### Experiment 0：Baseline

保持当前代码：

```text
Model: MLP
Epoch: 2
Batch: 15
LR: 0.001
```

记录：

```text
initial accuracy
final accuracy
training time
```

### Experiment 1：只增加 Epoch

```text
Model: MLP
Epoch: 5 / 10
Batch: 15
LR: 0.001
```

观察 accuracy、loss、是否仍在明显提升。

### Experiment 2：记录 Loss 曲线

增加：

```text
epoch average loss
```

并画出：

```text
loss vs epoch
test accuracy vs epoch
```

### Experiment 3：Batch Size 实验

固定模型、Epoch、Learning Rate，只修改：

```text
Batch = 15 / 32 / 64 / 128
```

### Experiment 4：升级 CNN

保持训练参数大致一致，比较：

```text
MLP vs CNN
```

观察：

- accuracy
- 收敛速度
- 参数量
- 模型结构差异

### Experiment 5：研究过拟合

如果训练 epoch 较多后：

```text
training accuracy ↑
test accuracy 停滞或 ↓
```

再尝试：

- Dropout
- weight decay
- 数据增强

---

## 8. 推荐实验记录表

| Model | Epoch | Batch | LR | Train Loss | Test Accuracy | Training Time |
|---|---:|---:|---:|---:|---:|---:|
| MLP | 2 | 15 | 0.001 | - | - | - |
| MLP | 5 | 15 | 0.001 | - | - | - |
| MLP | 10 | 15 | 0.001 | - | - | - |
| MLP | 10 | 64 | 0.001 | - | - | - |
| CNN | 10 | 64 | 0.001 | - | - | - |

核心原则：

> 一次尽量只修改一个主要变量。

否则无法判断性能变化到底来自哪里。

---

## 9. 下一窗口建议继续完成的任务

### Task 1

基于当前 `test.py` 写一个 **最小修改版本**：

只加入：

- epoch 从 2 增加到 5 或 10
- 每个 epoch 统计平均 loss
- 打印 test accuracy

不要一开始重构整个程序。

### Task 2

解释平均 loss 的实现，例如：

```python
running_loss = 0
```

以及为什么：

```python
loss.item()
```

可以用于记录 loss。

### Task 3

加入 matplotlib，分别画：

```text
epoch-loss curve
epoch-test accuracy curve
```

并解释图像应该如何解读。

### Task 4

完成一轮 MLP 实验之后，再给出 CNN 版本。

CNN 代码应以教学为目标，结构保持简单。

需要重点解释：

```python
nn.Conv2d
```

的参数：

```text
in_channels
out_channels
kernel_size
stride
padding
```

以及 feature map 的尺寸变化。

### Task 5

比较：

```text
MLP vs CNN
```

不仅比较准确率，还应解释：

- 为什么 CNN 更适合图像
- 参数共享
- 局部连接
- 空间结构
- feature map

---

## 10. 当前学习目标

该项目的目标不是：

> 用各种技巧把 MNIST accuracy 榨到极限。

当前更重要的是理解：

```text
Dataset
↓
DataLoader
↓
Batch
↓
Forward
↓
Loss
↓
Backward
↓
Gradient
↓
Optimizer
↓
Parameter Update
↓
Epoch
↓
Evaluation
```

之后再逐步进入：

```text
MLP
↓
CNN
↓
更复杂视觉模型
↓
机器人视觉 / Diffusion Policy / 具身智能
```

因此后续修改应优先采用：

> 小步修改 + 控制变量 + 解释原因 + 比较实验结果

而不是一次加入大量高级技巧。

---

## 11. 给下一窗口的一句话上下文

> 我正在通过一个 PyTorch MNIST 手写数字识别小项目学习神经网络训练流程。目前已有一个 `784→64→64→64→10` 的 MLP，Adam(lr=0.001)、batch_size=15、epoch=2。请先在不大幅重构代码的情况下，帮我增加 epoch、记录每个 epoch 的平均 loss 和 test accuracy，并解释每处修改；完成基础实验后再逐步升级 CNN。重点是教学和理解训练过程，而不是单纯追求最高准确率。
