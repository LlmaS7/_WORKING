# PyTorch 入门资源

## Knowledge

- [PyTorch 官方：Learn the Basics](https://docs.pytorch.org/tutorials/beginner/basics/intro.html)
  官方零基础路线，依次覆盖张量、数据集、模型、自动求导、优化与保存模型；作为本课程的主阅读材料。
- [PyTorch 官方：Datasets & DataLoaders](https://docs.pytorch.org/tutorials/beginner/basics/data_tutorial.html)
  解释 `Dataset` 与 `DataLoader` 的分工，以及批次张量的形状；用于理解 `get_data_loader()`。
- [PyTorch 官方：Build the Neural Network](https://docs.pytorch.org/tutorials/beginner/basics/buildmodel_tutorial.html)
  解释 `nn.Module`、`__init__()`、`forward()` 与全连接层；用于理解 `Net`。
- [PyTorch 官方：Optimizing Model Parameters](https://docs.pytorch.org/tutorials/beginner/basics/optimization_tutorial.html)
  解释 epoch、batch、loss、梯度清零、反向传播和优化器更新；用于理解训练循环。
- [PyTorch 官方：CrossEntropyLoss](https://docs.pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html)
  说明 `CrossEntropyLoss` 等价于 `LogSoftmax + NLLLoss` 的常见分类写法；用于比较原代码与推荐写法。

## Wisdom (Communities)

- [PyTorch Forums](https://discuss.pytorch.org/)
  官方社区，适合在遇到具体报错、形状不匹配或训练异常时检索和提问。
