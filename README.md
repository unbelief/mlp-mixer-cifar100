# MLP-Mixer CIFAR-100

这是一个基于 MLP-Mixer 的 CIFAR-100 图像分类实验项目。

## 运行环境

项目使用 PyTorch 和 torchvision。程序会根据当前环境选择 CUDA 或 CPU。

主要实验设置：
- 数据集：CIFAR-100
- 输入图像：32×32 RGB
- patch size：4
- batch size：64
- 默认训练轮数：30

## 项目文件

- `models/mlp_mixer.py`：MLP-Mixer 模型
- `config.py`：模型配置
- `data.py`：CIFAR-100 数据加载
- `train.py`：模型训练
- `evaluate.py`：测试集评估
- `predict.py`：单张样本预测
- `utils.py`：训练过程中的辅助函数
- `smoke_test.py`：简单的模型运行检查
- `experiments/ablation.py`：模型结构参数对比

## 安装

```bash
pip install -r requirements.txt
```

第一次运行时会自动下载 CIFAR-100 数据集。

## 训练

Small 模型：

```bash
python train.py --model small --epochs 30 --batch-size 64
```

Tiny 模型：

```bash
python train.py --model tiny --epochs 10 --batch-size 32
```

如果需要指定 CPU：

```bash
python train.py --model tiny --device cpu --epochs 10 --batch-size 32
```

如果使用 CUDA：

```bash
python train.py --model small --device cuda --epochs 30 --batch-size 64
```

## 测试

训练完成后可以使用保存的最佳模型进行测试：

```bash
python evaluate.py --checkpoint checkpoints/best.pt
```

测试结果和混淆矩阵会保存在 `results` 目录中。

## 运行检查

```bash
python smoke_test.py
```

## 结构参数对比

```bash
python experiments/ablation.py
```

该脚本主要用于查看不同 patch size、网络深度和 embedding dimension 对模型参数量的影响。

## 参考

Tolstikhin et al., MLP-Mixer: An all-MLP Architecture for Vision.