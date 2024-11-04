
# Triple-Branch Deep Fake Detection

基于三分支深度学习的假脸检测系统，创新性地结合空间域特征、频域特征和高频噪声特征，实现高精度的假脸识别。

## 项目特点

- 🚀 三分支创新架构
  - 空间域分支：捕捉图像的空间特征和纹理信息
  - 频域分支：分析图像的频率分布特征
  - 高频分支：专注于提取图像中的高频噪声特征
- 🔍 多维度特征融合
  - 特征级联融合
  - 自适应特征权重
- 📈 高性能表现
  - 优秀的检测准确率
  - 强大的泛化能力
- 🛠 完整工具链
  - 模块化设计
  - 完整的训练和推理流程
- 📊 全面的实验支持
  - 训练过程可视化
  - 详细的评估指标

## 环境要求

- Python 3.8+
- PyTorch 1.8+
- CUDA 11.0+ (推荐)

## 安装

1. 克隆仓库：

```bash
git clone https://github.com/zzyss-marker/deepfake.git
cd triple-branch-deepfake-detection
```


2. 创建虚拟环境：

```bash
conda create -n deepfake python=3.8
conda activate deepfake
```

3. 安装依赖：

```bash
pip install -r requirements.txt
```

## 项目结构

```
project_root/
│
├── configs/
│   └── config.py           # 配置文件
│
├── src/
│   ├── data/              # 数据处理模块
│   │   ├── dataset.py     # 数据集定义（包含频域转换）
│   │   └── transforms.py  # 数据增强
│   │
│   ├── models/            # 模型定义
│   │   ├── model.py      # 三分支网络架构
│   │   └── loss.py       # 损失函数
│   │
│   └── utils/             # 工具函数
│       ├── metrics.py     # 评估指标
│       └── helpers.py     # 辅助函数
│
├── train.py               # 训练脚本
├── predict.py             # 预测脚本
└── requirements.txt       # 项目依赖
```

## 模型架构

![Model Architecture](path_to_your_model_architecture_image.png)

三分支模型架构：

1. 空间域分支：使用 EfficientNet-B5 提取空间特征
2. 频域分支：分析 FFT 频谱特征
3. 高频分支：提取高频噪声特征
4. 特征融合：多层特征融合和自适应权重

## 使用方法

### 数据准备

1. 组织数据集结构：

```
data/
├── train/
│   ├── real/
│   └── fake/
└── val/
    ├── real/
    └── fake/
```

2. 生成训练标签：

```bash
python tools/prepare_data.py --data_dir data/ --output train_labels.csv
```

### 训练模型

1. 使用默认配置训练：

```bash
python train.py
```

2. 自定义配置训练：

```bash
python train.py --config configs/custom_config.py
```

3. 恢复训练：

```bash
python train.py --resume checkpoints/last.pth
```

### 模型预测

1. 单图像预测：

```bash
python predict.py --image path/to/image.jpg --model checkpoints/best.pth
```

2. 批量预测：

```bash
python predict.py --input_dir path/to/images/ --output results.csv
```

## 配置说明

主要配置参数（在 `configs/config.py` 中）：

```python
class TrainConfig:
    # 数据参数
    img_size = (256, 256)
    batch_size = 64
    num_workers = 4

    # 训练参数
    num_epochs = 5
    learning_rate = 1e-4
    max_lr = 1e-3
    weight_decay = 1e-4
    early_stop_patience = 10
    label_smoothing = 0.1

    # 模型参数
    model_name = 'efficientnet_b5'
    num_classes = 2
```

## 实验记录

使用 Weights & Biases 进行实验追踪：

1. 配置 wandb：

```bash
wandb login
```

2. 训练时自动记录：

- 训练/验证损失
- 准确率、AUC等指标
- 学习率变化
- 混淆矩阵
- 特征可视化

## 性能评估

在测试集上的表现：

| 指标    | 数值  |
| ------- | ----- |
| 准确率  | 99.4% |
| AUC     | 0.995 |
| F1 分数 | 0.984 |

## 引用

如果您使用了本项目的代码，请引用：

```bibtex
@misc{triple-branch-deepfake,
  author = {Your Name},
  title = {Triple-Branch Deep Fake Detection},
  year = {2024},
  publisher = {GitHub},
  url = {https://github.com/yourusername/triple-branch-deepfake-detection}
}
```

## 许可证

本项目采用 Apache2.0 许可证。详见 [LICENSE](LICENSE) 文件。

## 致谢

感谢我的小菜鸡。
