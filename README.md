# DeepETD: 内源性代谢物-靶蛋白相互作用预测深度学习模型

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-ee4c2c.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

## 📋 概述
DeepETD 是一个深度学习模型，专门用于预测内源性代谢物与靶蛋白之间的相互作用。该工具可通过识别代谢物的潜在蛋白靶点，加速药物发现和代谢通路分析。

## ✨ 功能特点
- 多模态数据整合（疾病、表型、结构信息）
- 处理不平衡数据集，支持正样本加权
- 带有早停机制的高效训练
- 支持 top-k 预测结果导出，便于下游分析

## 📥 安装

### 环境要求
- Python 3.8 或更高版本
- pip 包管理器

### 逐步安装指南
1. 克隆仓库：
   ```bash
   git https://github.com/qianwei1129/DeepETD.git
   ```

2. 安装依赖：
   ```bash
   pip install -r requirements.txt
   ```

## 📁 项目结构
```
DeepETD/
├── train.py              # 训练脚本
├── predict.py            # 预测脚本
├── requirements.txt      # Python依赖包
├── Data/                 # 数据目录
│   ├── disease_list.json
│   ├── phenotype.json
│   ├── pos_datasets.json
│   ├── neg_datasets.json
│   └── predict_datasets.json
├── models/               # 模型架构
├── utils/                # 工具函数
└── README.md
```

## 🚀 快速开始

### 1) 训练模型
使用您的数据集训练 DeepETD 模型：
```bash
python train.py \
  --disease_json ../Data/disease_list.json \
  --phenotype_json ../Data/phenotype.json \
  --positive_json ../Data/pos_datasets.json \
  --negative_json ../Data/neg_datasets.json \
  --predict_json ../Data/predict_datasets.json \
  --model_out best_model.pth \
  --epochs 20 \
  --patience 10 \
  --pos_weight 3.0
```

### 2) 进行预测
使用训练好的模型预测代谢物-蛋白质相互作用：
```bash
python predict.py \
  --disease_json ../Data/disease_list.json \
  --phenotype_json ../Data/phenotype.json \
  --positive_json ../Data/pos_datasets.json \
  --negative_json ../Data/neg_datasets.json \
  --predict_json ../Data/predict_datasets.json \
  --checkpoint best_model.pth \
  --out predictions.json
```



### 训练输出
- 模型检查点文件（`.pth`格式）
- 包含损失和指标的训练日志

### 预测输出
JSON格式的预测结果，结构如下：
```json
{
  "化合物ID": [
    {"protein": "P12345", "score": 0.95},
    {"protein": "Q67890", "score": 0.87},
    ...
  ],
  ...
}
```
每个化合物返回前20个得分最高的蛋白质。

## 📝 技术说明

### 模型架构
- 模型输出原始 logits
- 训练时使用 `BCEWithLogitsLoss`（数值更稳定）
- 仅在计算指标和预测时应用 `sigmoid` 激活函数

### 数据处理
- 词汇表大小根据拟合的标签编码器动态确定
- 处理空模态列表时回退到索引 0
- 支持自定义 `<UNK>` 标记配置

### 性能优化建议
- 根据数据集不平衡程度调整 `--pos_weight` 参数
- 监控验证损失以优化早停策略
- 如有可用GPU，可加速训练过程


