<div align="center">

<p align="center">
  <strong>DeepETD：基于多模态生物医学数据的相互作用预测深度学习框架</strong>
</p>

**选择语言 / Select Language:**

[![English](https://img.shields.io/badge/🇺🇸_English-Document-007acc?style=for-the-badge&logo=book&logoColor=white)](README(CN_EN)/README_EN.md)
[![中文](https://img.shields.io/badge/🇨🇳_中文-完整文档-28a745?style=for-the-badge&logo=book&logoColor=white)](README(CN_EN)/README_CN.md)

</div>

## 简介
DeepETD整合了多源生物医学数据，包括亚细胞定位、细胞表型和疾病数据。
然后采用具有注意力机制的深度学习算法，聚焦数据最相关部分，有效捕捉代谢物与蛋白质间的复杂关系，
增强了代谢物-蛋白质相互作用预测能力。预测结果存储在名为[EMTDD](http://otter-simm.com/EM/search.html)的数据库中。


## 架构

```editorconfig
├── 项目根目录
│   ├── 配置层 (Config Layer)
│   │   └── config.yaml (配置文件)
│   │
│   ├── 核心模块层 (Core Modules)
│   │   ├── data_loader.py (数据加载器)
│   │   ├── model.py (模型定义)
│   │   ├── train.py (训练脚本)
│   │   └── predict.py (预测脚本)
│   │
│   ├── 数据层 (Data Layer)
│   │   └── Data/
│   │       ├── neg_datasets.json (负样本数据)
│   │       ├── pos_datasets.json (正样本数据)
│   │       ├── predict_datasets.json (待预测数据)
│   │       ├── phenotype.json (表型数据)
│   │       ├── text_data.json (文本数据)
│   │       └── disease_list.json (疾病列表)
│   │
│   ├── 输出层 (Output Layer)
│   │   ├── Result/
│   │   │   ├── model.pth (模型权重)
│   │   │   └── predictions.json (预测结果)
│   │   └── Log/ (日志目录)
```

## 使用

`Result`中保存了预训练过的模型，数据来源详见论文（暂时还没发），
如果想个性化自定义模型训练参数，可以修改`config.yaml`中的参数；
通过更改`Data/text_data.json`内容，最终可以在`Result`中得到预测结果及其可视化

```editorconfig
## 输入数据text_data的标准格式
## 分别为化合物/靶点/化合物-疾病/化合物-表型/化合物-亚细胞定位/靶点-疾病/靶点-亚细胞定位
[
    {
        "compound": "Norepinephrine",
        "protein": "TNF",
        "compound_diseases": [
            "Hypertension"
        ],
        "compound_phenotypes": [
            "Proliferation"
        ],
        "compound_subcellular_locations": [
            "Nucleus"
        ],
        "protein_diseases": [
            "Rheumatoid arthritis"
        ],
        "protein_phenotypes": [
            "Inflammation"
        ],
        "protein_subcellular_locations": [
            "Nucleus"
        ]
    }
```




