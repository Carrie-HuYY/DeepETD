import os
import json
import random
import numpy as np
import torch
import warnings
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings('ignore')

# -----------------------------
# 默认编码器列表
# -----------------------------
DEFAULT_SUBCELLULAR_LOCATIONS = [
    "Nucleus", "Cytoplasm", "Mitochondria", "Endoplasmic Reticulum",
    "Golgi Apparatus", "Lysosome", "Plasma Membrane", "Nuclear Membrane",
    "Peroxisome", "Nucleolus", "Cytoskeleton", "Vacuole", "Chloroplast",
    "Plasmid", "Ribosome", "Flagellum", "Microvilli", "Vesicle",
    "Thylakoid", "Centrosome", "Synaptic Vesicle", "Endosome",
    "Nuclear Pore Complex"
]

# -----------------------------
# 设置随机种子
# -----------------------------

def set_seed(seed: int = 42):
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

# -----------------------------
# 编码器
# -----------------------------

def _normalize_list(xs):
    return [x.lower().strip() for x in xs]


def build_label_encoders(disease_json_path: str,
                         phenotype_json_path: str,
                         subcellular_locations=None):
    """
    构建三个标签函数编码器，将疾病/表型/亚细胞定位转换为数值编码
    :param disease_json_path: 对应Data/disease_list.json
    :param phenotype_json_path: 对应Data/phenotype_list.json
    :param subcellular_locations: 默认使用预定义列表，即'DEFAULT_SUBCELLULAR_LOCATIONS'
    :return: 返回三个LabelEncoder编码器
    """
    with open(disease_json_path, 'r', encoding='utf-8') as f:
        all_diseases = json.load(f)
    with open(phenotype_json_path, 'r', encoding='utf-8') as f:
        all_phenotypes = json.load(f)

    if subcellular_locations is None:
        subcellular_locations = DEFAULT_SUBCELLULAR_LOCATIONS

    disease_encoder = LabelEncoder().fit(_normalize_list(all_diseases))
    phenotype_encoder = LabelEncoder().fit(_normalize_list(all_phenotypes))
    subcellular_location_encoder = LabelEncoder().fit(_normalize_list(subcellular_locations))

    return disease_encoder, phenotype_encoder, subcellular_location_encoder

# -----------------------------
# Dataset
# -----------------------------

class InteractionDataset(Dataset):
    """
    设置代谢物-蛋白质相互作用的数据集
    """
    def __init__(self, samples, disease_encoder, phenotype_encoder, subcellular_location_encoder):
        self.samples = samples
        self.disease_encoder = disease_encoder
        self.phenotype_encoder = phenotype_encoder
        self.subcellular_location_encoder = subcellular_location_encoder

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample, label = self.samples[idx]

        # Defensively handle empty lists
        def safe_transform(encoder, xs):
            xs = _normalize_list(xs)
            if len(xs) == 0:
                # If a field is empty, duplicate the first class index (0) to keep shape [1]
                return torch.LongTensor([0])
            return torch.LongTensor(encoder.transform(xs))

        compound_diseases = safe_transform(self.disease_encoder, sample.get('compound_diseases', []))
        compound_phenotypes = safe_transform(self.phenotype_encoder, sample.get('compound_phenotypes', []))
        compound_subcellular_locations = safe_transform(self.subcellular_location_encoder, sample.get('compound_subcellular_locations', []))

        protein_diseases = safe_transform(self.disease_encoder, sample.get('protein_diseases', []))
        protein_phenotypes = safe_transform(self.phenotype_encoder, sample.get('protein_phenotypes', []))
        protein_subcellular_locations = safe_transform(self.subcellular_location_encoder, sample.get('protein_subcellular_locations', []))

        return (
            compound_diseases,
            compound_phenotypes,
            compound_subcellular_locations,
            protein_diseases,
            protein_phenotypes,
            protein_subcellular_locations,
        ), torch.LongTensor([label])

# -----------------------------
# Samplers & Dataloaders
# -----------------------------

def _load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def make_samples(positive_json_path: str,
                 negative_json_path: str,
                 text_json_path: str):
    """
    构建三种类型（正/负/纯文本）的样本并分配标签

    :param
        positive_json_path: 正样本（相互作用对）的JSON文件路径
        negative_json_path: 负样本（非相互作用对）的JSON文件路径
        text_json_path: 纯文本样本的JSON文件路径（用于推理/预测）

    :return
        positive_samples: 带标签1的正样本列表
        negative_samples: 带标签0的负样本列表
        text_samples: 带标签0的文本样本列表（标签仅占位，推理时不用）
        all_samples: 正负样本合并的训练集
    """

    positive_data = _load_json(positive_json_path)
    negative_data = _load_json(negative_json_path)
    text_data = _load_json(text_json_path)

    # label: 1 for positives, 0 for negatives; text samples also 0 (labels unused at inference)
    positive_samples = [(entry, 1) for entry in positive_data]
    negative_samples = [(entry, 0) for entry in negative_data]
    text_samples = [(entry, 0) for entry in text_data]

    all_samples = positive_samples + negative_samples
    return positive_samples, negative_samples, text_samples, all_samples


def get_dataloaders(
        disease_json_path: str,
        phenotype_json_path: str,
        positive_json_path: str,
        negative_json_path: str,
        batch_size: int = 16,
        val_split: float = 0.2,
        seed: int = 42,
):
    set_seed(seed)

    disease_encoder, phenotype_encoder, subcellular_location_encoder = build_label_encoders(
        disease_json_path, phenotype_json_path
    )

    _, _, text_samples, all_samples = make_samples(
        positive_json_path, negative_json_path,
    )

    train_samples, val_samples = train_test_split(all_samples, test_size=val_split, random_state=seed)

    train_dataset = InteractionDataset(train_samples, disease_encoder, phenotype_encoder, subcellular_location_encoder)
    val_dataset = InteractionDataset(val_samples, disease_encoder, phenotype_encoder, subcellular_location_encoder)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return {
        'train': train_loader,
        'val': val_loader,
        'encoders': {
            'disease': disease_encoder,
            'phenotype': phenotype_encoder,
            'subcellular': subcellular_location_encoder,
        }
    }


def extract_names_from_text_json(text_json_path: str):
    """Return two lists aligned with text samples: protein_names, compound_names.
    Fallback to generated names if keys are missing.
    """
    data = _load_json(text_json_path)
    protein_names = []
    compound_names = []
    for i, entry in enumerate(data):
        protein_names.append(entry.get('protein_name', f'protein_{i}'))
        compound_names.append(entry.get('compound_name', f'compound_{i}'))
    return protein_names, compound_names


def debug_data_pipeline(
        data_dir: str = "Data",  # 数据目录
        batch_size: int = 4,
        seed: int = 42,
        debug_level: int = 2  # 1=基础, 2=详细, 3=完整
):
    """
    调试DeepETD数据管道的完整函数（适配您的数据结构）

    参数:
        data_dir: 数据目录路径
        debug_level:
            1 - 基础统计信息
            2 - 详细样本检查 + 编码器信息
            3 - 完整批处理检查 + 数据形状
    """

    print("=" * 60)
    print("DeepETD 数据管道调试工具（适配版）")
    print("=" * 60)

    # 构建文件路径
    file_paths = {
        'disease': os.path.join(data_dir, "disease_list.json"),
        'phenotype': os.path.join(data_dir, "phenotype.json"),  # 注意：您的文件是phenotype.json
        'positive': os.path.join(data_dir, "pos_datasets.json"),  # 注意：您的文件是pos_datasets.json
        'negative': os.path.join(data_dir, "neg_datasets.json"),  # 注意：您的文件是neg_datasets.json
        'text': os.path.join(data_dir, "text_data.json"),  # 注意：您的文件是text_data.json
        'predict': os.path.join(data_dir, "predict_datasets.json")  # 额外的预测数据集
    }

    # 0. 设置随机种子
    set_seed(seed)
    print(f"[设置] 随机种子: {seed}")
    print(f"[设置] 数据目录: {data_dir}")

    # 1. 检查JSON文件是否存在
    print(f"\n[阶段1] 文件检查:")
    for key, path in file_paths.items():
        if os.path.exists(path):
            size = os.path.getsize(path) / 1024  # KB
            print(f"  ✓ {key:12} | {os.path.basename(path):20} | {size:.1f} KB")
        else:
            print(f"  ✗ {key:12} | {os.path.basename(path):20} | 文件不存在!")
            if key in ['disease', 'phenotype', 'positive', 'negative']:
                print(f"      ⚠️  关键文件缺失，无法继续!")
                return

    # 2. 加载和检查原始数据
    print(f"\n[阶段2] 原始数据统计:")
    try:
        # 检查文件内容格式
        print(f"  正在检查文件格式...")

        for key, path in file_paths.items():
            if os.path.exists(path):
                try:
                    with open(path, 'r', encoding='utf-8') as f:
                        data = json.load(f)

                    if isinstance(data, list):
                        print(f"    {key:12}: {len(data)} 个条目 (列表格式)")
                    elif isinstance(data, dict):
                        print(f"    {key:12}: 字典格式，{len(data)} 个键")
                    else:
                        print(f"    {key:12}: 未知格式: {type(data)}")

                except json.JSONDecodeError as e:
                    print(f"    {key:12}: JSON解析错误: {e}")
                except Exception as e:
                    print(f"    {key:12}: 读取错误: {e}")

        # 加载具体数据
        diseases = _load_json(file_paths['disease'])
        phenotypes = _load_json(file_paths['phenotype'])
        positives = _load_json(file_paths['positive'])
        negatives = _load_json(file_paths['negative'])
        texts = _load_json(file_paths['text']) if os.path.exists(file_paths['text']) else []

        print(f"\n  详细统计:")
        print(f"    疾病列表: {len(diseases)} 个疾病")
        print(f"    表型列表: {len(phenotypes)} 个表型")
        print(f"    正样本: {len(positives)} 个")
        print(f"    负样本: {len(negatives)} 个")

        if texts:
            print(f"    文本数据: {len(texts)} 个")

        # 检查正样本的格式
        if positives and len(positives) > 0:
            first_pos = positives[0]
            print(f"\n  [正样本格式检查]:")
            print(f"    类型: {type(first_pos)}")
            if isinstance(first_pos, dict):
                print(f"    包含的键: {list(first_pos.keys())}")

                # 检查关键字段是否存在
                required_fields = ['compound_diseases', 'protein_diseases']
                for field in required_fields:
                    if field in first_pos:
                        print(f"    ✓ {field}: 存在 ({len(first_pos[field])} 个值)")
                    else:
                        print(f"    ✗ {field}: 缺失!")

        if debug_level >= 2:
            print(f"\n  [数据示例]:")
            if diseases and len(diseases) > 0:
                print(f"    前3个疾病: {diseases[:3]}")
            if phenotypes and len(phenotypes) > 0:
                print(f"    前3个表型: {phenotypes[:3]}")

    except Exception as e:
        print(f"  ✗ 数据加载失败: {e}")
        import traceback
        traceback.print_exc()
        return

    # 3. 构建编码器
    print(f"\n[阶段3] 编码器构建:")
    try:
        disease_enc, pheno_enc, loc_enc = build_label_encoders(
            file_paths['disease'],
            file_paths['phenotype']
        )

        print(f"  疾病编码器: {len(disease_enc.classes_)} 个类别")
        print(f"  表型编码器: {len(pheno_enc.classes_)} 个类别")
        print(f"  定位编码器: {len(loc_enc.classes_)} 个类别")

        if debug_level >= 2:
            print(f"\n  [编码器示例]:")
            # 显示部分类别映射
            print(f"    疾病编码前5个:")
            for i, cls in enumerate(disease_enc.classes_[:5]):
                print(f"      {i}: {cls}")

            print(f"\n    表型编码前5个:")
            for i, cls in enumerate(pheno_enc.classes_[:5]):
                print(f"      {i}: {cls}")

            # 测试编码
            if diseases and len(diseases) > 0:
                test_disease = diseases[0]
                try:
                    encoded = disease_enc.transform([test_disease.lower().strip()])
                    print(f"\n  [编码测试] '{test_disease}' -> {encoded[0]}")
                except:
                    print(f"\n  [编码测试] 警告: 无法编码 '{test_disease}'")

    except Exception as e:
        print(f"  ✗ 编码器构建失败: {e}")
        import traceback
        traceback.print_exc()
        return

    # 4. 制作样本（适配您的文件命名）
    print(f"\n[阶段4] 样本制作:")
    try:
        pos_samples, neg_samples, text_samples, all_samples = make_samples(
            file_paths['positive'],  # pos_datasets.json
            file_paths['negative'],  # neg_datasets.json
            file_paths['text'] if os.path.exists(file_paths['text']) else file_paths['positive']  # 备用
        )

        print(f"  正样本元组: {len(pos_samples)} 个")
        print(f"  负样本元组: {len(neg_samples)} 个")
        print(f"  文本样本元组: {len(text_samples)} 个")
        print(f"  总训练样本: {len(all_samples)} 个")

        if debug_level >= 2 and len(pos_samples) > 0:
            print(f"\n  [正样本示例]:")
            sample, label = pos_samples[0]
            print(f"    标签: {label}")

            # 显示样本内容
            for key, value in sample.items():
                if isinstance(value, list):
                    print(f"    {key}: {value[:5]}{'...' if len(value) > 5 else ''} ({len(value)}个)")
                else:
                    print(f"    {key}: {str(value)[:50]}{'...' if len(str(value)) > 50 else ''}")

            # 检查特征字段是否存在
            required_features = [
                'compound_diseases', 'compound_phenotypes', 'compound_subcellular_locations',
                'protein_diseases', 'protein_phenotypes', 'protein_subcellular_locations'
            ]

            print(f"\n  [特征检查]:")
            for feature in required_features:
                if feature in sample:
                    value = sample[feature]
                    if isinstance(value, list):
                        status = f"存在 ({len(value)}个值)"
                    else:
                        status = f"存在 (类型: {type(value).__name__})"
                    print(f"    ✓ {feature}: {status}")
                else:
                    print(f"    ✗ {feature}: 缺失!")

    except Exception as e:
        print(f"  ✗ 样本制作失败: {e}")
        import traceback
        traceback.print_exc()
        return

    # 5. 创建Dataset
    print(f"\n[阶段5] Dataset创建:")
    try:
        # 划分训练集
        train_samples, val_samples = train_test_split(
            all_samples, test_size=0.2, random_state=seed
        )

        train_dataset = InteractionDataset(
            train_samples, disease_enc, pheno_enc, loc_enc
        )
        val_dataset = InteractionDataset(
            val_samples, disease_enc, pheno_enc, loc_enc
        )
        text_dataset = InteractionDataset(
            text_samples, disease_enc, pheno_enc, loc_enc
        )

        print(f"  训练Dataset大小: {len(train_dataset)}")
        print(f"  验证Dataset大小: {len(val_dataset)}")
        print(f"  文本Dataset大小: {len(text_dataset)}")

        if debug_level >= 2 and len(train_dataset) > 0:
            # 检查一个样本
            print(f"\n  [Dataset样本处理示例]:")
            features, label = train_dataset[0]
            (comp_diseases, comp_phenos, comp_locs,
             prot_diseases, prot_phenos, prot_locs) = features

            print(f"    标签: {label.item()} (1=正样本, 0=负样本)")

            # 显示处理后的索引
            print(f"    代谢物疾病索引: {comp_diseases.tolist()}")
            print(f"    代谢物表型索引: {comp_phenos.tolist()}")
            print(f"    代谢物定位索引: {comp_locs.tolist()}")
            print(f"    蛋白质疾病索引: {prot_diseases.tolist()}")
            print(f"    蛋白质表型索引: {prot_phenos.tolist()}")
            print(f"    蛋白质定位索引: {prot_locs.tolist()}")

            # 解码回文本查看
            def safe_decode(indices, encoder, name):
                if len(indices) > 0:
                    try:
                        decoded = encoder.inverse_transform(indices.numpy())
                        return list(decoded)
                    except:
                        return f"[解码失败: {indices.tolist()}]"
                return ["[空]"]

            print(f"\n    [解码结果]:")
            print(f"      代谢物疾病: {safe_decode(comp_diseases, disease_enc, '疾病')}")
            print(f"      代谢物表型: {safe_decode(comp_phenos, pheno_enc, '表型')}")
            print(f"      代谢物定位: {safe_decode(comp_locs, loc_enc, '定位')}")

    except Exception as e:
        print(f"  ✗ Dataset创建失败: {e}")
        import traceback
        traceback.print_exc()
        return

    # 6. DataLoader测试
    print(f"\n[阶段6] DataLoader测试:")
    try:
        # 自定义collate函数处理变长序列
        def custom_collate_fn(batch):
            """处理变长序列的自定义collate函数"""
            features_batch, labels_batch = zip(*batch)

            # 分离6种特征
            (comp_diseases, comp_phenos, comp_locs,
             prot_diseases, prot_phenos, prot_locs) = zip(*features_batch)

            # 对变长序列进行填充
            from torch.nn.utils.rnn import pad_sequence

            comp_diseases_padded = pad_sequence(comp_diseases, batch_first=True, padding_value=0)
            comp_phenos_padded = pad_sequence(comp_phenos, batch_first=True, padding_value=0)
            comp_locs_padded = pad_sequence(comp_locs, batch_first=True, padding_value=0)
            prot_diseases_padded = pad_sequence(prot_diseases, batch_first=True, padding_value=0)
            prot_phenos_padded = pad_sequence(prot_phenos, batch_first=True, padding_value=0)
            prot_locs_padded = pad_sequence(prot_locs, batch_first=True, padding_value=0)

            labels = torch.cat(labels_batch)

            return (comp_diseases_padded, comp_phenos_padded, comp_locs_padded,
                    prot_diseases_padded, prot_phenos_padded, prot_locs_padded), labels

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=custom_collate_fn
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=custom_collate_fn
        )
        text_loader = DataLoader(
            text_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=custom_collate_fn
        )

        print(f"  训练DataLoader批次数: {len(train_loader)}")
        print(f"  验证DataLoader批次数: {len(val_loader)}")
        print(f"  文本DataLoader批次数: {len(text_loader)}")

        if debug_level >= 3:
            print(f"\n  [第一批数据形状]:")
            for loader_name, loader in [
                ("训练", train_loader),
                ("验证", val_loader),
                ("文本", text_loader)
            ]:
                try:
                    batch = next(iter(loader))
                    features, labels = batch
                    (comp_diseases, comp_phenos, comp_locs,
                     prot_diseases, prot_phenos, prot_locs) = features

                    print(f"\n    {loader_name}Loader:")
                    print(f"      标签: {labels.shape} (值: {labels.tolist()})")
                    print(f"      代谢物疾病: {comp_diseases.shape}")
                    print(f"      代谢物表型: {comp_phenos.shape}")
                    print(f"      代谢物定位: {comp_locs.shape}")
                    print(f"      蛋白质疾病: {prot_diseases.shape}")
                    print(f"      蛋白质表型: {prot_phenos.shape}")
                    print(f"      蛋白质定位: {prot_locs.shape}")

                    # 显示实际内容
                    print(f"      [内容示例] 第一个样本:")
                    print(f"        代谢物疾病: {comp_diseases[0].tolist()}")

                except StopIteration:
                    print(f"\n    {loader_name}Loader: 无数据")
                except Exception as e:
                    print(f"\n    {loader_name}Loader错误: {e}")

    except Exception as e:
        print(f"  ✗ DataLoader测试失败: {e}")
        import traceback
        traceback.print_exc()
        return

    # 7. 数据质量检查
    print(f"\n[阶段7] 数据质量检查:")

    # 检查空数据
    if len(all_samples) > 0:
        empty_counts = {
            'compound_diseases': 0,
            'compound_phenotypes': 0,
            'compound_subcellular_locations': 0,
            'protein_diseases': 0,
            'protein_phenotypes': 0,
            'protein_subcellular_locations': 0,
        }

        check_samples = all_samples[:min(100, len(all_samples))]
        for sample, _ in check_samples:
            for key in empty_counts.keys():
                value = sample.get(key, [])
                if not value or (isinstance(value, list) and len(value) == 0):
                    empty_counts[key] += 1

        print(f"  前{len(check_samples)}个样本的空特征统计:")
        for key, count in empty_counts.items():
            percentage = count / len(check_samples) * 100
            print(f"    {key:35}: {count:3} 个空 ({percentage:.1f}%)")

    # 检查标签分布
    if len(all_samples) > 0:
        positive_count = sum(1 for _, label in all_samples if label == 1)
        negative_count = len(all_samples) - positive_count
        print(f"\n  标签分布:")
        print(f"    正样本 (1): {positive_count} ({positive_count / len(all_samples) * 100:.1f}%)")
        print(f"    负样本 (0): {negative_count} ({negative_count / len(all_samples) * 100:.1f}%)")

        if positive_count == 0 or negative_count == 0:
            print(f"  ⚠️  警告: 数据不平衡!")

    # 8. 快速完整性测试
    print(f"\n[阶段8] 快速完整性测试:")
    try:
        # 修改get_dataloaders函数以适应您的文件结构
        def get_dataloaders_custom(
                batch_size: int = 16,
                val_split: float = 0.2,
                seed: int = 42,
        ):
            set_seed(seed)

            disease_encoder, phenotype_encoder, subcellular_location_encoder = build_label_encoders(
                file_paths['disease'], file_paths['phenotype']
            )

            _, _, text_samples, all_samples = make_samples(
                file_paths['positive'],
                file_paths['negative'],
                file_paths['text'] if os.path.exists(file_paths['text']) else file_paths['positive']
            )

            train_samples, val_samples = train_test_split(all_samples, test_size=val_split, random_state=seed)

            train_dataset = InteractionDataset(train_samples, disease_encoder, phenotype_encoder,
                                               subcellular_location_encoder)
            val_dataset = InteractionDataset(val_samples, disease_encoder, phenotype_encoder,
                                             subcellular_location_encoder)
            text_dataset = InteractionDataset(text_samples, disease_encoder, phenotype_encoder,
                                              subcellular_location_encoder)

            # 使用自定义collate函数
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate_fn)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=custom_collate_fn)
            text_loader = DataLoader(text_dataset, batch_size=batch_size, shuffle=False, collate_fn=custom_collate_fn)

            return {
                'train': train_loader,
                'val': val_loader,
                'text': text_loader,
                'encoders': {
                    'disease': disease_encoder,
                    'phenotype': phenotype_encoder,
                    'subcellular': subcellular_location_encoder,
                }
            }

        # 测试
        loaders = get_dataloaders_custom(
            batch_size=2,
            val_split=0.2,
            seed=seed
        )

        print(f"  ✓ 自定义get_dataloaders函数正常工作")
        print(f"  ✓ 返回字典包含: {list(loaders.keys())}")

        # 测试一个批次
        if len(loaders['train']) > 0:
            train_batch = next(iter(loaders['train']))
            features, labels = train_batch
            print(f"  ✓ 成功加载一个训练批次")
            print(f"     批次大小: {labels.shape[0]}")
            print(f"     标签分布: {labels.tolist()}")

    except Exception as e:
        print(f"  ✗ 完整性测试失败: {e}")
        import traceback
        traceback.print_exc()

    print(f"\n" + "=" * 60)
    print("调试完成!")
    print("=" * 60)

    # 返回调试结果
    return {
        'file_paths': file_paths,
        'encoders': {
            'disease': disease_enc,
            'phenotype': pheno_enc,
            'location': loc_enc
        },
        'datasets': {
            'train': train_dataset,
            'val': val_dataset,
            'text': text_dataset
        },
        'statistics': {
            'total_samples': len(all_samples),
            'positive_samples': positive_count if 'positive_count' in locals() else 0,
            'negative_samples': negative_count if 'negative_count' in locals() else 0,
            'class_ratio': positive_count / len(all_samples) if len(all_samples) > 0 else 0
        }
    }


# 轻量级调试函数（快速检查）
def quick_debug_sample(data_dir="Data", sample_index=0, seed=42):
    """快速调试单个样本的处理过程（适配您的文件结构）"""
    set_seed(seed)

    print("快速调试 - 单个样本处理流程（适配版）")
    print("-" * 50)

    # 构建文件路径
    disease_path = os.path.join(data_dir, "disease_list.json")
    phenotype_path = os.path.join(data_dir, "phenotype.json")
    positive_path = os.path.join(data_dir, "pos_datasets.json")

    if not all(os.path.exists(p) for p in [disease_path, phenotype_path, positive_path]):
        print("文件不存在，请检查路径:")
        print(f"  疾病文件: {disease_path} {'✓' if os.path.exists(disease_path) else '✗'}")
        print(f"  表型文件: {phenotype_path} {'✓' if os.path.exists(phenotype_path) else '✗'}")
        print(f"  正样本文件: {positive_path} {'✓' if os.path.exists(positive_path) else '✗'}")
        return

    # 1. 加载一个正样本
    try:
        positives = _load_json(positive_path)
        print(f"[文件加载] 正样本数量: {len(positives)}")

        if len(positives) == 0:
            print("错误: 正样本文件为空!")
            return

        if sample_index >= len(positives):
            print(f"样本索引 {sample_index} 超出范围 (0-{len(positives) - 1})")
            sample_index = 0

        sample = positives[sample_index]
        print(f"\n[原始样本 {sample_index}]:")
        print(f"  类型: {type(sample)}")

        if isinstance(sample, dict):
            # 显示所有键
            print(f"  包含的键: {list(sample.keys())}")

            # 显示关键信息
            for key in ['compound_id', 'compound_name', 'protein_id', 'protein_name']:
                if key in sample:
                    print(f"  {key}: {sample[key]}")

            # 显示特征
            feature_keys = [k for k in sample.keys() if 'diseases' in k or 'phenotypes' in k or 'locations' in k]
            for key in feature_keys:
                value = sample[key]
                if isinstance(value, list):
                    print(f"  {key}: {value[:3]}{'...' if len(value) > 3 else ''} ({len(value)}个)")
                else:
                    print(f"  {key}: {value}")
        else:
            print(f"  样本不是字典格式: {type(sample)}")
            return

    except Exception as e:
        print(f"文件加载失败: {e}")
        return

    # 2. 构建编码器
    try:
        disease_enc, pheno_enc, loc_enc = build_label_encoders(
            disease_path, phenotype_path
        )
        print(f"\n[编码器构建] 成功")
        print(f"  疾病类别数: {len(disease_enc.classes_)}")
        print(f"  表型类别数: {len(pheno_enc.classes_)}")

    except Exception as e:
        print(f"编码器构建失败: {e}")
        return

    # 3. 模拟Dataset处理
    try:
        mock_samples = [(sample, 1)]  # 标签为1
        dataset = InteractionDataset(mock_samples, disease_enc, pheno_enc, loc_enc)

        # 4. 获取处理后的数据
        features, label = dataset[0]
        (comp_diseases, comp_phenos, comp_locs,
         prot_diseases, prot_phenos, prot_locs) = features

        print(f"\n[处理后结果]:")
        print(f"  标签: {label.item()} (1=正样本)")

        # 解码展示函数
        def decode_indices(indices, encoder, feature_name):
            if len(indices) > 0:
                try:
                    decoded = encoder.inverse_transform(indices.numpy())
                    return list(decoded)
                except Exception as e:
                    return f"[解码错误: {e}]"
            return ["[空]"]

        # 显示编码结果
        print(f"\n  [编码索引]:")
        print(f"    代谢物疾病: {comp_diseases.tolist()}")
        print(f"    代谢物表型: {comp_phenos.tolist()}")
        print(f"    代谢物定位: {comp_locs.tolist()}")
        print(f"    蛋白质疾病: {prot_diseases.tolist()}")
        print(f"    蛋白质表型: {prot_phenos.tolist()}")
        print(f"    蛋白质定位: {prot_locs.tolist()}")

        print(f"\n  [解码回文本]:")
        print(f"    代谢物疾病: {decode_indices(comp_diseases, disease_enc, '疾病')}")
        print(f"    代谢物表型: {decode_indices(comp_phenos, pheno_enc, '表型')}")
        print(f"    代谢物定位: {decode_indices(comp_locs, loc_enc, '定位')}")
        print(f"    蛋白质疾病: {decode_indices(prot_diseases, disease_enc, '疾病')}")
        print(f"    蛋白质表型: {decode_indices(prot_phenos, pheno_enc, '表型')}")
        print(f"    蛋白质定位: {decode_indices(prot_locs, loc_enc, '定位')}")

        return dataset[0]

    except Exception as e:
        print(f"数据处理失败: {e}")
        import traceback
        traceback.print_exc()
        return None


# 新增：文件内容检查工具
def check_file_contents(data_dir="Data"):
    """检查每个文件的具体内容格式"""
    print("=" * 60)
    print("文件内容格式检查")
    print("=" * 60)

    files = {
        "疾病列表": "disease_list.json",
        "表型列表": "phenotype.json",
        "正样本": "pos_datasets.json",
        "负样本": "neg_datasets.json",
        "文本数据": "text_data.json",
        "预测数据": "predict_datasets.json"
    }

    for name, filename in files.items():
        path = os.path.join(data_dir, filename)
        print(f"\n[{name}] {filename}:")

        if not os.path.exists(path):
            print(f"  ✗ 文件不存在")
            continue

        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            print(f"  ✓ 文件格式: {type(data).__name__}")

            if isinstance(data, list):
                print(f"     条目数: {len(data)}")
                if len(data) > 0:
                    first_item = data[0]
                    print(f"     第一条类型: {type(first_item).__name__}")

                    if isinstance(first_item, dict):
                        print(f"     包含的键: {list(first_item.keys())[:8]}{'...' if len(first_item) > 8 else ''}")

                        # 检查关键字段
                        key_check = {
                            'diseases': ['compound_diseases', 'protein_diseases'],
                            'phenotypes': ['compound_phenotypes', 'protein_phenotypes'],
                            'locations': ['compound_subcellular_locations', 'protein_subcellular_locations']
                        }

                        for category, fields in key_check.items():
                            found = []
                            for field in fields:
                                if field in first_item:
                                    value = first_item[field]
                                    if isinstance(value, list):
                                        found.append(f"{field}({len(value)})")
                                    else:
                                        found.append(f"{field}(类型:{type(value).__name__})")

                            if found:
                                print(f"     {category}: {', '.join(found)}")

            elif isinstance(data, dict):
                print(f"     键数量: {len(data)}")
                print(f"     前5个键: {list(data.keys())[:5]}")

        except json.JSONDecodeError as e:
            print(f"  ✗ JSON解析错误: {e}")
        except Exception as e:
            print(f"  ✗ 读取错误: {e}")
