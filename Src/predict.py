import yaml
import json
import pandas as pd
import numpy as np
import torch
from data_loader import get_dataloaders, extract_names_from_text_json, set_seed
from model import InteractionPredictionModel_NoAttention, InteractionPredictionModel


def predict(model, dataloader):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    scores = []
    with torch.no_grad():
        for inputs, _ in dataloader:
            (cd, cp, cs, pd, pp, ps) = inputs
            cd, cp, cs, pd, pp, ps = cd.to(device), cp.to(device), cs.to(device), pd.to(device), pp.to(device), ps.to(device)
            logits = model(cd, cp, cs, pd, pp, ps)
            probs = torch.sigmoid(logits).cpu().numpy().ravel()
            scores.extend(probs.tolist())
    return np.array(scores)


def save_topk_per_compound(scores, protein_names, compound_names, output_path, topk=20):
    """
    使用pandas的简洁版本
    """
    # 创建DataFrame
    df = pd.DataFrame({
        'Compound': compound_names,
        'Protein': protein_names,
        'Score': scores
    })

    # 按化合物分组，对每个化合物按分数排序
    results = {}

    # 使用groupby处理
    for compound, group in df.groupby('Compound'):
        # 排序并取TopK
        sorted_group = group.sort_values('Score', ascending=False).head(topk)

        results[compound] = {
            "Protein Names": sorted_group['Protein'].tolist(),
            "Prediction Scores": sorted_group['Score'].tolist(),
            "Score Type": "sigmoid_probability"
        }

    # 保存
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=4)

    print(f"已保存 {len(results)} 个化合物的Top{topk}结果到: {output_path}")
    return results


def DeepETD_predict(config_path='predict_config.yaml'):
    """

    :param config_path:
    :return:
    """
    # 加载配置
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)

    # 设置随机种子
    set_seed(cfg['prediction']['seed'])

    # 获取数据加载器
    loaded = get_dataloaders(
        disease_json_path=cfg['data']['disease_json'],
        phenotype_json_path=cfg['data']['phenotype_json'],
        positive_json_path=cfg['data']['positive_json'],
        negative_json_path=cfg['data']['negative_json'],
        text_json_path=cfg['data']['text_json'],
        batch_size=cfg['prediction']['batch_size'],
        val_split=cfg['prediction']['val_split'],
        seed=cfg['prediction']['seed'],
    )

    enc = loaded['encoders']
    model_params = cfg['model']['params'].copy()
    model_params.update({
        'num_diseases': len(enc['disease'].classes_),
        'num_phenotypes': len(enc['phenotype'].classes_),
        'num_subcellular_locations': len(enc['subcellular'].classes_),
    })

    model_class = InteractionPredictionModel if cfg['model']['use_attention'] \
        else InteractionPredictionModel_NoAttention
    model = model_class(**model_params)

    state = torch.load(cfg['model']['checkpoint'], map_location='cpu')
    model.load_state_dict(state)

    layer_stats = diagnose_model_issues(model, loaded['text'], loaded['encoders'])

    protein_names, compound_names = extract_names_from_text_json(cfg['data']['text_json'])

    scores = predict(model, loaded['text'])

    n = min(len(scores), len(protein_names), len(compound_names))
    save_topk_per_compound(
        scores[:n],
        protein_names[:n],
        compound_names[:n],
        cfg['prediction']['output_file'],
        topk=cfg['prediction']['topk']
    )


def diagnose_model_issues(model, dataloader, encoders):
    """
    诊断模型问题的根本原因
    """
    print("🔍 开始模型诊断...")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    # 1. 检查模型参数
    print("\n1. 模型参数检查:")
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   总参数: {total_params:,}")
    print(f"   可训练参数: {trainable_params:,}")
    print(f"   冻结参数: {total_params - trainable_params:,}")

    # 检查梯度
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"   {name}: shape={param.shape}, grad={param.grad is not None}")

    # 2. 检查输入数据
    print("\n2. 输入数据检查:")
    for i, (inputs, labels) in enumerate(dataloader):
        if i >= 1:  # 只检查第一个批次
            break

        (cd, cp, cs, pd, pp, ps) = inputs

        print(f"   批次 {i}:")
        print(f"     cd形状: {cd.shape}, 范围: [{cd.min():.3f}, {cd.max():.3f}]")
        print(f"     cd唯一值: {torch.unique(cd)}")
        print(f"     标签分布: 正={torch.sum(labels).item()}, 负={len(labels) - torch.sum(labels).item()}")

        # 检查是否有NaN或Inf
        for name, tensor in [("cd", cd), ("cp", cp), ("cs", cs),
                             ("pd", pd), ("pp", pp), ("ps", ps)]:
            if torch.isnan(tensor).any():
                print(f"    ⚠️ {name} 包含NaN!")
            if torch.isinf(tensor).any():
                print(f"    ⚠️ {name} 包含无穷值!")

    # 3. 检查编码器
    print("\n3. 编码器检查:")
    print(f"   疾病编码器类别数: {len(encoders['disease'].classes_)}")
    print(f"   表型编码器类别数: {len(encoders['phenotype'].classes_)}")
    print(f"   亚细胞定位编码器类别数: {len(encoders['subcellular'].classes_)}")

    # 4. 前向传播测试
    print("\n4. 前向传播测试:")
    model.eval()
    with torch.no_grad():
        for i, (inputs, _) in enumerate(dataloader):
            if i >= 2:
                break

            (cd, cp, cs, pd, pp, ps) = inputs
            cd, cp, cs, pd, pp, ps = cd.to(device), cp.to(device), cs.to(device), pd.to(device), pp.to(device), ps.to(
                device)

            # 逐层检查
            logits = model(cd, cp, cs, pd, pp, ps)
            probs = torch.sigmoid(logits)

            print(f"   批次 {i}:")
            print(f"     输入形状: cd={cd.shape}")
            print(f"     输出logits: {logits}")
            print(f"     预测概率: {probs}")

            # 检查输出是否相同
            if i == 0:
                first_logits = logits
            else:
                if torch.allclose(first_logits, logits, rtol=1e-3):
                    print("    ⚠️ 不同批次的输出完全相同！")

    # 5. 检查模型权重
    print("\n5. 模型权重检查:")
    layer_stats = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            stats = {
                'name': name,
                'mean': param.mean().item(),
                'std': param.std().item(),
                'min': param.min().item(),
                'max': param.max().item(),
                'zero_ratio': (param == 0).sum().item() / param.numel()
            }
            layer_stats.append(stats)

            if stats['std'] < 1e-6:
                print(f"    ⚠️ {name}: 权重标准差太小 ({stats['std']:.6f})")
            if stats['zero_ratio'] > 0.9:
                print(f"    ⚠️ {name}: 超过90%的权重为0")

    return layer_stats


# 使用诊断函数


if __name__ == '__main__':
    DeepETD_predict(config_path='config.yaml')

