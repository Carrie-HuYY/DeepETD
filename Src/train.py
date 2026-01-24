from datetime import datetime
import numpy as np
import os
import json
from types import SimpleNamespace
from scipy.special import expit
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import roc_auc_score, accuracy_score

from data_loader import get_dataloaders, set_seed
from model import InteractionPredictionModel_NoAttention, InteractionPredictionModel

import warnings
warnings.filterwarnings("ignore")


def log_init():
    """
    初始化日志文档
    :return: 返回日志的路径及文件名
    """

    log_dir = "Log"
    os.makedirs(log_dir, exist_ok=True)
    log_filename = f"training_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"

    return os.path.join(log_dir, log_filename)


def log_write(logfile, msg):
    """
    将时间写入日志
    :param logfile: 日志文件名
    :param msg: 日志路径
    :return:
    """

    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    log_dir = os.path.dirname(logfile)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)

    with open(logfile, 'a', encoding='utf-8') as f:
        f.write(f"[{ts}] {msg}\n")


def load_config(config_path='config.json'):
    """
    加载JSON配置文件
    """

    with open(config_path, 'r', encoding='utf-8') as f:
        config_dict = json.load(f)

    # 转换为对象（更易访问）
    def dict_to_obj(d):
        if isinstance(d, dict):
            return SimpleNamespace(**{k: dict_to_obj(v) for k, v in d.items()})
        elif isinstance(d, list):
            return [dict_to_obj(item) for item in d]
        else:
            return d

    return dict_to_obj(config_dict)


def train_model(train_loader,
                val_loader,
                model,
                optimizer,
                pos_weight=1.0,
                epochs=10,
                patience=3,
                seed=42,
                model_save_path="Result/best_model.pth"):
    """
    用于训练DeepETD模型

    :param train_loader: 训练数据加载器
    :param val_loader: 验证数据加载器
    :param model: 待训练的深度学习模型
    :param optimizer: 优化器
    :param pos_weight: 正样本权重，用于处理类别不平衡问题，默认为 1
    :param epochs: 最大训练轮数，默认为 10
    :param patience: 早停耐心值（多少轮无改善后停止），默认为 3
    :param seed: 随机种子，确保可重复性，默认为 42
    :param model_save_path: 最佳模型保存路径

    :return: 无返回值，但会保存最佳模型到指定路径
    """

    logfile = log_init()

    set_seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], device=device))

    best_auc = 0.0
    patience_counter = 0

    log_write(logfile, "🚀 Starting training...")

    for epoch in range(epochs):
        log_write(logfile, f"\n🔁 Epoch {epoch+1}/{epochs}")
        model.train()
        running_loss = 0.0
        all_y = []
        all_logits = []

        for i, (inputs, labels) in enumerate(train_loader):
                (cd, cp, cs, pd, pp, ps) = inputs
                cd, cp, cs, pd, pp, ps = cd.to(device), cp.to(device), cs.to(device), pd.to(device), pp.to(device), ps.to(device)
                labels = labels.to(device).float()

                logits = model(cd, cp, cs, pd, pp, ps)
                loss = criterion(logits, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                all_y.append(labels.detach().cpu().numpy())
                all_logits.append(logits.detach().cpu().numpy())

        all_y = np.concatenate(all_y).ravel()
        all_probs = expit(np.concatenate(all_logits).ravel())
        train_auc = roc_auc_score(all_y, all_probs)
        train_acc = accuracy_score(all_y, (all_probs > 0.5).astype(int))
        avg_train_loss = running_loss / max(1, len(train_loader))
        log_write(logfile, f"✅ Train | Loss: {avg_train_loss:.4f} | AUC: {train_auc:.4f} | Acc: {train_acc:.4f}")

        model.eval()
        v_loss = 0.0
        vy, vlogits = [], []
        with torch.no_grad():
            for i, (inputs, labels) in enumerate(val_loader):
                (cd, cp, cs, pd, pp, ps) = inputs
                cd, cp, cs, pd, pp, ps = cd.to(device), cp.to(device), cs.to(device), pd.to(device), pp.to(device), ps.to(device)
                labels = labels.to(device).float()

                logits = model(cd, cp, cs, pd, pp, ps)
                loss = criterion(logits, labels)

                v_loss += loss.item()
                vy.append(labels.cpu().numpy())
                vlogits.append(logits.cpu().numpy())

        vy = np.concatenate(vy).ravel()
        vprobs = expit(np.concatenate(vlogits).ravel())
        val_auc = roc_auc_score(vy, vprobs)
        val_acc = accuracy_score(vy, (vprobs > 0.5).astype(int))
        avg_val_loss = v_loss / max(1, len(val_loader))

        log_write(logfile, f"🧪 Val   | Loss: {avg_val_loss:.4f} | AUC: {val_auc:.4f} | Acc: {val_acc:.4f}")

        if val_auc > best_auc:
            best_auc = val_auc
            torch.save(model.state_dict(), model_save_path)
            patience_counter = 0
            log_write(logfile, f"💾 New best (AUC={val_auc:.4f}). Saved to {model_save_path}")
        else:
            patience_counter += 1
            log_write(logfile, f"⚠️  No improvement. Patience {patience_counter}/{patience}")

        if patience_counter >= patience:
            log_write(logfile, f"⏹️  Early stopping at epoch {epoch+1}")
            break

    log_write(logfile, f"🎉 Training completed. Best Val AUC: {best_auc:.4f}")


import yaml


def DeepETD_Train(config_path='config.yaml'):
    """
    从配置文件运行完整训练流程
    """

    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)

    loaders = get_dataloaders(
        disease_json_path=cfg['data']['disease_json'],
        phenotype_json_path=cfg['data']['phenotype_json'],
        positive_json_path=cfg['data']['positive_json'],
        negative_json_path=cfg['data']['negative_json'],
        text_json_path=cfg['data']['text_json'],
        batch_size=cfg['training']['batch_size'],
        val_split=cfg['training']['val_split'],
        seed=cfg['training']['seed'],
    )

    enc = loaders['encoders']

    model_params = cfg['model']['params'].copy()
    model_params.update({
        'num_diseases': len(enc['disease'].classes_),
        'num_phenotypes': len(enc['phenotype'].classes_),
        'num_subcellular_locations': len(enc['subcellular'].classes_),
    })

    model_class = InteractionPredictionModel if cfg['model']['use_attention'] \
        else InteractionPredictionModel_NoAttention
    model = model_class(**model_params)

    train_cfg = cfg['training']
    optimizer = optim.AdamW(
        model.parameters(),
        lr=train_cfg['lr'],
        weight_decay=train_cfg['weight_decay']
    )

    train_model(
        loaders['train'],
        loaders['val'],
        model,
        optimizer,
        pos_weight=train_cfg['pos_weight'],
        epochs=train_cfg['epochs'],
        patience=train_cfg['patience'],
        seed=train_cfg['seed'],
        model_save_path=cfg['model']['out_path'],
    )


if __name__ == "__main__":
    DeepETD_Train(config_path='config.yaml')