"""
================================================================================
训练循环 + 损失函数
================================================================================
六种损失的组合训练 + 早停 + 评估
================================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from sklearn.metrics import roc_auc_score, accuracy_score
import time
import copy

from config import (
    DEVICE, EPOCHS, LR, WEIGHT_DECAY, EARLY_STOP_PATIENCE, SEED,
    LAMBDA_RISK, LAMBDA_GRADE, LAMBDA_CONTRAST, LAMBDA_GAMMA_REG,
)
from layer2_encoder import DualOutputEncoder
from layer3_gamma import CrossRelationPropagation
from layer4_temporal import TemporalEncoder
from classifier import MultiTaskHeads


class RiskWhiteListModel(nn.Module):
    """
    ==========================================================================
    完整模型：Layer 2 + Layer 3 + Layer 4 + 预测头
    ==========================================================================
    """
    def __init__(self, in_dims: dict, edge_types: list):
        super().__init__()
        self.encoder = DualOutputEncoder(in_dims, edge_types)
        # 提取所有唯一的边类型名称
        edge_names = list(set(et[1] for et in edge_types))
        self.edge_names = edge_names
        self.gamma_module = CrossRelationPropagation(edge_names=edge_names)
        self.temporal = TemporalEncoder(
            input_dim=self.encoder.hidden * (self.encoder.num_layers + 1)
        )
        # h_v(384) + h_risk(128) = 512
        fusion_dim = self.encoder.hidden * self.encoder.num_layers + self.encoder.hidden
        self.heads = MultiTaskHeads(in_dim=self.temporal.gru.hidden_size)
        self.fusion_dim = fusion_dim

    def forward(self, x_dict: dict, edge_index_dict: dict, num_enterprises: int,
                x_struct: dict = None, x_missing: dict = None, struct_hint: dict = None):
        # Layer 2: 双通道编码（含自适应门控阀）
        h_v, m_v_r = self.encoder(x_dict, edge_index_dict,
                                  x_struct=x_struct, x_missing=x_missing,
                                  struct_hint=struct_hint)
        # Layer 3: 跨关系风险传播
        s_v, h_risk, gamma = self.gamma_module(m_v_r, edge_index_dict, num_enterprises)
        # 融合
        h_fusion = torch.cat([h_v, h_risk], dim=-1)  # (N, 384+128)
        # Layer 4: 时序编码（假数据模式：重复当前月）
        z_v = self.temporal(h_fusion)
        # 预测头
        logit_white, logit_risk, logit_grade = self.heads(z_v)
        return logit_white, logit_risk, logit_grade, gamma, h_fusion


def compute_losses(logit_white, logit_risk, logit_grade,
                   y_white, y_risk, y_grade, mask, gamma, h_fusion):
    """
    ==========================================================================
    计算所有损失。

    L_total = L_white + λ1·L_risk + λ2·L_grade
            + λ3·L_contrastive + λ5·||Γ - I||_F

    注意：L_distill 在 Layer 7 蒸馏阶段单独计算，训练 GNN 时不包含
    ==========================================================================
    """
    # 主监督：白名单预测（带类平衡权重）
    n_pos = y_white[mask].sum()
    n_neg = mask.sum() - n_pos
    pos_weight = torch.tensor([n_neg / n_pos]).clamp(max=100.0).to(logit_white.device)
    L_white = F.binary_cross_entropy_with_logits(
        logit_white[mask].squeeze(-1), y_white[mask].float(),
        pos_weight=pos_weight,
    )
    # 辅助：风险预测
    L_risk = F.binary_cross_entropy_with_logits(
        logit_risk[mask].squeeze(-1), y_risk[mask].float()
    )
    # 辅助：企业分级
    L_grade = F.cross_entropy(logit_grade[mask], y_grade[mask])

    # 对比学习：InfoNCE（同类靠近、异类远离）
    h_norm = F.normalize(h_fusion[mask], dim=-1)
    sim = h_norm @ h_norm.T  # 相似度矩阵
    y_bin = y_white[mask].float()
    pos_mask = (y_bin.unsqueeze(0) == y_bin.unsqueeze(1)).float()
    pos_mask.fill_diagonal_(0)
    L_contrast = - (pos_mask * F.log_softmax(sim, dim=-1)).sum(dim=-1).mean()

    # Γ 正则化：趋向单位矩阵（防过拟合退化）
    L_gamma_reg = ((gamma - torch.eye(gamma.shape[0], device=gamma.device)) ** 2).mean()

    L_total = (L_white
               + LAMBDA_RISK * L_risk
               + LAMBDA_GRADE * L_grade
               + LAMBDA_CONTRAST * L_contrast
               + LAMBDA_GAMMA_REG * L_gamma_reg)

    loss_dict = {
        "total": L_total.item(), "white": L_white.item(),
        "risk": L_risk.item(), "grade": L_grade.item(),
        "contrast": L_contrast.item(), "gamma_reg": L_gamma_reg.item(),
    }
    return L_total, loss_dict


@torch.no_grad()
def evaluate_model(model, data, mask):
    """评估：AUC + Accuracy"""
    model.eval()
    logit_w, logit_r, logit_g, gamma, _ = model(
        data.x_dict, data.edge_index_dict, data.num_enterprises,
        x_struct=data.x_struct, x_missing=data.x_missing,
        struct_hint=data.struct_hint,
    )
    prob_w = torch.sigmoid(logit_w[mask]).cpu().squeeze(-1)
    y_true = data.y_white[mask].cpu()

    auc = roc_auc_score(y_true, prob_w) if y_true.sum() > 0 and (1-y_true).sum() > 0 else 0.5
    acc = accuracy_score(y_true, (prob_w >= 0.5).int())
    return auc, acc, gamma


def train(model, data, epochs: int = EPOCHS):
    """完整训练循环"""
    device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")
    print(f"  设备: {device}")

    model = model.to(device)
    data = data.to(device)

    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    best_auc = 0
    best_state = None
    patience = EARLY_STOP_PATIENCE

    print(f"\n  开始训练 ({epochs} epochs, 早停={patience})...")
    t0 = time.time()

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()

        logit_w, logit_r, logit_g, gamma, h_f = model(
            data.x_dict, data.edge_index_dict, data.num_enterprises,
            x_struct=data.x_struct, x_missing=data.x_missing,
            struct_hint=data.struct_hint,
        )
        loss, loss_dict = compute_losses(
            logit_w, logit_r, logit_g,
            data.y_white, data.y_risk, data.y_grade,
            data.train_mask, gamma, h_f,
        )
        loss.backward()
        optimizer.step()

        # 验证集评估
        val_auc, val_acc, _ = evaluate_model(model, data, data.val_mask)
        if val_auc > best_auc:
            best_auc = val_auc
            best_state = copy.deepcopy(model.state_dict())
            patience = EARLY_STOP_PATIENCE
        else:
            patience -= 1

        if (epoch + 1) % 50 == 0 or patience == 0:
            print(f"  Epoch {epoch+1:3d} | Loss {loss_dict['total']:.4f} "
                  f"| Val AUC {val_auc:.4f} | Best {best_auc:.4f}")

        if patience <= 0:
            print(f"  早停于 epoch {epoch+1}")
            break

    elapsed = time.time() - t0
    print(f"  训练完成, {elapsed:.1f}s, 最佳验证 AUC={best_auc:.4f}")

    model.load_state_dict(best_state)
    return model
