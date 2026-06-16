"""
================================================================================
训练循环 + 损失函数
================================================================================
v4.1: 手动 mini-batch 训练（2-hop 子图采样），零额外依赖，CPU 友好。
每批采样 ~512 种子节点 + 2 跳邻居 → 前向 < 300 MB，反向 < 500 MB。
================================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import roc_auc_score, accuracy_score
import time
import copy
import gc

from config import (
    DEVICE, EPOCHS, LR, WEIGHT_DECAY, EARLY_STOP_PATIENCE, SEED,
    LAMBDA_RISK, LAMBDA_GRADE, LAMBDA_CONTRAST, LAMBDA_GAMMA_REG,
    BATCH_SIZE,
)
from layer2_encoder import DualOutputEncoder
from layer3_gamma import CrossRelationPropagation
from layer4_temporal import TemporalEncoder
from classifier import MultiTaskHeads
from batch_sampler import sample_subgraph


class RiskWhiteListModel(nn.Module):
    """
    ==========================================================================
    完整模型：Layer 2(SAGE) + Layer 3(Γ) + Layer 4(GRU) + 预测头

    维度流（v4.1）：
      h_v(SAGE last layer, 128) + h_risk(Γ, 128) = h_fusion(256)
      → Temporal(256→64→GRU(64,64)) → Z_v(64)
      → MultiTaskHeads(64→64→1/1/5)
    ==========================================================================
    """
    def __init__(self, in_dims: dict, edge_types: list):
        super().__init__()
        self.encoder = DualOutputEncoder(in_dims, edge_types)
        # 提取所有唯一的边类型名称
        edge_names = list(set(et[1] for et in edge_types))
        self.edge_names = edge_names
        self.gamma_module = CrossRelationPropagation(edge_names=edge_names)
        # Lazy init：input_dim 传 None，首次 forward 自动适配 h_fusion 真实维度
        self.temporal = TemporalEncoder(input_dim=None)
        # h_v(128) + h_risk(128) = 256
        self.fusion_dim = self.encoder.hidden + self.encoder.hidden
        self.heads = MultiTaskHeads(in_dim=self.temporal.hidden)

    def forward(self, x_dict: dict, edge_index_dict: dict, num_enterprises: int,
                x_struct: dict = None, x_missing: dict = None, struct_hint: dict = None,
                x_seq: torch.Tensor = None):
        # Layer 2: 双通道编码（含自适应门控阀）
        h_v, m_v_r = self.encoder(x_dict, edge_index_dict,
                                  x_struct=x_struct, x_missing=x_missing,
                                  struct_hint=struct_hint)
        # Layer 3: 跨关系风险传播
        s_v, h_risk, gamma = self.gamma_module(m_v_r, edge_index_dict, num_enterprises)
        # 融合
        h_fusion = torch.cat([h_v, h_risk], dim=-1)  # (N, 256)
        # Layer 4: 时序编码（v4.3: GRU + MLP + Temporal Gate）
        z_v = self.temporal(h_fusion, x_seq=x_seq)
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
    n_pos_per_sample = pos_mask.sum(dim=-1).clamp(min=1)
    L_contrast = - (pos_mask * F.log_softmax(sim, dim=-1)).sum(dim=-1) / n_pos_per_sample
    L_contrast = L_contrast.mean()

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
        x_seq=data.x_seq if hasattr(data, "x_seq") and data.x_seq is not None else None,
    )
    prob_w = torch.sigmoid(logit_w[mask]).cpu().squeeze(-1)
    y_true = data.y_white[mask].cpu()

    auc = roc_auc_score(y_true, prob_w) if y_true.sum() > 0 and (1-y_true).sum() > 0 else 0.5
    acc = accuracy_score(y_true, (prob_w >= 0.5).int())
    return auc, acc, gamma


def train(model, data, epochs: int = EPOCHS):
    """Mini-batch 训练循环（手动 2-hop 子图采样）"""
    device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")
    print(f"  设备: {device}")
    print(f"  Mini-batch: batch_size={BATCH_SIZE}, 2-hop 子图采样")

    model = model.to(device)
    data = data.to(device)

    # 训练节点列表
    train_ent_idx = torch.where(data.train_mask)[0]
    steps_per_epoch = max(1, len(train_ent_idx) // BATCH_SIZE)
    print(f"  每 epoch ~{steps_per_epoch} 步")

    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.5,
                                   patience=15, min_lr=1e-6)
    best_auc = 0
    best_state = None
    patience = EARLY_STOP_PATIENCE

    print(f"\n  开始训练 ({epochs} epochs, 早停={patience})...")
    print(f"  {'Epoch':>5s} | {'Loss':>6s} {'W':>6s} | "
          f"{'α':>5s} {'Γdiag':>6s} | {'ValAUC':>7s} {'Best':>7s}")
    print(f"  {'─'*5:>5s}─┼{'─'*7:>7s}─┼{'─'*7:>7s}─┼{'─'*7:>7s}─┼{'─'*7:>7s}")
    t0 = time.time()

    for epoch in range(epochs):
        model.train()

        # 随机打乱训练节点
        perm = torch.randperm(len(train_ent_idx), device=device)
        shuffled = train_ent_idx[perm]

        total_loss = 0.0
        total_white = 0.0
        n_steps = 0

        for start in range(0, len(shuffled), BATCH_SIZE):
            seed_nodes = shuffled[start:start + BATCH_SIZE]
            if seed_nodes.numel() == 0:
                continue

            # ── 2-hop 子图采样 ──
            batch = sample_subgraph(data, seed_nodes, num_hops=2)
            if batch is None or batch["n_ent"] == 0:
                continue

            # ── 门控阀数据：用 nid 从全局 tensor 切片 ──
            nid = batch["nid"]
            x_struct_batch = {}
            x_missing_batch = {}
            struct_hint_batch = {}
            if data.x_struct:
                for nt, xs in data.x_struct.items():
                    if nt == "enterprise":
                        x_struct_batch[nt] = xs[nid]
            if data.x_missing:
                for nt, xm in data.x_missing.items():
                    if nt == "enterprise":
                        x_missing_batch[nt] = xm[nid]
            if data.struct_hint:
                for nt, sh in data.struct_hint.items():
                    if nt == "enterprise":
                        struct_hint_batch[nt] = sh[nid]

            # ── 季度序列切片 ──
            x_seq_batch = None
            if data.x_seq is not None:
                x_seq_batch = data.x_seq[nid]  # (B, 4, DIM+1)

            # ── 前向 ──
            optimizer.zero_grad()
            logit_w, logit_r, logit_g, gamma, h_f = model(
                batch["x_dict"], batch["edge_index_dict"], batch["n_ent"],
                x_struct=x_struct_batch if x_struct_batch else None,
                x_missing=x_missing_batch if x_missing_batch else None,
                struct_hint=struct_hint_batch if struct_hint_batch else None,
                x_seq=x_seq_batch,
            )

            # ── 损失（仅种子节点） ──
            loss, loss_dict = compute_losses(
                logit_w, logit_r, logit_g,
                data.y_white[nid], data.y_risk[nid], data.y_grade[nid],
                batch["seed_mask"], gamma, h_f,
            )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss_dict["total"]
            total_white += loss_dict["white"]
            n_steps += 1

            # 释放批次中间量
            del batch, nid, x_struct_batch, x_missing_batch, struct_hint_batch, x_seq_batch
            del logit_w, logit_r, logit_g, gamma, h_f

        # 周期性 gc
        if (epoch + 1) % 5 == 0:
            gc.collect()

        avg_loss = total_loss / max(n_steps, 1)
        avg_white = total_white / max(n_steps, 1)

        # ── 验证集评估（全量、无 grad） ──
        val_auc, val_acc, gamma_val = evaluate_model(model, data, data.val_mask)
        scheduler.step(val_auc)  # Val AUC 不涨时 LR 自动减半
        if val_auc > best_auc:
            best_auc = val_auc
            best_state = copy.deepcopy(model.state_dict())
            patience = EARLY_STOP_PATIENCE
        else:
            patience -= 1

        # ── 诊断输出（每 10 轮或早停触发） ──
        if (epoch + 1) % 10 == 0 or patience == 0:
            alpha_val = 0.0
            if hasattr(model.encoder, "gates") and model.encoder.gates is not None:
                gate_ent = model.encoder.gates["enterprise"] if "enterprise" in model.encoder.gates else None
                if gate_ent is not None and hasattr(gate_ent, "last_alpha_mean"):
                    alpha_val = gate_ent.last_alpha_mean
            gamma_diag_val = gamma_val.diag().mean().item()

            print(f"  {epoch+1:5d} | {avg_loss:6.4f} {avg_white:6.4f} | "
                  f"{alpha_val:5.3f} {gamma_diag_val:6.3f} | {val_auc:7.4f} {best_auc:7.4f}")

        if patience <= 0:
            print(f"  早停于 epoch {epoch+1}")
            break

    elapsed = time.time() - t0
    print(f"  训练完成, {elapsed:.1f}s, 最佳验证 AUC={best_auc:.4f}")

    model.load_state_dict(best_state)
    return model
