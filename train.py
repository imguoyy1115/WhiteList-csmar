"""
================================================================================
训练循环 — 超图异构双通道模型 v5
================================================================================
内存优化版：
  - AMP 混合精度（显存砍 ~40%）
  - 单次全批量前向（去掉假 mini-batch 循环，原来每 epoch 算 6 遍）
  - 每 epoch 强制清理 CUDA cache + GC
  - HeteroConv 层数减为 1（6 边 SAGEConv 的中间张量是最大内存杀手）
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
    DEVICE, EPOCHS, LR, LR_HYPER, WEIGHT_DECAY, EARLY_STOP_PATIENCE, SEED,
    LAMBDA_RISK, LAMBDA_GRADE, LAMBDA_GAMMA_REG, LAMBDA_STRUCT,
    HIDDEN_DIM, HYPER_HIDDEN, FUSION_HIDDEN, USE_AMP,
)
from hypergraph.hypergraph_conv import MultiViewHyperEncoder
from heterogeneous.hetero_encoder import HeteroChannelEncoder
from fusion.fusion_gate import FusionGate
from layers.feature_gate import AdaptiveFeatureGate
from layers.layer3_gamma import CrossRelationPropagation
from layers.layer4_temporal import TemporalEncoder
from classifier import MultiTaskHeads


class HyperHeteroModel(nn.Module):
    """
    ==========================================================================
    v5 完整模型：超图异构双通道 + Γ 风险传播 + GRU 时序

    数据流：
      Enterprise 特征 (N, 13)
        ├─→ FeatureGate → X_gated
        ├─→ MultiViewHyperEncoder → h_struct (N, 128)
        └─→ HeteroChannelEncoder  → h_feat   (N, 128)
              ↓
        FusionGate(h_struct, h_feat, hint) → h_fusion (N, 64)
              ↓
        Γ 跨关系风险传播 → h_risk (N, 128)
              ↓
        Concat[h_fusion, h_risk] → (N, 192)
              ↓
        TemporalEncoder(h_seq) → z_v (N, 64)
              ↓
        MultiTaskHeads → logit_white, logit_risk, logit_grade
    ==========================================================================
    """

    def __init__(self, in_dims: dict, edge_types: list):
        super().__init__()

        # ── 自适应门控阀 ──
        ent_dim = in_dims.get("enterprise", 13)
        self.feature_gate = AdaptiveFeatureGate(
            feature_dim=ent_dim,
            struct_hint_dim=8,
            hidden=64,
        )

        # ── 同构通道：多视图超图 ──
        self.hyper_encoder = MultiViewHyperEncoder(
            in_dim=ent_dim,
            hidden=HYPER_HIDDEN,
        )

        # ── 异构通道：特征图 ──
        self.hetero_encoder = HeteroChannelEncoder(
            in_dims=in_dims,
            edge_types=edge_types,
            hidden=HIDDEN_DIM,
        )

        # ── 双通道融合 ──
        self.fusion_gate = FusionGate(
            struct_dim=HYPER_HIDDEN,
            feat_dim=HIDDEN_DIM,
            hidden=FUSION_HIDDEN,
            hint_dim=8,
        )

        # ── Γ 矩阵 ──
        edge_names = list(set(et[1] for et in edge_types))
        self.gamma_module = CrossRelationPropagation(edge_names=edge_names)

        # ── 时序编码器（lazy init） ──
        self.temporal = TemporalEncoder(input_dim=None)

        # ── 预测头 ──
        self.fusion_dim = FUSION_HIDDEN + HIDDEN_DIM  # h_fusion(64) + h_risk(128) = 192
        self.heads = MultiTaskHeads(in_dim=self.temporal.hidden)

        # ── per-relation 消息提取器 ──
        self.msg_extractors = nn.ModuleDict()
        for ename in edge_names:
            self.msg_extractors[ename] = nn.Linear(FUSION_HIDDEN, HIDDEN_DIM)

    def forward(self, x_dict: dict, edge_index_dict: dict,
                hyperedges: dict, num_enterprises: int,
                x_struct: dict = None, x_missing: dict = None,
                struct_hint: dict = None, x_seq: torch.Tensor = None):
        """
        ==========================================================================
        完整前向传播（AMP 友好）。
        ==========================================================================
        """
        N_ent = x_dict["enterprise"].shape[0]
        device = x_dict["enterprise"].device

        # ── 1. FeatureGate ──
        x_ent = x_dict["enterprise"]
        m_ent = x_missing.get("enterprise") if x_missing else None
        xs_ent = x_struct.get("enterprise") if x_struct else None
        hint_ent = struct_hint.get("enterprise") if struct_hint else None

        if m_ent is not None and xs_ent is not None:
            if hint_ent is None:
                hint_ent = torch.zeros(x_ent.shape[0], 8, device=device)
            x_gated_ent = self.feature_gate(x_ent, m_ent, hint_ent, xs_ent)
        else:
            x_gated_ent = x_ent

        x_dict_gated = {**x_dict, "enterprise": x_gated_ent}

        # ── 2. 同构通道：超图编码 ──
        h_struct = self.hyper_encoder(x_gated_ent, hyperedges)  # (N_ent, 128)

        # ── 3. 异构通道：特征图编码 ──
        h_feat = self.hetero_encoder(x_dict_gated, edge_index_dict)  # (N_ent, 128)
        if h_feat is None:
            h_feat = torch.zeros(N_ent, HIDDEN_DIM, device=device)

        # ── 4. 双通道融合 ──
        h_fusion = self.fusion_gate(h_struct, h_feat, struct_hint=hint_ent)  # (N_ent, 64)

        # ── 5. Γ 跨关系风险传播 ──
        m_v_r = {}
        for ename in self.msg_extractors:
            m_v_r[ename] = self.msg_extractors[ename](h_fusion)

        s_v, h_risk, gamma = self.gamma_module(m_v_r, edge_index_dict, num_enterprises)
        # h_risk: (N_ent, 128)

        # ── 6. 融合 → 时序 ──
        h_combined = torch.cat([h_fusion, h_risk], dim=-1)  # (N_ent, 192)

        # ── 7. 时序编码 ──
        z_v = self.temporal(h_combined, x_seq=x_seq)  # (N_ent, 64)

        # ── 8. 预测头 ──
        logit_white, logit_risk, logit_grade = self.heads(z_v)

        return logit_white, logit_risk, logit_grade, gamma, h_fusion


# ═══════════════════════════════════════════════════════════
# 损失函数
# ═══════════════════════════════════════════════════════════
def compute_losses(logit_white, logit_risk, logit_grade,
                   y_white, y_risk, y_grade, mask, gamma,
                   h_fusion, hyperedges=None, h_struct=None):
    # 主监督
    n_pos = y_white[mask].sum()
    n_neg = mask.sum() - n_pos
    pos_weight = torch.tensor([n_neg / max(n_pos, 1)]).clamp(max=100.0).to(logit_white.device)
    L_white = F.binary_cross_entropy_with_logits(
        logit_white[mask].squeeze(-1), y_white[mask].float(),
        pos_weight=pos_weight,
    )
    L_risk = F.binary_cross_entropy_with_logits(
        logit_risk[mask].squeeze(-1), y_risk[mask].float()
    )
    L_grade = F.cross_entropy(logit_grade[mask], y_grade[mask])

    # Γ 正则化
    L_gamma_reg = ((gamma - torch.eye(gamma.shape[0], device=gamma.device)) ** 2).mean()

    # 超图结构一致性（采样评估，不全量遍历防 OOM）
    L_struct = torch.tensor(0.0, device=logit_white.device)
    if hyperedges and LAMBDA_STRUCT > 0:
        prob_w = torch.sigmoid(logit_white.squeeze(-1))
        count = 0
        # 每个视图只采样前 50 条超边（避免全量遍历撑爆显存）
        for view_name, he_list in hyperedges.items():
            sample_n = min(len(he_list), 50)
            for i in range(sample_n):
                he = he_list[i].to(logit_white.device)
                if len(he) >= 2:
                    preds = prob_w[he]
                    L_struct += ((preds - preds.mean()) ** 2).mean()
                    count += 1
        if count > 0:
            L_struct /= count

    L_total = (L_white
               + LAMBDA_RISK * L_risk
               + LAMBDA_GRADE * L_grade
               + LAMBDA_GAMMA_REG * L_gamma_reg
               + LAMBDA_STRUCT * L_struct)

    loss_dict = {
        "total": L_total.item(), "white": L_white.item(),
        "risk": L_risk.item(), "grade": L_grade.item(),
        "gamma_reg": L_gamma_reg.item(), "struct": L_struct.item(),
    }
    return L_total, loss_dict


# ═══════════════════════════════════════════════════════════
# 评估
# ═══════════════════════════════════════════════════════════
@torch.no_grad()
def evaluate_model(model, data, mask):
    model.eval()
    logit_w, logit_r, logit_g, gamma, _ = model(
        data.x_dict, data.edge_index_dict, data.hyperedges,
        data.num_enterprises,
        x_struct=data.x_struct, x_missing=data.x_missing,
        struct_hint=data.struct_hint,
        x_seq=data.x_seq if hasattr(data, "x_seq") and data.x_seq is not None else None,
    )
    prob_w = torch.sigmoid(logit_w[mask]).cpu().squeeze(-1)
    y_true = data.y_white[mask].cpu()
    auc = roc_auc_score(y_true, prob_w) if y_true.sum() > 0 and (1 - y_true).sum() > 0 else 0.5
    acc = accuracy_score(y_true, (prob_w >= 0.5).int())
    return auc, acc, gamma


def _report_memory(device: torch.device):
    """打印当前 GPU 显存使用情况"""
    if device.type == "cuda":
        allocated = torch.cuda.memory_allocated(device) / 1024**3
        reserved = torch.cuda.memory_reserved(device) / 1024**3
        print(f"  GPU 显存: 已分配 {allocated:.2f} GB, 已预留 {reserved:.2f} GB")
    else:
        print(f"  运行在 CPU，未使用 GPU")


# ═══════════════════════════════════════════════════════════
# 训练入口
# ═══════════════════════════════════════════════════════════
def train(model, data, epochs: int = EPOCHS):
    device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")
    print(f"  设备: {device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(device)}")
        print(f"  显存总量: {torch.cuda.get_device_properties(device).total_memory / 1024**3:.1f} GB")

    model = model.to(device)
    data = data.to(device)

    _report_memory(device)

    # ── AMP 混合精度 ──
    use_amp = USE_AMP and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp) if use_amp else None
    print(f"  AMP 混合精度: {'启用' if use_amp else '关闭'}")

    # ── 优化器 ──
    optimizer = AdamW([
        {"params": model.hyper_encoder.parameters(), "lr": LR_HYPER},
        {"params": model.hetero_encoder.parameters(), "lr": LR},
        {"params": model.fusion_gate.parameters(), "lr": LR},
        {"params": model.feature_gate.parameters(), "lr": LR},
        {"params": model.gamma_module.parameters(), "lr": LR * 0.1},
        {"params": model.temporal.parameters(), "lr": LR},
        {"params": model.heads.parameters(), "lr": LR},
        {"params": model.msg_extractors.parameters(), "lr": LR},
    ], lr=LR, weight_decay=WEIGHT_DECAY)

    scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.5,
                                   patience=15, min_lr=1e-6)
    best_auc = 0
    best_state = None
    patience = EARLY_STOP_PATIENCE

    # ── 预构建 forward 参数（不变的部分只传一次） ──
    x_seq = data.x_seq if hasattr(data, "x_seq") and data.x_seq is not None else None

    print(f"\n  开始训练 ({epochs} epochs, 早停={patience})...")
    print(f"  {'Epoch':>5s} | {'Loss':>7s} {'W':>7s} | "
          f"{'Γdiag':>7s} | {'ValAUC':>7s} {'Best':>7s} | {'Time':>7s}")
    print(f"  {'─'*5:>5s}─┼{'─'*8:>8s}─┼{'─'*8:>8s}─┼{'─'*7:>7s}─┼{'─'*7:>7s}─┼{'─'*7:>7s}")
    t0 = time.time()
    t_epoch_start = time.time()

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()

        # ── 全批量前向（AMP） ──
        with torch.amp.autocast("cuda", enabled=use_amp):
            logit_w, logit_r, logit_g, gamma, h_f = model(
                data.x_dict, data.edge_index_dict, data.hyperedges,
                data.num_enterprises,
                x_struct=data.x_struct, x_missing=data.x_missing,
                struct_hint=data.struct_hint,
                x_seq=x_seq,
            )

            loss, loss_dict = compute_losses(
                logit_w, logit_r, logit_g,
                data.y_white, data.y_risk, data.y_grade,
                data.train_mask, gamma, h_f,
                hyperedges=data.hyperedges,
            )

        # ── 反向 ──
        if use_amp:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        del logit_w, logit_r, logit_g, gamma, h_f, loss

        # ── 验证 ──
        val_auc, val_acc, gamma_val = evaluate_model(model, data, data.val_mask)
        scheduler.step(val_auc)

        if val_auc > best_auc:
            best_auc = val_auc
            best_state = copy.deepcopy({k: v.cpu() for k, v in model.state_dict().items()})
            patience = EARLY_STOP_PATIENCE
        else:
            patience -= 1

        if device.type == "cuda":
            torch.cuda.empty_cache()
        gc.collect()

        # ── 日志（每 epoch 输出，CPU 调试不嫌多） ──
        gamma_diag_val = gamma_val.diag().mean().item()
        t_epoch_end = time.time()
        elapsed_epoch = t_epoch_end - t_epoch_start
        t_epoch_start = t_epoch_end

        print(f"  {epoch+1:5d} | {loss_dict['total']:7.4f} {loss_dict['white']:7.4f} | "
              f"{gamma_diag_val:7.3f} | {val_auc:7.4f} {best_auc:7.4f} | {elapsed_epoch:6.1f}s"
              + (" [BEST]" if patience == EARLY_STOP_PATIENCE else ""))

        if patience <= 0:
            print(f"  早停于 epoch {epoch+1}")
            break

    elapsed = time.time() - t0
    print(f"  训练完成, {elapsed:.1f}s, 最佳验证 AUC={best_auc:.4f}")

    # 恢复最佳模型（从 CPU copy 回 device）
    model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    return model
