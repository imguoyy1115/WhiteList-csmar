"""
================================================================================
自适应门控阀（Adaptive Feature Gate）
================================================================================
根据每个节点的特征完整度，逐维度学习"信原始值还是信结构推算值"。
零特征率低 → α→1，退化为直通（不损失原始信息）
零特征率高 → α→0，自动走结构特征（邻域聚合替代缺失值）

输入: X (N, D), M (N, D), struct_hint (N, S), X_struct (N, D)
输出: X_gated (N, D) — 自适应融合后的特征
================================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class AdaptiveFeatureGate(nn.Module):
    """
    ==========================================================================
    每个节点 × 每个维度独立学习信任权重 α[i,d]

    gate_input = concat[ X, M, expand(struct_hint) ]
    α = sigmoid( MLP(gate_input) )
    out = α ⊙ X  +  (1-α) ⊙ X_struct
    ==========================================================================
    """
    def __init__(self, feature_dim: int, struct_hint_dim: int = 8,
                 hidden: int = 64):
        super().__init__()
        self.feature_dim = feature_dim

        # 门控网络：输入 = X(D) + M(D) + struct_hint(S) → 输出 α(D)
        gate_in = feature_dim + feature_dim + struct_hint_dim
        self.gate_net = nn.Sequential(
            nn.Linear(gate_in, hidden),
            nn.ReLU(),
            nn.Linear(hidden, feature_dim),
        )

        # 结构特征精炼：X_struct 可能只是邻域均值，用轻量线性层精炼
        self.struct_refine = nn.Sequential(
            nn.Linear(feature_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, feature_dim),
        )

    def forward(self, X: torch.Tensor, M: torch.Tensor,
                struct_hint: torch.Tensor,
                X_struct: torch.Tensor) -> torch.Tensor:
        """
        X:           (N, D)  原始特征（含缺失值填 0 / 均值 / scaler 变换后的值）
        M:           (N, D)  缺失指示 0=真实 1=缺失
        struct_hint: (N, S)  图结构统计量（度、是否上市等）
        X_struct:    (N, D)  邻域聚合的结构特征

        返回: X_gated (N, D)  自适应融合后的特征
        """
        N, D = X.shape
        device = X.device

        # 1. struct_hint 广播到 (N, D) 以便 concat
        if struct_hint.shape[1] < D:
            # 用线性层把 S 维扩到 D 维（或用 repeat）
            S = struct_hint.shape[1]
            hint_expanded = torch.zeros(N, D, device=device)
            # 简单策略：前 S 维用 struct_hint，后面补零
            hint_expanded[:, :S] = struct_hint
        else:
            hint_expanded = struct_hint[:, :D]

        # 2. 门控输入
        gate_input = torch.cat([X, M, hint_expanded], dim=-1)  # (N, 2D + D)
        alpha = torch.sigmoid(self.gate_net(gate_input))        # (N, D)

        # 3. 精炼结构特征
        X_struct_refined = self.struct_refine(X_struct)          # (N, D)

        # 4. 自适应融合
        X_gated = alpha * X + (1.0 - alpha) * X_struct_refined

        return X_gated

    def get_trust_stats(self, X, M, struct_hint, X_struct):
        """返回信任度统计（用于诊断和分析）"""
        with torch.no_grad():
            N, D = X.shape
            S = struct_hint.shape[1]
            hint_expanded = torch.zeros(N, D, device=X.device)
            hint_expanded[:, :S] = struct_hint
            gate_input = torch.cat([X, M, hint_expanded], dim=-1)
            alpha = torch.sigmoid(self.gate_net(gate_input))

        return {
            "alpha_mean": alpha.mean().item(),
            "alpha_per_node_mean": alpha.mean(dim=1),
            "alpha_high_trust_ratio": (alpha > 0.7).float().mean().item(),
            "alpha_low_trust_ratio": (alpha < 0.3).float().mean().item(),
            "avg_alpha_by_missing": {
                "known_features": alpha[M < 0.5].mean().item(),
                "missing_features": alpha[M > 0.5].mean().item(),
            },
        }
