"""
================================================================================
Layer 3: Cross-Relation Risk Propagation（Γ 矩阵 + 风险状态分离）
================================================================================
核心创新模块。不碰 h_v，只吃 Layer 2 的 m_v^r。
- Γ 矩阵：6×6 可学习参数，控制风险在各关系类型之间的迁移强度
- 风险状态：每种入边类型独立维护 s_v^r
- 状态传播：s_v^{r,(ℓ+1)} = Σ α_{vu}^r × W_r × (Σ Γ[r'][r] × s_u^{r'})
================================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from config import GAMMA_DIM, HIDDEN_DIM, EDGE_TYPE_NAMES


class CrossRelationPropagation(nn.Module):
    """
    ==========================================================================
    跨关系风险传播模块

    参数:
      edge_names:     边类型名称列表（从数据中动态获取）
      hidden:         消息维度 (128)
      gamma_init:     Γ 矩阵初始化方式 ("identity" | "random")
    ==========================================================================
    """
    def __init__(self, edge_names: list = None, hidden: int = HIDDEN_DIM,
                 gamma_init: str = "identity"):
        super().__init__()
        if edge_names is None:
            edge_names = EDGE_TYPE_NAMES  # fallback
        self.edge_names = edge_names
        self.R = len(edge_names)
        self.hidden = hidden

        # ---- Γ 矩阵：关系迁移强度 ----
        raw_gamma = torch.randn(self.R, self.R)
        if gamma_init == "identity":
            raw_gamma = raw_gamma * 0.01 + torch.eye(self.R) * 1.0
        self.gamma_raw = nn.Parameter(raw_gamma)

        # ---- 每种边类型的变换矩阵 W_r ----
        self.W_r = nn.ModuleDict()
        for ename in self.edge_names:
            self.W_r[ename] = nn.Linear(hidden, hidden)

        # ---- 语义级注意力 ----
        self.semantic_attn = nn.Linear(hidden, 1)

    def get_gamma(self) -> torch.Tensor:
        """获取归一化后的 Γ 矩阵"""
        return F.softmax(self.gamma_raw, dim=-1)

    def forward(self, m_v_r: dict, edge_index_dict: dict,
                num_enterprises: int):
        gamma = self.get_gamma()
        N = num_enterprises
        device = gamma.device
        R = self.R

        # ---- 1. 初始化风险状态 ----
        s_v = {}
        for ename in self.edge_names:
            if ename in m_v_r:
                s_v[ename] = self.W_r[ename](m_v_r[ename])
            else:
                s_v[ename] = torch.zeros(N, self.hidden, device=device)

        # ---- 2. 跨关系迁移 ----
        s_v_new = {}
        for r_out_idx, r_out in enumerate(self.edge_names):
            migrated = torch.zeros(N, self.hidden, device=device)
            for r_in_idx, r_in in enumerate(self.edge_names):
                migrated = migrated + gamma[r_in_idx, r_out_idx] * s_v[r_in]
            s_v_new[r_out] = migrated

        # ---- 3. 语义级融合 → h_risk ----
        risk_stacked = torch.stack(
            [s_v_new[e] for e in self.edge_names], dim=1
        )  # (N, R, hidden)
        attn_scores = F.softmax(
            self.semantic_attn(risk_stacked).squeeze(-1), dim=-1
        )  # (N, R)
        h_risk = (attn_scores.unsqueeze(-1) * risk_stacked).sum(dim=1)  # (N, hidden)

        return s_v_new, h_risk, gamma
