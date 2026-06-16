"""
================================================================================
Layer 3: Cross-Relation Risk Propagation（Γ 矩阵 + 风险状态分离）
================================================================================
v5.1: Γ 矩阵仅在同构边（企业→企业）之间传播，特征查找边不参与混合。

核心逻辑:
  - 同构边: trade, equity, legal_rep — 风险可跨关系传播
  - 特征边: has_financial, has_lawsuit, uses_scf — 仅过 W_r 注入信号，
    不参与 Γ 混合（"ROA 指标向应付账款周转率传播风险"在语义上不成立）

参数:
  edge_names:     所有边类型名称（6 种）
  risk_edge_names: 参与 Γ 风险传播的边类型（3 种同构边）
  hidden:         消息维度 (128)
  gamma_init:     Γ 矩阵初始化方式 ("identity" | "random")
================================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from config import HIDDEN_DIM, EDGE_TYPE_NAMES


class CrossRelationPropagation(nn.Module):
    """
    ==========================================================================
    跨关系风险传播模块（v5.1 — 限定同构边）

    数据流:
      1. W_r: 每种边独立变换 m_v^r → s_v^r
      2. Γ 混合: 仅同构边之间互相传播（trade ↔ equity ↔ legal_rep）
      3. 特征边: s_v 保持不变，不参与 Γ
      4. 语义注意力: 所有边（同构 + 特征）加权融合 → h_risk
    ==========================================================================
    """
    def __init__(self, edge_names: list = None,
                 risk_edge_names: list = None,
                 hidden: int = HIDDEN_DIM,
                 gamma_init: str = "identity",
                 ablation: bool = False):
        super().__init__()
        if edge_names is None:
            edge_names = EDGE_TYPE_NAMES
        self.edge_names = edge_names
        self.R = len(edge_names)                    # 全部边类型数（6）
        self.hidden = hidden

        # ── 风险边：参与 Γ 混合的同构边（3） ──
        if risk_edge_names is None:
            # 默认：全部边都参与（向后兼容）
            self.risk_edge_names = edge_names
        else:
            self.risk_edge_names = risk_edge_names
        self.R_gamma = len(self.risk_edge_names)    # Γ 矩阵维度（3）
        self.ablation = ablation

        # 建立风险边 ↔ 全部边之间的索引映射
        self.risk_to_all = [self.edge_names.index(e) for e in self.risk_edge_names]

        # ---- Γ 矩阵：仅同构边之间的关系迁移强度 (R_gamma × R_gamma) ----
        raw_gamma = torch.randn(self.R_gamma, self.R_gamma)
        if gamma_init == "identity":
            raw_gamma = raw_gamma * 0.01 + torch.eye(self.R_gamma) * 1.0
        self.gamma_raw = nn.Parameter(raw_gamma)

        # ---- 每种边类型的变换矩阵 W_r（所有边都有） ----
        self.W_r = nn.ModuleDict()
        for ename in self.edge_names:
            self.W_r[ename] = nn.Linear(hidden, hidden)

        # ---- 语义级注意力（所有边参与融合） ----
        self.semantic_attn = nn.Linear(hidden, 1)

    def get_gamma(self) -> torch.Tensor:
        """获取归一化后的 Γ 矩阵（消融模式下退化为单位矩阵）"""
        if self.ablation:
            return torch.eye(self.R_gamma, device=self.gamma_raw.device)
        return F.softmax(self.gamma_raw, dim=-1)

    def forward(self, m_v_r: dict, edge_index_dict: dict,
                num_enterprises: int):
        """
        ==========================================================================
        输入:
          m_v_r:           每种边的消息 (N, hidden)
          edge_index_dict: 边索引字典（当前未使用，保留兼容）
          num_enterprises: 企业数 N

        输出:
          s_v_new:  风险状态字典
          h_risk:   语义融合后的风险表示 (N, hidden)
          gamma:    Γ 矩阵（用于日志/可视化）
        ==========================================================================
        """
        gamma = self.get_gamma()  # (R_gamma, R_gamma)
        N = num_enterprises
        device = gamma.device

        # ---- 1. 初始化风险状态：每种边过 W_r ----
        s_v = {}
        for ename in self.edge_names:
            if ename in m_v_r:
                s_v[ename] = self.W_r[ename](m_v_r[ename])
            else:
                s_v[ename] = torch.zeros(N, self.hidden, device=device)

        # ---- 2. 跨关系迁移：仅同构边之间 ----
        s_v_new = {}

        # 2a. 同构边：Γ 混合
        for r_out in self.risk_edge_names:
            migrated = torch.zeros(N, self.hidden, device=device)
            r_out_idx = self.risk_edge_names.index(r_out)
            for r_in_idx, r_in in enumerate(self.risk_edge_names):
                migrated = migrated + gamma[r_in_idx, r_out_idx] * s_v[r_in]
            s_v_new[r_out] = migrated

        # 2b. 特征边：不参与 Γ，直接保留
        for ename in self.edge_names:
            if ename not in self.risk_edge_names:
                s_v_new[ename] = s_v[ename]

        # ---- 3. 语义级融合 → h_risk（所有边参与） ----
        risk_stacked = torch.stack(
            [s_v_new[e] for e in self.edge_names], dim=1
        )  # (N, R, hidden)
        attn_scores = F.softmax(
            self.semantic_attn(risk_stacked).squeeze(-1), dim=-1
        )  # (N, R)
        h_risk = (attn_scores.unsqueeze(-1) * risk_stacked).sum(dim=1)  # (N, hidden)

        # ---- 4. 展开为全尺寸 Γ（特征边部分 = 单位矩阵，便于打印/正则） ----
        gamma_full = torch.eye(self.R, device=device)
        for i, ri in enumerate(self.risk_to_all):
            for j, rj in enumerate(self.risk_to_all):
                gamma_full[ri, rj] = gamma[i, j]

        return s_v_new, h_risk, gamma_full
