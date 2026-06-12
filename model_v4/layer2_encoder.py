"""
================================================================================
Layer 2: 异构图编码器（R-GAT，双通道输出）
================================================================================
输出两个不重叠的产物：
  h_v    — 全局融合表示（JK concat 3 层）→ 喂给 Layer 4 时序编码
  m_v^r  — 第 3 层每种边类型的独立消息 → 喂给 Layer 3 Γ 矩阵

当前实现：HeteroConv(GATConv) — 本质为 R-GAT
论文投稿时可替换为 torch_geometric.nn.HGTConv
================================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, HeteroConv, Linear
from config import (
    HIDDEN_DIM, NUM_HEADS, DROPOUT, NUM_GNN_LAYERS, EDGE_TYPES, EDGE_TYPE_NAMES
)
from feature_gate import AdaptiveFeatureGate


class DualOutputEncoder(nn.Module):
    """
    ==========================================================================
    双通道输出异构图编码器（支持自适应门控阀）

    前向流程:
      X_raw → [AdaptiveFeatureGate] → X_gated → [proj] → [3×HeteroConv] → h_v + m_v^r

    门控阀在投影之前工作——决定每个节点 × 每个特征维度信任原始值还是结构推算值。
    如果数据不提供 x_struct / x_missing，门控自动退化为直通（α=1）。
    ==========================================================================
    """
    def __init__(self, in_dims: dict, edge_types: list, hidden: int = HIDDEN_DIM,
                 num_heads: int = NUM_HEADS, dropout: float = DROPOUT,
                 use_gate: bool = True):
        super().__init__()
        self.hidden = hidden
        self.num_layers = NUM_GNN_LAYERS
        self.dropout = nn.Dropout(dropout)
        self.edge_types = edge_types
        self.use_gate = use_gate

        # ---- 自适应门控阀 ----
        # 门控在投影之前工作：原始特征维度 D = in_dims[ntype]
        # X_struct 和 M 都是 D 维，struct_hint 是 8 维
        ent_dim = in_dims.get("enterprise", hidden)
        if use_gate:
            self.gate = AdaptiveFeatureGate(
                feature_dim=ent_dim,
                struct_hint_dim=8,
                hidden=64,
            )
        else:
            self.gate = None

        # ---- 节点类型投影层 ----
        self.proj = nn.ModuleDict()
        for ntype, dim in in_dims.items():
            if dim is not None:
                self.proj[ntype] = Linear(dim, hidden)

        # ---- 3 层异构卷积（每层动态匹配边类型） ----
        self.convs = nn.ModuleList()
        for i in range(NUM_GNN_LAYERS):
            heads = num_heads if i < NUM_GNN_LAYERS - 1 else 1
            conv_dict = {}
            for etype in edge_types:
                conv_dict[etype] = GATConv(
                    hidden, hidden // heads, heads=heads,
                    dropout=dropout, add_self_loops=False
                )
            self.convs.append(HeteroConv(conv_dict, aggr="mean"))

        # ---- 消息提取器 ----
        self.msg_extractors = nn.ModuleDict()
        for etype in edge_types:
            ename = etype[1]  # 用边类型名称作为 key
            self.msg_extractors[ename] = nn.Linear(hidden, hidden)

    def forward(self, x_dict: dict, edge_index_dict: dict,
                x_struct: dict = None, x_missing: dict = None,
                struct_hint: dict = None):
        """
        输入: x_dict = {ntype: (N, d_in)}
              edge_index_dict = {(src, etype, dst): [2, E]}
              x_struct:     {ntype: (N, D)}  邻域聚合特征（可选）
              x_missing:    {ntype: (N, D)}  缺失指示（可选）
              struct_hint:  {ntype: (N, S)}  结构统计量（可选）
        输出:
              h_v:   (N_ent, hidden * num_layers)  ← JK 拼接
              m_v_r: {ename: (N_ent, hidden)}      ← 分关系消息
        """
        # ---- 自适应门控阀（在投影前） ----
        if self.gate is not None and x_struct is not None and x_missing is not None:
            x_gated = {}
            for ntype in x_dict:
                if ntype in x_struct and ntype in x_missing:
                    hint = struct_hint.get(ntype) if struct_hint else None
                    if hint is None:
                        hint = torch.zeros(x_dict[ntype].shape[0], 8, device=x_dict[ntype].device)
                    x_gated[ntype] = self.gate(
                        x_dict[ntype], x_missing[ntype], hint, x_struct[ntype]
                    )
                else:
                    x_gated[ntype] = x_dict[ntype]
        else:
            x_gated = x_dict  # 无门控数据 → 直通

        # ---- 投影到统一维度 ----
        h = {}
        for ntype in self.proj:
            if ntype in x_gated:
                h[ntype] = self.proj[ntype](x_gated[ntype])
            else:
                h[ntype] = self.proj[ntype](x_dict[ntype])

        # ---- 3 层消息传递 ----
        layer_outputs = []  # 存每层的 enterprise 输出
        for i, conv in enumerate(self.convs):
            h = conv(h, edge_index_dict)
            h = {k: self.dropout(F.relu(v)) for k, v in h.items()}
            layer_outputs.append(h["enterprise"])

        # ---- JK 连接 → 全局融合表示 ----
        h_v = torch.cat(layer_outputs, dim=-1)  # (N_ent, hidden * 3)

        # ---- 提取分关系消息（从最后一层的 enterprise 输出） ----
        m_v_r = {}
        ent_out = layer_outputs[-1]  # (N_ent, hidden)
        for ename in self.msg_extractors:
            m_v_r[ename] = self.msg_extractors[ename](ent_out)

        return h_v, m_v_r
