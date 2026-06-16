"""
================================================================================
Layer 2: 异构图编码器（SAGEConv，双通道输出，低显存版）
================================================================================
输出两个不重叠的产物：
  h_v    — 最后一层 enterprise 表示 → 喂给 Layer 4 时序编码
  m_v^r  — 分关系消息 → 喂给 Layer 3 Γ 矩阵

改动（v4.1）：
  - GATConv → SAGEConv：去掉了 multi-head attention，显存降低 ~40%
  - 去掉 JK 拼接：只用最后一层输出（128-dim），不再 concat 三层
  - 消息提取处加 .detach()：阻断二次反向图，进一步省显存
  - 门控阀仍在投影前逐节点类型工作
================================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, HeteroConv, Linear
from config import (
    HIDDEN_DIM, DROPOUT, NUM_GNN_LAYERS,
)
from feature_gate import AdaptiveFeatureGate


class DualOutputEncoder(nn.Module):
    """
    ==========================================================================
    双通道输出异构图编码器（SAGEConv + 自适应门控阀）

    前向流程:
      X_raw → [Gate] → X_gated → [Proj] → [2×SAGEConv] → h_v + m_v^r

    显存对比（17.7 万节点，46.3 万 trade 边）：
      GAT 3层 JK:  峰值 ~2.0 GB
      SAGE 2层 无JK: 峰值 ~1.0 GB
    ==========================================================================
    """
    def __init__(self, in_dims: dict, edge_types: list, hidden: int = HIDDEN_DIM,
                 dropout: float = DROPOUT, use_gate: bool = True):
        super().__init__()
        self.hidden = hidden
        self.num_layers = NUM_GNN_LAYERS
        self.dropout = nn.Dropout(dropout)
        self.edge_types = edge_types
        self.use_gate = use_gate

        # ---- 自适应门控阀（每个节点类型独立） ----
        if use_gate:
            self.gates = nn.ModuleDict()
            for ntype, dim in in_dims.items():
                if dim is not None and dim > 0:
                    self.gates[ntype] = AdaptiveFeatureGate(
                        feature_dim=dim,
                        struct_hint_dim=8,
                        hidden=64,
                    )
        else:
            self.gates = None

        # ---- 节点类型投影层 ----
        self.proj = nn.ModuleDict()
        for ntype, dim in in_dims.items():
            if dim is not None:
                self.proj[ntype] = Linear(dim, hidden)

        # ---- 2 层 SAGE 卷积 ----
        self.convs = nn.ModuleList()
        for _ in range(NUM_GNN_LAYERS):
            conv_dict = {}
            for etype in edge_types:
                # 已投影到统一维度 hidden，直接显式传入，避免 lazy init
                conv_dict[etype] = SAGEConv(hidden, hidden)
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
              h_v:   (N_ent, hidden)          ← 仅最后一层
              m_v_r: {ename: (N_ent, hidden)} ← 分关系消息
        """
        # ---- 1. 自适应门控阀（投影前） ----
        x_gated = {}
        for ntype, x in x_dict.items():
            # 收集门控所需输入
            m = x_missing.get(ntype) if x_missing else None
            xs = x_struct.get(ntype) if x_struct else None
            hint = struct_hint.get(ntype) if struct_hint else None

            if self.use_gate and ntype in self.gates and m is not None and xs is not None:
                if hint is None:
                    hint = torch.zeros(x.shape[0], 8, device=x.device)
                x_gated[ntype] = self.gates[ntype](x, m, hint, xs)
            else:
                x_gated[ntype] = x

        # ---- 2. 投影到统一维度 ----
        h = {}
        for ntype in self.proj:
            if ntype in x_gated:
                h[ntype] = self.proj[ntype](x_gated[ntype])
            else:
                h[ntype] = self.proj[ntype](x_dict[ntype])

        # ---- 3. 2 层消息传递（不存中间层） ----
        for conv in self.convs:
            h = conv(h, edge_index_dict)
            h = {k: self.dropout(F.relu(v)) for k, v in h.items()}

        # ---- 4. 最后一层 enterprise 表示 ----
        h_v = h["enterprise"]  # (N_ent, hidden)

        # ---- 5. 分关系消息（.detach() 阻断反向图二次膨胀） ----
        ent_detached = h_v.detach()
        m_v_r = {}
        for ename in self.msg_extractors:
            m_v_r[ename] = self.msg_extractors[ename](ent_detached)

        return h_v, m_v_r
