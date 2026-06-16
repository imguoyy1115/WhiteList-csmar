"""
================================================================================
超图卷积层（Hypergraph Convolution）
================================================================================
实现 "Node → Hyperedge → Node" 的两阶段消息传递。

支持 4 张超图独立卷积 + 多视图注意力融合。

参考：HyperGCN (Yadati et al., NeurIPS 2019)
      HGNN+ (Gao et al., AAAI 2020)
简化实现：mean/attention 聚合，适合 CPU + mini-batch 训练。
================================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from config import HYPER_HIDDEN, HYPER_LAYERS, DROPOUT


class HypergraphConv(nn.Module):
    """
    ==========================================================================
    单张超图上的卷积层。

    前向流程：
      1. Node → Hyperedge:  聚合超边内所有节点的特征 → 超边表示
      2. Hyperedge → Node:  聚合包含某节点的所有超边表示 → 更新节点表示
      3. 残差连接 + LayerNorm + Dropout
    ==========================================================================
    """
    def __init__(self, in_dim: int, out_dim: int, aggr: str = "mean",
                 dropout: float = DROPOUT):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.aggr = aggr

        # 节点 → 超边 的消息变换
        self.node_to_edge = nn.Linear(in_dim, out_dim)
        # 超边 → 节点 的消息变换
        self.edge_to_node = nn.Linear(out_dim, out_dim)

        # 如果 in_dim ≠ out_dim，需要投影做残差
        self.skip = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()

        self.norm = nn.LayerNorm(out_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, hyperedges: list) -> torch.Tensor:
        """
        ==========================================================================
        x: (N, in_dim)  所有 enterprise 节点的特征
        hyperedges: list of (k,) tensors, 每条超边是一组节点 ID

        返回: (N, out_dim)  更新后的节点特征
        ==========================================================================
        """
        N = x.shape[0]
        device = x.device

        if not hyperedges:
            # 没有超边 → 直通
            return self.dropout(F.relu(self.skip(x)))

        # ── Step 1: Node → Hyperedge ──
        #     对每条超边，聚合其成员节点的特征
        x_proj = self.node_to_edge(x)  # (N, out_dim)

        edge_reprs = []  # 每条超边的表示
        edge_sizes = []  # 每条超边的大小（用于归一化）

        for he in hyperedges:
            he = he.to(device)
            members = x_proj[he]  # (k, out_dim)
            if self.aggr == "mean":
                edge_repr = members.mean(dim=0)  # (out_dim,)
            elif self.aggr == "attention":
                # 简化 attention: 用节点特征自身做 self-attention
                attn = torch.softmax((members * members).sum(dim=-1), dim=0)  # (k,)
                edge_repr = (members * attn.unsqueeze(-1)).sum(dim=0)
            else:
                edge_repr = members.mean(dim=0)
            edge_reprs.append(edge_repr)
            edge_sizes.append(len(he))

        edge_matrix = torch.stack(edge_reprs)  # (E, out_dim)
        edge_matrix = self.edge_to_node(edge_matrix)
        edge_matrix = F.relu(edge_matrix)

        # ── Step 2: Hyperedge → Node ──
        #     对每个节点，聚合所有包含它的超边表示
        node_agg = torch.zeros(N, self.out_dim, device=device)
        node_cnt = torch.zeros(N, device=device)

        for i, he in enumerate(hyperedges):
            he = he.to(device)
            node_agg[he] += edge_matrix[i]  # scatter add
            node_cnt[he] += 1

        # 归一化（避免度偏置）
        node_cnt = node_cnt.clamp(min=1)
        h_new = node_agg / node_cnt.unsqueeze(-1)

        # ── 残差 + Norm + Dropout ──
        h_new = h_new + self.skip(x)
        h_new = self.norm(h_new)
        h_new = self.dropout(F.relu(h_new))

        return h_new


class MultiViewHyperEncoder(nn.Module):
    """
    ==========================================================================
    多视图超图编码器（同构通道）

    对 4 张超图分别做卷积，然后通过注意力融合。

    输入: x (N, in_dim)  +  hyperedges_dict {"supply": [...], "equity": [...], ...}
    输出: h_struct (N, hidden)  结构角色表示
    ==========================================================================
    """
    def __init__(self, in_dim: int, hidden: int = HYPER_HIDDEN,
                 num_layers: int = HYPER_LAYERS, dropout: float = DROPOUT):
        super().__init__()
        self.hidden = hidden
        self.num_layers = num_layers

        # 输入投影
        self.input_proj = nn.Linear(in_dim, hidden)

        # 每个视图独立的多层超图卷积
        self.view_convs = nn.ModuleDict()
        view_names = ["supply", "equity", "legal_rep", "industry"]
        for vname in view_names:
            layers = []
            for _ in range(num_layers):
                layers.append(HypergraphConv(hidden, hidden, aggr="mean", dropout=dropout))
            self.view_convs[vname] = nn.ModuleList(layers)

        # 视图级注意力融合
        self.view_attn = nn.Linear(hidden, 1)
        self.view_names = view_names

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor,
                hyperedges: dict) -> torch.Tensor:
        """
        ==========================================================================
        x: (N, in_dim)  enterprise 节点的初始特征（13维，投影前）
        hyperedges: {"supply": [tensor, ...], "equity": [...], ...}

        返回: h_struct (N, hidden)  融合后的结构角色表示
        ==========================================================================
        """
        x = self.input_proj(x)  # (N, hidden)

        view_outputs = []
        for vname in self.view_names:
            h_v = x
            for conv in self.view_convs[vname]:
                h_v = conv(h_v, hyperedges.get(vname, []))
            view_outputs.append(h_v)  # (N, hidden)

        # 注意力融合
        stacked = torch.stack(view_outputs, dim=1)  # (N, 4, hidden)
        attn_scores = self.view_attn(stacked).squeeze(-1)  # (N, 4)
        attn_weights = F.softmax(attn_scores, dim=-1)  # (N, 4)

        h_struct = (stacked * attn_weights.unsqueeze(-1)).sum(dim=1)  # (N, hidden)
        h_struct = self.dropout(h_struct)

        return h_struct
