"""
================================================================================
超图卷积层（Hypergraph Convolution）— 向量化版本
================================================================================
实现 "Node → Hyperedge → Node" 的两阶段消息传递。

与旧版（for-loop）的关键区别：
  - 用 _scatter_mean 一次性处理所有超边，无需逐条遍历
  - 10,660 条超边从 10,660 次 GPU 调用 → 2 次 scatter 操作
  - 预期加速 50-200x（超图卷积部分）

输入格式：
  hyperedge_index: (2, E_total) LongTensor
    [0, :] = hyperedge ID
    [1, :] = node ID
  等价于将超边列表展平为 COO 格式。
================================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from config import HYPER_HIDDEN, HYPER_LAYERS, DROPOUT


def _scatter_mean(src: torch.Tensor, index: torch.Tensor,
                  dim: int = 0, dim_size: int = None) -> torch.Tensor:
    """
    纯 PyTorch _scatter_mean —— 不依赖 torch_scatter。
    等价于 torch_scatter._scatter_mean(src, index, dim, dim_size)。
    """
    if dim_size is None:
        dim_size = int(index.max().item()) + 1 if index.numel() > 0 else 0

    if dim_size == 0:
        return torch.zeros(0, src.size(1) if src.ndim > 1 else 1,
                          dtype=src.dtype, device=src.device)

    # index_add_ 求和
    out = torch.zeros(dim_size, src.size(1) if src.ndim > 1 else 1,
                      dtype=src.dtype, device=src.device)
    out.index_add_(0, index, src)

    # 计数（每目标行被 hit 多少次）
    count = torch.zeros(dim_size, device=src.device)
    count.index_add_(0, index, torch.ones(index.size(0), device=src.device))
    count = count.clamp(min=1)

    return out / count.unsqueeze(-1) if out.ndim > 1 else out / count


def hyperedges_list_to_index(he_list: list, device: torch.device) -> torch.Tensor:
    """
    将超边列表（每条超边是一个 node ID tensor）转换为扁平索引。

    输入: [tensor([A,B,C]), tensor([D,E]), ...]
    输出: tensor([[0,0,0,1,1,...],   ← hyperedge ID
                  [A,B,C,D,E,...]])   ← node ID
    """
    if not he_list:
        return torch.zeros((2, 0), dtype=torch.long, device=device)

    he_ids_parts = []
    node_ids_parts = []
    for i, he in enumerate(he_list):
        n = len(he)
        he_ids_parts.append(torch.full((n,), i, dtype=torch.long))
        node_ids_parts.append(he.long())

    he_ids = torch.cat(he_ids_parts)
    node_ids = torch.cat(node_ids_parts)
    return torch.stack([he_ids, node_ids]).to(device)


class HypergraphConv(nn.Module):
    """
    ==========================================================================
    单张超图上的卷积层（向量化 scatter 实现）。

    前向流程：
      1. Node → Hyperedge:  _scatter_mean(x_proj[node_ids], he_ids) → 超边表示
      2. Hyperedge → Node:  _scatter_mean(edge_reprs[he_ids], node_ids) → 节点更新
      3. 残差连接 + LayerNorm + Dropout
    ==========================================================================
    """
    def __init__(self, in_dim: int, out_dim: int,
                 dropout: float = DROPOUT):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim

        self.node_to_edge = nn.Linear(in_dim, out_dim)
        self.edge_to_node = nn.Linear(out_dim, out_dim)
        self.skip = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()
        self.norm = nn.LayerNorm(out_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor,
                hyperedge_index: torch.Tensor) -> torch.Tensor:
        """
        x: (N, in_dim)
        hyperedge_index: (2, E_total)  [he_id, node_id]
        返回: (N, out_dim)
        """
        if hyperedge_index.numel() == 0:
            return self.dropout(F.relu(self.skip(x)))

        N = x.shape[0]
        he_ids = hyperedge_index[0]   # (E_total,)
        node_ids = hyperedge_index[1]  # (E_total,)

        # ── Step 1: Node → Hyperedge ──
        x_proj = self.node_to_edge(x)                        # (N, out_dim)
        edge_reprs = _scatter_mean(x_proj[node_ids], he_ids, dim=0)  # (E, out_dim)
        edge_reprs = self.edge_to_node(edge_reprs)
        edge_reprs = F.relu(edge_reprs)

        # ── Step 2: Hyperedge → Node ──
        node_agg = _scatter_mean(
            edge_reprs[he_ids], node_ids,
            dim=0, dim_size=N
        )  # (N, out_dim)

        # ── 残差 + Norm + Dropout ──
        h_new = node_agg + self.skip(x)
        h_new = self.norm(h_new)
        h_new = self.dropout(F.relu(h_new))

        return h_new


class MultiViewHyperEncoder(nn.Module):
    """
    ==========================================================================
    多视图超图编码器（同构通道）

    对 4 张超图分别做卷积，然后通过注意力融合。

    输入: x (N, in_dim) + hyperedges_dict
          支持两种格式：
            - list of tensors（自动转换为 flat index，首次转换后缓存）
            - (2, E) flat index tensor（直接使用）
    输出: h_struct (N, hidden)
    ==========================================================================
    """
    def __init__(self, in_dim: int, hidden: int = HYPER_HIDDEN,
                 num_layers: int = HYPER_LAYERS, dropout: float = DROPOUT):
        super().__init__()
        self.hidden = hidden
        self.num_layers = num_layers

        self.input_proj = nn.Linear(in_dim, hidden)

        self.view_convs = nn.ModuleDict()
        self.view_names = ["supply", "equity", "legal_rep", "industry"]
        for vname in self.view_names:
            layers = []
            for _ in range(num_layers):
                layers.append(HypergraphConv(hidden, hidden, dropout=dropout))
            self.view_convs[vname] = nn.ModuleList(layers)

        self.view_attn = nn.Linear(hidden, 1)
        self.dropout = nn.Dropout(dropout)

        # 缓存：首次将 list 格式转 flat index 后存这里，后续直接复用
        self._index_cache = {}

    def _get_flat_index(self, vname: str, he_data, device: torch.device):
        """获取或构建超边的 flat index 格式"""
        if isinstance(he_data, torch.Tensor) and he_data.ndim == 2:
            # 已经是 (2, E) 格式
            return he_data.to(device)

        # list 格式 → 转换并缓存
        cache_key = f"{vname}_{device}"
        if cache_key not in self._index_cache:
            self._index_cache[cache_key] = hyperedges_list_to_index(he_data, torch.device("cpu"))
        return self._index_cache[cache_key].to(device)

    def forward(self, x: torch.Tensor,
                hyperedges: dict) -> torch.Tensor:
        """
        x: (N, in_dim)
        hyperedges: {"supply": [...], "equity": [...], ...}
        返回: h_struct (N, hidden)
        """
        x = self.input_proj(x)  # (N, hidden)

        view_outputs = []
        for vname in self.view_names:
            he_data = hyperedges.get(vname, [])
            he_index = self._get_flat_index(vname, he_data, x.device)

            h_v = x
            for conv in self.view_convs[vname]:
                h_v = conv(h_v, he_index)
            view_outputs.append(h_v)  # (N, hidden)

        # 注意力融合
        stacked = torch.stack(view_outputs, dim=1)  # (N, 4, hidden)
        attn_scores = self.view_attn(stacked).squeeze(-1)  # (N, 4)
        attn_weights = F.softmax(attn_scores, dim=-1)  # (N, 4)

        h_struct = (stacked * attn_weights.unsqueeze(-1)).sum(dim=1)  # (N, hidden)
        h_struct = self.dropout(h_struct)

        return h_struct
