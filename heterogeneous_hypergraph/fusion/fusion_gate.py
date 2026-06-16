"""
================================================================================
双通道融合门（Fusion Gate）— v5 新增
================================================================================
将同构通道（超图结构表示 h_struct）和异构通道（特征感知表示 h_feat）
通过 per-node 注意力门融合为统一的 h_fusion。

与 FeatureGate 的分工：
  FeatureGate（v4 保留）: 逐维度 α[d] — 决定信原始值还是结构推算（特征维度层面）
  FusionGate（v5 新增）:   每节点 gate — 决定信超图结构还是异构特征（通道维度层面）

门控语义：
  gate → 1.0: 该企业的预测更依赖超图结构位置（中小企业、孤立节点）
  gate → 0.0: 该企业的预测更依赖自身财务/风险特征（上市公司、数据完整）
================================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from config import FUSION_HIDDEN, DROPOUT


class FusionGate(nn.Module):
    """
    ==========================================================================
    双通道注意力融合

    输入:
      h_struct: (N, D_struct)  超图通道输出的结构角色表示
      h_feat:   (N, D_feat)    异构通道输出的特征感知表示
      struct_hint: (N, S)      图结构统计量（可选，给门提供上下文）

    输出:
      h_fusion: (N, D_out)     融合后的统一表示
    ==========================================================================
    """
    def __init__(self, struct_dim: int, feat_dim: int,
                 hidden: int = FUSION_HIDDEN, hint_dim: int = 0,
                 dropout: float = DROPOUT):
        super().__init__()

        # 确保两通道维度对齐
        self.struct_proj = nn.Linear(struct_dim, hidden) if struct_dim != hidden else nn.Identity()
        self.feat_proj = nn.Linear(feat_dim, hidden) if feat_dim != hidden else nn.Identity()

        # 门控网络：输入 = [h_struct || h_feat || hint]
        gate_in = hidden * 2 + hint_dim
        self.gate_net = nn.Sequential(
            nn.Linear(gate_in, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
            nn.Sigmoid(),
        )

        # 融合后投影
        self.fusion_proj = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, h_struct: torch.Tensor, h_feat: torch.Tensor,
                struct_hint: torch.Tensor = None) -> torch.Tensor:
        """
        ==========================================================================
        h_struct: (N, D_struct)
        h_feat:   (N, D_feat)
        struct_hint: (N, S) or None

        返回: h_fusion (N, hidden)
        ==========================================================================
        """
        hs = self.struct_proj(h_struct)  # (N, hidden)
        hf = self.feat_proj(h_feat)      # (N, hidden)

        # 构建门控输入
        if struct_hint is not None:
            gate_in = torch.cat([hs, hf, struct_hint], dim=-1)
        else:
            gate_in = torch.cat([hs, hf], dim=-1)

        gate = self.gate_net(gate_in)  # (N, 1)

        # 融合
        h_fused = gate * hs + (1.0 - gate) * hf
        h_fusion = self.fusion_proj(h_fused)
        h_fusion = self.dropout(h_fusion)

        return h_fusion
