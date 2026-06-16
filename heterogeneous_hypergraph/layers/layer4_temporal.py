"""
================================================================================
Layer 4: 时序编码器（v4.3 — GRU + MLP + Temporal Gate）
================================================================================
输入: h_fusion (N, 256) + x_seq (N, 4, DIM+1)
输出: z_v (N, hidden) → 用于白名单预测 + 风险预测 + 企业分级

双路径设计:
  路径A — GRU（季度时序）:
    x_seq[:,:,:DIM] 4 个季度的原始特征 → GRU(26→32, 4步) → gru_z(N,64)
  路径B — MLP（静态投影）:
    h_fusion → compress → MLP → mlp_z(N,64)

  Temporal Gate:
    has_any = 该节点在任意季度有真实特征
    trust   = sigmoid( Linear([h_fusion || has_any]) )  → 每节点一个标量
    z_v     = trust * gru_z  +  (1-trust) * mlp_z

行为:
  上市企业（有季度数据）→ trust→1 → GRU 时序信号生效
  CP企业（全零季度）     → trust→0 → MLP 纯静态，GRU 输出被忽略

与 FeatureGate 的区别:
  FeatureGate: 逐维度 α[d] — 信原始值 vs 结构推算（工作在特征维度）
  TemporalGate: 每节点 trust — 信 GRU 时序 vs MLP 静态（工作在节点维度）
================================================================================
"""

import torch
import torch.nn as nn
from config import GRU_HIDDEN, DROPOUT


class TemporalEncoder(nn.Module):
    """
    ==========================================================================
    双路径时序编码器 + Temporal Gate

    v4.3: GRU 处理季度序列 + MLP fallback + 可学习融合门
    ==========================================================================
    """
    def __init__(self, input_dim: int = None, hidden: int = 64,
                 gru_hidden: int = 32, num_layers: int = 2,
                 dropout: float = DROPOUT):
        super().__init__()
        self.hidden = hidden
        self.gru_hidden = gru_hidden
        self.num_layers = num_layers
        self.input_dim = input_dim

        # ---- Lazy init: 首次 forward 时按实际维度构建 ----
        self.compress = None       # h_fusion → hidden
        self.mlp = None            # MLP 路径B
        self.gru = None            # GRU 路径A
        self.gru_proj = None       # GRU 输出 → hidden
        self.gate_net = None       # Temporal Gate
        self._built = False

    def _build(self, fusion_dim: int, seq_dim: int, device: torch.device):
        """首次 forward 时懒构建所有子模块"""
        self.compress = nn.Linear(fusion_dim, self.hidden).to(device)

        # 路径B: MLP fallback
        self.mlp = nn.Sequential(
            nn.Linear(self.hidden, self.hidden),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(self.hidden, self.hidden),
        ).to(device)

        # 路径A: GRU 时序编码
        # 输入: x_seq 的 4 个季度 (DIM 维特征 + 1 维 has_feature 标记)
        self.gru = nn.GRU(
            input_size=seq_dim,          # DIM + 1
            hidden_size=self.gru_hidden, # 32
            num_layers=2,
            batch_first=True,
            dropout=DROPOUT if self.num_layers > 1 else 0.0,
        ).to(device)
        self.gru_proj = nn.Linear(self.gru_hidden, self.hidden).to(device)

        # Temporal Gate: 学习什么时候信 GRU
        self.gate_net = nn.Sequential(
            nn.Linear(fusion_dim + 1, self.hidden),
            nn.ReLU(),
            nn.Linear(self.hidden, 1),
            nn.Sigmoid(),
        ).to(device)

        self._built = True

    def forward(self, h_fusion: torch.Tensor,
                x_seq: torch.Tensor = None) -> torch.Tensor:
        """
        ==========================================================================
        输入:
          h_fusion: (N, 256)      SAGE + Γ 融合后的静态表示
          x_seq:    (N, 4, D+1)   4 个季度的原始特征 + has_feature_q 标记

        输出:
          z_v: (N, hidden)  融合后的企业表示
        ==========================================================================
        """
        N = h_fusion.shape[0]
        device = h_fusion.device

        # Lazy init
        if not self._built:
            seq_dim = x_seq.shape[-1] if x_seq is not None else 1
            self._build(h_fusion.shape[-1], seq_dim, device)

        # 路径B: MLP（始终可用）
        z0 = self.compress(h_fusion)   # (N, hidden)
        mlp_z = self.mlp(z0)           # (N, hidden)

        # 路径A: GRU（需要 x_seq）
        if x_seq is not None and self.gru is not None:
            gru_out, _ = self.gru(x_seq)              # (N, 4, gru_hidden)
            gru_last = gru_out[:, -1, :]              # (N, gru_hidden)  取最后一步
            gru_z = self.gru_proj(gru_last)           # (N, hidden)

            # Temporal Gate
            has_any = x_seq[:, :, -1:].max(dim=1).values  # (N, 1)  任一季有真实特征
            gate_in = torch.cat([h_fusion, has_any], dim=-1)  # (N, fusion_dim+1)
            trust = self.gate_net(gate_in)            # (N, 1)  per-node gate

            z_v = trust * gru_z + (1.0 - trust) * mlp_z
        else:
            z_v = mlp_z

        return z_v
