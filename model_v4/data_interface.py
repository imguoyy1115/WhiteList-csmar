"""
================================================================================
数据接口定义 — 所有数据加载器必须遵守的契约
================================================================================
无论数据来自 CSMAR / 赛题官方 / 假数据，最终必须产出以下标准格式。
数据到手后：写一个 load_real_data() 返回这个 dataclass 的实例即可。
================================================================================
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
import torch


@dataclass
class HeteroGraphData:
    """
    ==========================================================================
    异构图数据 — 模型唯一标准输入接口
    ==========================================================================
    """
    # ---- 节点特征 ----
    x_dict: Dict[str, torch.Tensor]
    # {"enterprise": (N_ent, d_ent), "person": (N_per, d_per),
    #  "riskevent": (N_risk, d_risk), "project": (N_proj, d_proj)}

    # ---- 边索引（PyG 格式: [2, num_edges]） ----
    edge_index_dict: Dict[Tuple[str, str, str], torch.Tensor]
    # {("enterprise","trade","enterprise"): tensor([2, E_trade]), ...}

    # ---- 标签 ----
    y_white: torch.Tensor     # (N_ent,) 0/1 白名单标签
    y_risk: torch.Tensor      # (N_ent,) 0/1 风险标签（t+3）
    y_grade: torch.Tensor     # (N_ent,) 0-4 五级分层标签

    # ---- Mask（按时间切分） ----
    train_mask: torch.Tensor  # (N_ent,) bool
    val_mask: torch.Tensor    # (N_ent,) bool
    test_mask: torch.Tensor   # (N_ent,) bool

    # ---- 时序快照索引 ----
    snapshots: List[Dict] = field(default_factory=list)
    # [{"month": "2023-01", "active_nodes": [...], "active_edges": {...}}, ...]

    # ---- 元信息 ----
    num_enterprises: int = 0
    num_months: int = 36
    y_labels: List[str] = field(default_factory=lambda: ["非白名单", "白名单"])
    grade_labels: List[str] = field(default_factory=lambda: ["S","A","B","C","D"])

    # ---- 图结构指标（用于 Layer 5 的 ChainValue 维度） ----
    graph_metrics: Optional[Dict[str, torch.Tensor]] = None
    # {"pagerank": (N_ent,), "betweenness": (N_ent,), "kcore": (N_ent,),
    #  "degree": (N_ent,), "core_distance": (N_ent,)}

    # ---- 政策标签（用于 Layer 5 的 PolicyMatch 维度） ----
    policy_tags: Optional[torch.Tensor] = None  # (N_ent,) 0/1

    # ---- 自适应门控阀数据（Feature Gate） ----
    x_struct: Optional[Dict[str, torch.Tensor]] = None
    # {"enterprise": (N_ent, D)} 邻域聚合的结构特征，维度和 x_dict 对齐

    x_missing: Optional[Dict[str, torch.Tensor]] = None
    # {"enterprise": (N_ent, D)} 缺失指示变量 0=真实值 1=缺失

    struct_hint: Optional[Dict[str, torch.Tensor]] = None
    # {"enterprise": (N_ent, S)} 图结构统计量（度、PageRank 等），供门控阀感知

    # ---- 季度序列特征（v4.3 Temporal GRU） ----
    x_seq: Optional[torch.Tensor] = None  # (N_ent, 4, DIM+1) Q1-Q4 特征 + has_feature_q 标记


    def to(self, device: str) -> "HeteroGraphData":
        """一键移到 GPU/CPU"""
        self.x_dict = {k: v.to(device) for k, v in self.x_dict.items()}
        self.edge_index_dict = {k: v.to(device) for k, v in self.edge_index_dict.items()}
        self.y_white = self.y_white.to(device)
        self.y_risk = self.y_risk.to(device)
        self.y_grade = self.y_grade.to(device)
        self.train_mask = self.train_mask.to(device)
        self.val_mask = self.val_mask.to(device)
        self.test_mask = self.test_mask.to(device)
        if self.graph_metrics:
            self.graph_metrics = {k: v.to(device) for k, v in self.graph_metrics.items()}
        if self.policy_tags is not None:
            self.policy_tags = self.policy_tags.to(device)
        if self.x_struct:
            self.x_struct = {k: v.to(device) for k, v in self.x_struct.items()}
        if self.x_missing:
            self.x_missing = {k: v.to(device) for k, v in self.x_missing.items()}
        if self.struct_hint:
            self.struct_hint = {k: v.to(device) for k, v in self.struct_hint.items()}
        if self.x_seq is not None:
            self.x_seq = self.x_seq.to(device)
        return self

    def summary(self):
        """打印数据集概览"""
        print("=" * 60)
        print("  异构图数据概览")
        print("=" * 60)
        for ntype, x in self.x_dict.items():
            print(f"  {ntype}: {x.shape[0]} 节点, {x.shape[1]} 维特征")
        for etype, ei in self.edge_index_dict.items():
            print(f"  {etype[0]} --{etype[1]}--> {etype[2]}: {ei.shape[1]} 条边")
        print(f"  白名单正样本: {(self.y_white==1).sum().item()}")
        print(f"  训练/验证/测试: {self.train_mask.sum().item()}/{self.val_mask.sum().item()}/{self.test_mask.sum().item()}")

    def to_pyg_heterodata(self):
        """
        ==================================================================
        转换为 PyG 的 HeteroData，供 NeighborLoader 做 mini-batch 采样用。
        n_id 字段会保留每个 batch 中节点映射回全局 ID，用于索引 x_struct 等。
        ==================================================================
        """
        from torch_geometric.data import HeteroData

        d = HeteroData()
        for ntype, x in self.x_dict.items():
            d[ntype].x = x
        for etype, ei in self.edge_index_dict.items():
            d[etype].edge_index = ei

        # 标签（放在 enterprise 上）
        d["enterprise"].y_white = self.y_white
        d["enterprise"].y_risk = self.y_risk
        d["enterprise"].y_grade = self.y_grade
        d["enterprise"].train_mask = self.train_mask
        d["enterprise"].val_mask = self.val_mask
        d["enterprise"].test_mask = self.test_mask

        # 门控阀数据：作为节点属性挂载，NeighborLoader 会自动跟随采样
        if self.x_struct:
            for ntype, xs in self.x_struct.items():
                d[ntype].x_struct = xs
        if self.x_missing:
            for ntype, xm in self.x_missing.items():
                d[ntype].x_missing = xm
        if self.struct_hint:
            for ntype, sh in self.struct_hint.items():
                d[ntype].struct_hint = sh

        d.num_enterprises = self.num_enterprises
        return d
