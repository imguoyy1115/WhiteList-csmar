"""
================================================================================
CSMAR 数据加载器 v4 — 全向量化 + 数据过滤版
================================================================================
v4 新增三道数据过滤：
  1. 排除金融业（IndustryCode 以 J 开头）→ 提升 Scaler 拟合质量
  2. 标记 ST/异常公司（StateTypeCode ≠ "1"）→ 不删除但追踪
  3. 时间窗口切边（trade≥2022, lawsuit≥2020）→ 减少噪声 + 降显存
================================================================================
"""
import os, sys
import numpy as np
import pandas as pd
import torch
from collections import defaultdict

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_interface import HeteroGraphData
from config import SEED

CSMAR_DIR = r"D:\Users\imguoyyy\PycharmProjects\WhiteList\csmar"
SEED = 42; np.random.seed(SEED)

# ═══════════════════════════════════════════════════════════
# 过滤配置
# ═══════════════════════════════════════════════════════════
EXCLUDE_FINANCE = True       # 排除金融业（J 开头行业代码）
MIN_TRADE_DATE = "2022-01-01"  # trade 边只保留 2022 年及之后
MIN_LAWSUIT_DATE = "2020-01-01"  # 诉讼只保留 2020 年及之后

# ── helpers ──
def _clean(s):
    return str(s).replace("（","(").replace("）",")").replace(" ","").replace("*","").strip()

def _read(folder, fname):
    p = os.path.join(CSMAR_DIR, folder, fname)
    return pd.read_csv(p, encoding="utf-8-sig", low_memory=False, on_bad_lines="skip")

def _stock(s):
    s = str(s).strip()
    return s.zfill(6) if s.replace('.','').isdigit() else s

# ── global entity cache ──
_ent_cache = {}
def ent_id(code_or_name, E):
    key = str(code_or_name)
    if key in _ent_cache:
        return _ent_cache[key]
    code = _stock(key) if key[:1].isdigit() else ""
    result = -1
    if code in E["code2id"]:
        result = E["code2id"][code]
    else:
        nm = _clean(key)
        if nm in E["cp_matched"]:
            result = E["code2id"][E["cp_matched"][nm]]
        elif nm in E["s2id"]:
            result = E["s2id"][nm]
    _ent_cache[key] = result
    return result

# ═══════════════════════════════════════════════════════════
# Step 1: Load
# ═══════════════════════════════════════════════════════════
def load_all():
    print("[1/5] Loading CSVs...")
    T = {}
    T["solvency"]   = _read("偿债能力164658534","FI_T1.csv")
    T["profit"]     = _read("盈利能力164846481","FI_T5.csv")
    T["operation"]  = _read("经营能力165028388","FI_T4.csv")
    T["growth"]     = _read("发展能力165246963","FI_T8.csv")
    T["cashflow"]   = _read("现金流分析165410036","FI_T6.csv")
    T["risklevel"]  = _read("风险水平165544812","FI_T7.csv")
    T["shareholder"]= _read("十大股东文件170142830","HLD_Shareholders.csv")
    T["controller"] = _read("上市公司控制人文件170232148","HLD_Contrshr.csv")
    T["lawsuit"]    = _read("诉讼仲裁明细表170517712","LA_DETAIL.csv")
    T["scf_ov"]     = _read("供应链金融总体情况表163329193","SCF_Overview.csv")
    T["scf_stats"]  = _read("供应链金融业务统计表163124673","SCF_BusinessStats.csv")
    T["scf_credit"] = _read("商业信用及话语权表162729271","SCF_TradeCreditPower.csv")
    T["scf_ar"]     = _read("应收账款主要欠款人信息表161752742","SCF_ARMajorDebtors.csv")
    T["scf_ap"]     = _read("预付账款主要欠款人信息表162411085","SCF_APMajorCreditors.csv")
    for k,v in T.items(): print(f"  {k}: {len(v)} rows")
    return T

# ═══════════════════════════════════════════════════════════
# Step 2: Entities (vectorized)
# ═══════════════════════════════════════════════════════════
def build_entities(T):
    print("\n[2/5] Entity discovery...")
    filters = {"finance_excluded": 0, "st_marked": 0}

    # ── 行业信息：排除金融业 ──
    finance_codes = set()
    if EXCLUDE_FINANCE:
        for dfk in ["scf_ov", "scf_credit"]:
            df = T[dfk]
            if "IndustryCode" not in df.columns or "Symbol" not in df.columns:
                continue
            mask_fin = df["IndustryCode"].astype(str).str.strip().str.startswith("J")
            codes = df.loc[mask_fin, "Symbol"].astype(str).str.strip().str.zfill(6)
            finance_codes.update(codes[codes.str.isdigit()].values)
        print(f"  金融业企业: {len(finance_codes)}")

    # ── ST 标记 ──
    st_codes = set()
    for dfk in ["scf_credit", "scf_ar", "scf_ap"]:
        df = T[dfk]
        if "StateTypeCode" not in df.columns or "Symbol" not in df.columns:
            continue
        mask_st = df["StateTypeCode"].astype(str).str.strip() != "1"
        codes = df.loc[mask_st, "Symbol"].astype(str).str.strip().str.zfill(6)
        st_codes.update(codes[codes.str.isdigit()].values)
    print(f"  ST/异常状态企业: {len(st_codes)}")

    # listed companies (排除金融业)
    listed = set()
    for k in ["solvency","profit","operation","growth","cashflow","risklevel"]:
        codes = T[k]["Stkcd"].dropna().astype(str).str.strip().str.zfill(6)
        listed.update(codes[codes.str.isdigit()].values)
    filters["finance_excluded"] = len(listed & finance_codes)
    if EXCLUDE_FINANCE:
        listed -= finance_codes  # ← 核心过滤：踢掉金融业
    print(f"  Listed: {len(listed)} (剔除金融业 {filters['finance_excluded']} 家)")

    # name→code hashmap (vectorized: 1M rows → 50K unique names)
    name2code = {}
    pairs = [("scf_ov","Symbol","ShortName"),("scf_credit","Symbol","ShortName"),
             ("shareholder","Stkcd","S0301a"),("controller","Stkcd","S0701a")]
    for dfk, code_col, name_col in pairs:
        df = T[dfk]
        if code_col not in df.columns or name_col not in df.columns:
            continue
        codes = df[code_col].dropna().astype(str).str.strip().str.zfill(6)
        names = (df[name_col].dropna().astype(str)
                 .str.replace("（","(",regex=False)
                 .str.replace("）",")",regex=False)
                 .str.replace(" ","",regex=False)
                 .str.replace("*","",regex=False).str.strip())
        mask = codes.str.isdigit() & (names.str.len() > 0)
        for c,n in zip(codes[mask].values, names[mask].values):
            name2code[n] = c
    print(f"  name2code: {len(name2code)} entries")

    # counterparty names from trade tables
    cp_seen = set()
    for dfk in ["scf_ar","scf_ap"]:
        df = T[dfk]
        nms = (df["DebtName"].dropna().astype(str)
               .str.replace("（","(",regex=False)
               .str.replace("）",")",regex=False)
               .str.replace(" ","",regex=False)
               .str.replace("*","",regex=False).str.strip())
        cp_seen.update(nms[nms.str.len() > 1].values)
    print(f"  CP names (unique): {len(cp_seen)}")

    # match
    cp_matched, cp_standalone = {}, set()
    for nm in cp_seen:
        if nm in name2code and name2code[nm] in listed:
            cp_matched[nm] = name2code[nm]
        else:
            cp_standalone.add(nm)
    print(f"  Matched: {len(cp_matched)}, Standalone: {len(cp_standalone)}")

    # ---- 预扫描股东名称，防止 build_edges 动态分配越界 ID ----
    sh_df = T["shareholder"]
    sh_names_raw = sh_df["S0301a"].dropna().astype(str)
    sh_names_clean = (sh_names_raw
        .str.replace("（","(",regex=False)
        .str.replace("）",")",regex=False)
        .str.replace(" ","",regex=False)
        .str.replace("*","",regex=False).str.strip())
    existing = cp_seen | set(cp_matched.keys()) | cp_standalone | set(name2code.keys())
    sh_new = 0
    for nm in sh_names_clean.unique():
        nm_str = str(nm)
        if len(nm_str) <= 1 or nm_str in existing:
            continue
        cp_standalone.add(nm_str)
        sh_new += 1
    print(f"  Shareholder names added: {sh_new}")

    # global IDs
    all_codes = sorted(listed | set(cp_matched.values()))
    code2id = {c:i for i,c in enumerate(all_codes)}
    s2id = {nm: len(all_codes)+i for i,nm in enumerate(sorted(cp_standalone))}
    n_total = len(all_codes) + len(cp_standalone)
    print(f"  Total nodes: {n_total}")

    return {
        "n_total": n_total, "n_listed": len(listed),
        "code2id": code2id, "s2id": s2id,
        "name2code": name2code, "cp_matched": cp_matched,
        "st_codes": st_codes, "finance_codes": finance_codes,
    }

# ═══════════════════════════════════════════════════════════
# Step 3: Features (vectorized)
# ═══════════════════════════════════════════════════════════
def build_features(T, E):
    print("\n[3/5] Node features...")
    n, nl = E["n_total"], E["n_listed"]
    DIM = 25  # 23 维财务特征 + 2 维诉讼严重程度
    X = np.zeros((n, DIM), dtype=np.float32)
    M = np.ones((n, DIM), dtype=np.float32)  # 缺失指示：1=缺失 0=已填充
    col = 0

    # 3.1 financial indicators
    fin_map = {
        "solvency":{"F010101A":"CR","F010701B":"DAR","F011201A":"ICR"},
        "profit":{"F050201B":"ROA","F050501B":"ROE"},
        "operation":{"F040201B":"ART","F040801B":"APT"},
        "growth":{"F080601A":"TAGR","F081601B":"REVGR"},
        "cashflow":{"F060101B":"CFONI"},
        "risklevel":{"F070101B":"DFL","F070201B":"DOL"},
    }
    for tab_key, fmap in fin_map.items():
        df = T[tab_key].copy()
        df["code"] = df["Stkcd"].astype(str).apply(_stock)
        if "Typrep" in df.columns:
            # 取半年报(B)，一年两期(6月+12月)，方便后续时序建模
            df = df[df["Typrep"].astype(str).str.upper() == "B"]
        df = df.sort_values("Accper").groupby("code").last().reset_index()
        eid_arr = df["code"].apply(lambda c: ent_id(c, E)).values
        for fcol in fmap:
            v = pd.to_numeric(df[fcol], errors="coerce").fillna(0).clip(-10,100).values
            for i in range(len(df)):
                eid = eid_arr[i]
                if 0 <= eid < nl:
                    X[eid, col] = v[i]
                    M[eid, col] = 0.0  # 标记为已填充
            col += 1

    # 3.2 trade credit (vectorized: precompute eid, then loop)
    df_cr = T["scf_credit"].copy()
    eid_cr = df_cr["Symbol"].astype(str).apply(lambda x: ent_id(x, E)).values
    cr_cols = ["AccountPayable","Prepayment","AccountReceivable","TotalAssets",
               "ProvidedTradeCredit","ObtainedTradeCredit","BankLoanSize","SupplierPower1"]
    for fcol in cr_cols:
        if fcol in df_cr.columns:
            v = pd.to_numeric(df_cr[fcol], errors="coerce").fillna(0).values
            for i in range(len(df_cr)):
                eid = eid_cr[i]
                if 0 <= eid < nl:
                    X[eid, col] = v[i]
                    M[eid, col] = 0.0  # 标记为已填充
        col += 1

    # 3.3 SCF flags
    df_ov = T["scf_ov"].copy()
    eid_ov = df_ov["Symbol"].astype(str).apply(lambda x: ent_id(x, E)).values
    for fcol in ["IsSCFBusiness","IsSCDigitalization","IsSCFServicePlatform"]:
        if fcol in df_ov.columns:
            v = pd.to_numeric(df_ov[fcol], errors="coerce").fillna(0).values
            for i in range(len(df_ov)):
                eid = eid_ov[i]
                if 0 <= eid < nl:
                    X[eid, col] = v[i]
                    M[eid, col] = 0.0  # 标记为已填充
        col += 1

    # 3.4 诉讼严重程度特征（LAValue 按企业聚合 + 缩尾 + 双指标）
    #     对 LAValue 做缩尾处理（下限 1 元，上限 P99.5=14.2 亿），缺失填中位数
    #     指标1: log(1 + 时间窗口内 LAValue 总和) — 累计风险暴露
    #     指标2: log(1 + 加权严重分) — 小案(<500万)=1, 中案(500-5000万)=3, 大案(>5000万)=10
    la_full = T["lawsuit"]
    min_la_ts = pd.Timestamp(MIN_LAWSUIT_DATE)
    la_dates = None
    for dc in ["EventSignDate", "DeclareDate"]:
        if dc in la_full.columns:
            la_dates = pd.to_datetime(la_full[dc], errors="coerce")
            break
    la_val = pd.to_numeric(la_full["LAValue"], errors="coerce")
    la_median = la_val.median()
    la_val = la_val.fillna(la_median).clip(lower=1, upper=1_420_000_000)
    la_src = la_full["Symbol"].astype(str).apply(lambda x: ent_id(x, E)).values

    ent_la_sum = defaultdict(float)
    ent_severity = defaultdict(float)
    for i in range(len(la_full)):
        if la_dates is not None:
            d = la_dates.iloc[i]
            if pd.notna(d) and d < min_la_ts:
                continue
        eid = int(la_src[i])
        if eid < 0 or eid >= nl:
            continue
        v = la_val.iloc[i]
        ent_la_sum[eid] += v
        # 严重程度加权
        if v < 5_000_000:
            ent_severity[eid] += 1
        elif v < 50_000_000:
            ent_severity[eid] += 3
        else:
            ent_severity[eid] += 10

    for i in range(n):
        X[i, col] = np.log1p(ent_la_sum.get(i, 0))
        M[i, col] = 0.0
    col += 1
    for i in range(n):
        X[i, col] = np.log1p(ent_severity.get(i, 0))
        M[i, col] = 0.0
    col += 1

    # 特征归一化：只在有真实值且非 ST 的上市公司上拟合，再 transform 全量
    from sklearn.preprocessing import StandardScaler
    st_codes = E.get("st_codes", set())
    code2id = E["code2id"]
    # 找到 ST 公司对应的 ID
    st_ids = set()
    for code in st_codes:
        if code in code2id:
            st_ids.add(code2id[code])
    mask_normal = np.ones(nl, dtype=bool)
    for sid in st_ids:
        if sid < nl:
            mask_normal[sid] = False
    mask_real = (X[:nl].sum(axis=1) > 0)
    mask_fit = mask_real & mask_normal  # 有真实特征 + 非 ST
    scaler = StandardScaler().fit(X[:nl][mask_fit])
    X = scaler.transform(X)

    print(f"  Feature dim: {DIM}, coverage: {mask_real.sum()}/{nl}"
          f" (ST: {len(st_ids & set(range(nl)))})")

    # 3.5 季度序列特征 X_seq (N, 4, DIM+1) — v4.3 Temporal GRU
    #     前 DIM 列：季度特征（当前为 Q4 重复，待真实季度数据就绪后替换）
    #     第 DIM 列：has_feature_q — 该季度是否有真实数据
    X_seq = np.zeros((n, 4, DIM + 1), dtype=np.float32)
    for q in range(4):
        X_seq[:, q, :DIM] = X.copy()  # 当前用 Q4 特征作为各季度基线
        X_seq[:, q, DIM] = (X.sum(axis=-1) > 0).astype(np.float32)

    # 应用 scaler 到 X 和各季度
    X = scaler.transform(X)
    for q in range(4):
        X_seq[:, q, :DIM] = scaler.transform(X_seq[:, q, :DIM])

    return X, M, X_seq

# ═══════════════════════════════════════════════════════════
# Step 4: Edges (vectorized)
# ═══════════════════════════════════════════════════════════
def build_edges(T, E):
    print("\n[4/5] Building edges...")
    ei = {}

    # 4.1 trade（加时间窗口过滤）
    trade_set = set()
    trade_skipped = 0
    min_trade_ts = pd.Timestamp(MIN_TRADE_DATE)
    for dfk in ["scf_ar","scf_ap"]:
        df = T[dfk]
        if "Symbol" not in df.columns or "DebtName" not in df.columns:
            continue
        # 时间过滤
        if "EndDate" in df.columns:
            dates = pd.to_datetime(df["EndDate"], errors="coerce")
        else:
            dates = None
        src = df["Symbol"].astype(str).apply(lambda x: ent_id(x, E)).values
        dst = df["DebtName"].astype(str).apply(lambda x: ent_id(x, E)).values
        amt = pd.to_numeric(df.get("EndingAmount", 0), errors="coerce").fillna(0).values
        for i in range(len(df)):
            if dates is not None:
                d = dates.iloc[i]
                if pd.notna(d) and d < min_trade_ts:
                    trade_skipped += 1
                    continue
            s,d_id = int(src[i]), int(dst[i])
            if s >= 0 and d_id >= 0 and s != d_id:
                trade_set.add((s, d_id, float(amt[i])))
    print(f"  trade: {len(trade_set)} edges (跳过 {trade_skipped} 条过期 trade)")

    # 4.2 equity（采样20万行）
    eq_set = set()
    df_sh = T["shareholder"]
    LIMIT = 200000
    if len(df_sh) > LIMIT:
        df_sh = df_sh.sample(LIMIT, random_state=SEED)
    dst = df_sh["Stkcd"].astype(str).apply(lambda x: ent_id(x, E)).values
    ratio = pd.to_numeric(df_sh["S0306a"], errors="coerce").fillna(0).values
    names = df_sh["S0301a"].astype(str)
    for i in range(len(df_sh)):
        d,r = int(dst[i]), float(ratio[i])
        if d < 0 or r <= 0: continue
        s = ent_id(names.iloc[i], E)
        if s < 0:
            # 名字未注册 → 跳过（已在 build_entities 预扫描）
            continue
        if s != d:
            eq_set.add((int(s), d, r))
    print(f"  equity: {len(eq_set)} edges")

    # 4.3 same_industry
    print(f"  same_industry: 0 (skipped)")

    # 4.4 involved_in — 已移除（v4.2）
    #     原因：边方向 enterprise→riskevent，消息只流入 riskevent 节点，
    #     下游只取 h["enterprise"]，riskevent embedding 从未被使用。
    #     诉讼信息已通过企业级特征（log 总金额 + 加权严重分）进入模型。
    print(f"  involved_in: 0 (已移除，诉讼信息已并入企业特征)")

    # to tensors
    if trade_set:
        ei[("enterprise","trade","enterprise")] = torch.tensor(
            [(s,d) for s,d,w in trade_set]).long().t()
    if eq_set:
        ei[("enterprise","equity","enterprise")] = torch.tensor(
            [(s,d) for s,d,w in eq_set]).long().t()

    return ei

# ═══════════════════════════════════════════════════════════
# Step 4.5: 结构特征（邻域聚合 + 图统计量）
# ═══════════════════════════════════════════════════════════
def compute_structural_features(X, edge_index, n, nl):
    """
    从图结构计算两种产物：
      X_struct:      (n, D) 邻域聚合特征（维度对齐 X，可直接替代）
      struct_hint:   (n, S) 图结构统计量（少量维度，供门控阀感知用）
    """
    D = X.shape[1]
    X_t = torch.tensor(X)
    ei_ent = {}  # 只取 enterprise→enterprise 的边
    for (src, rel, dst), ei in edge_index.items():
        if src == "enterprise" and dst == "enterprise":
            ei_ent[rel] = ei
        elif src == "enterprise" and dst == "riskevent":
            ei_ent[rel] = ei  # involved_in: 出边也算

    # ---- 邻域聚合：对每个节点，聚合其邻居的特征均值 ----
    X_struct = np.zeros_like(X)
    for rel, ei in ei_ent.items():
        src, dst = ei[0].numpy(), ei[1].numpy()
        # dst ← src: 从 src 聚合到 dst
        agg = np.zeros((n, D), dtype=np.float64)
        cnt = np.zeros(n, dtype=np.int32)
        np.add.at(agg, dst, X[src])
        np.add.at(cnt, dst, 1)
        # src ← dst: 反向也聚合（无向化处理）
        np.add.at(agg, src, X[dst])
        np.add.at(cnt, src, 1)
        # 对 cnt>0 的节点取均值
        mask = cnt > 0
        agg[mask] /= cnt[mask, np.newaxis]
        X_struct += agg
    # 归一化（除以边类型数）
    n_rel = max(len(ei_ent), 1)
    X_struct /= n_rel

    # 孤立节点用上市公司均值填充
    mask_isolated = (X_struct.sum(axis=1) == 0)
    listed_mean = X[:nl][X[:nl].sum(axis=1) > 0].mean(axis=0) if nl > 0 else X.mean(axis=0)
    X_struct[mask_isolated] = listed_mean

    # ---- 结构统计量（每个节点 8 维） ----
    S_DIM = 8
    struct_hint = np.zeros((n, S_DIM), dtype=np.float32)

    # [0] is_listed
    struct_hint[:nl, 0] = 1.0

    # [1] has_feature
    struct_hint[:, 1] = (X.sum(axis=1) != 0).astype(np.float32)

    # [2-4] degree per edge type (trade, equity, involved_in)
    for rel, ei in ei_ent.items():
        col = {"trade": 2, "equity": 3, "involved_in": 4}.get(rel, -1)
        if col < 0:
            continue
        deg = np.bincount(ei[0].numpy(), minlength=n) + np.bincount(ei[1].numpy(), minlength=n)
        struct_hint[:, col] += deg.astype(np.float32)

    # [5] total degree
    struct_hint[:, 5] = struct_hint[:, 2:5].sum(axis=1)

    # [6] log(1+degree) — 解决长尾分布
    struct_hint[:, 6] = np.log1p(struct_hint[:, 5])

    # [7] feature coverage ratio（仅对上市公司有效）
    for i in range(nl):
        struct_hint[i, 7] = (X[i] != 0).mean()

    # 归一化度相关特征到 [0,1]
    for c in [2, 3, 4, 5, 6]:
        mx = struct_hint[:, c].max()
        if mx > 0:
            struct_hint[:, c] /= mx

    return X_struct.astype(np.float32), struct_hint.astype(np.float32)

# ═══════════════════════════════════════════════════════════
# Step 5: Labels + Assembly
# ═══════════════════════════════════════════════════════════
def build_labels_and_assemble(T, E, X, M_features, X_seq, edge_index):
    print("\n[5/5] Labels + assembly...")
    n, nl = E["n_total"], E["n_listed"]

    y_white = np.ones(n, dtype=np.float32)
    y_risk  = np.zeros(n, dtype=np.float32)
    y_grade = np.full(n, 2, dtype=int)

    # bad debt → risk
    bad = defaultdict(list)
    df_ar = T["scf_ar"]
    eid_ar = df_ar["Symbol"].astype(str).apply(lambda x: ent_id(x, E)).values
    ratio_ar = pd.to_numeric(df_ar.get("EndingBadDebtProRatio", 0), errors="coerce").fillna(0).values
    for i in range(len(df_ar)):
        eid = int(eid_ar[i])
        if eid >= 0:
            bad[eid].append(float(ratio_ar[i]))
    for eid, ratios in bad.items():
        if np.mean(ratios) > 20:
            y_white[eid] = 0; y_risk[eid] = 1

    # execution → risk
    exec_cnt = defaultdict(int)
    la = T["lawsuit"]
    eid_la = la["Symbol"].astype(str).apply(lambda x: ent_id(x, E)).values
    exec_st = la["ExecutnStus"].fillna("").astype(str).values
    for i in range(len(la)):
        eid = int(eid_la[i])
        if eid >= 0 and "执行" in exec_st[i]:
            exec_cnt[eid] += 1
    for eid, cnt in exec_cnt.items():
        y_risk[eid] = 1
        if cnt >= 3: y_white[eid] = 0

    # grade
    for i in range(nl):
        if y_white[i]==1 and y_risk[i]==0: y_grade[i]=0
        elif y_white[i]==1 and y_risk[i]==1: y_grade[i]=1
        elif y_white[i]==0 and y_risk[i]==0: y_grade[i]=2
        else: y_grade[i]=3

    # feature dict + 结构特征 + 缺失指示
    print("  计算结构特征...")
    X_struct, struct_hint = compute_structural_features(X, edge_index, n, nl)

    # 缺失指示变量 M: 来自 build_features（在归一化前追踪，0=已填充 1=缺失）
    M = M_features.astype(np.float32)

    x_dict = {
        "enterprise": torch.tensor(X),
    }

    x_struct = {"enterprise": torch.tensor(X_struct)}
    x_missing = {"enterprise": torch.tensor(M)}
    struct_hint_dict = {"enterprise": torch.tensor(struct_hint)}

    # masks — 分层随机切分，确保 train/val/test 标签分布一致
    from sklearn.model_selection import train_test_split
    n_train, n_val = int(nl*0.7), int(nl*0.15)
    all_idx = np.arange(nl)
    y_listed = y_white[:nl]
    # 第一刀：train 70% vs rest 30%，按白名单标签分层
    train_idx, rest_idx = train_test_split(
        all_idx, test_size=0.3, stratify=y_listed, random_state=SEED)
    # 第二刀：rest 对半分 val 15% / test 15%
    y_rest = y_listed[rest_idx]
    val_idx, test_idx = train_test_split(
        rest_idx, test_size=0.5, stratify=y_rest, random_state=SEED)

    train = torch.zeros(n, dtype=bool); train[train_idx] = True
    val   = torch.zeros(n, dtype=bool); val[val_idx] = True
    test  = torch.zeros(n, dtype=bool); test[test_idx] = True

    data = HeteroGraphData(
        x_dict=x_dict, edge_index_dict=edge_index,
        y_white=torch.tensor(y_white), y_risk=torch.tensor(y_risk),
        y_grade=torch.tensor(y_grade),
        train_mask=train, val_mask=val, test_mask=test,
        num_enterprises=n,
        x_struct=x_struct, x_missing=x_missing,
        struct_hint=struct_hint_dict,
        x_seq=torch.tensor(X_seq),  # v4.3: (N, 4, DIM+1) 季度序列
    )
    data.summary()
    return data

# ═══════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════
def load_csmar_data():
    import time
    stages = []
    t0 = time.time()
    T = load_all(); stages.append(("Load",time.time()-t0))
    t0 = time.time()
    E = build_entities(T); stages.append(("Entities",time.time()-t0))
    t0 = time.time()
    X, M_feat, X_seq = build_features(T, E); stages.append(("Features",time.time()-t0))
    t0 = time.time()
    ei = build_edges(T, E); stages.append(("Edges",time.time()-t0))
    t0 = time.time()
    data = build_labels_and_assemble(T, E, X, M_feat, X_seq, ei); stages.append(("Assembly",time.time()-t0))
    print("\nTiming:")
    for name,sec in stages:
        print(f"  {name}: {sec:.1f}s")
    print("\n  数据过滤统计:")
    print(f"    金融业排除: {EXCLUDE_FINANCE}, ST标记: {len(E['st_codes'])} 家")
    print(f"    Trade时间窗口: ≥{MIN_TRADE_DATE}, 诉讼时间窗口: ≥{MIN_LAWSUIT_DATE}")
    return data

if __name__ == "__main__":
    import time; t0 = time.time()
    data = load_csmar_data()
    print(f"\nTotal: {time.time()-t0:.1f}s")
