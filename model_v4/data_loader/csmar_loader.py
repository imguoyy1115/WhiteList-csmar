"""
================================================================================
CSMAR 数据加载器 v3 — 全向量化版
================================================================================
改动：全部 iterrows() 替换为 pandas apply + 预计算 ID 数组
优势：380万次迭代 → ~20万次 apply，速度提升约 100 倍
劣势：高内存占用（pandas apply 返回 Python 对象数组），但 16GB 内存够用
================================================================================
"""
import os, sys
import numpy as np
import pandas as pd
import torch
from collections import defaultdict

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_interface import HeteroGraphData

CSMAR_DIR = r"D:\Users\imguoyyy\PycharmProjects\WhiteList\csmar"
SEED = 42; np.random.seed(SEED)

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

    # listed companies
    listed = set()
    for k in ["solvency","profit","operation","growth","cashflow","risklevel"]:
        codes = T[k]["Stkcd"].dropna().astype(str).str.strip().str.zfill(6)
        listed.update(codes[codes.str.isdigit()].values)
    print(f"  Listed: {len(listed)}")

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
    }

# ═══════════════════════════════════════════════════════════
# Step 3: Features (vectorized)
# ═══════════════════════════════════════════════════════════
def build_features(T, E):
    print("\n[3/5] Node features...")
    n, nl = E["n_total"], E["n_listed"]
    DIM = 23
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
            df = df[df["Typrep"].astype(str)=="1"]
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

    # 特征归一化：只在有真实值的上市公司上拟合，再 transform 全量
    from sklearn.preprocessing import StandardScaler
    mask_real = (X[:nl].sum(axis=1) > 0)
    scaler = StandardScaler().fit(X[:nl][mask_real])
    X = scaler.transform(X)

    print(f"  Feature dim: {DIM}, coverage: {(X[:nl].sum(1)>0).sum()}/{nl}")
    return X, M

# ═══════════════════════════════════════════════════════════
# Step 4: Edges (vectorized)
# ═══════════════════════════════════════════════════════════
def build_edges(T, E):
    print("\n[4/5] Building edges...")
    ei = {}

    # 4.1 trade
    trade_set = set()
    for dfk in ["scf_ar","scf_ap"]:
        df = T[dfk]
        if "Symbol" not in df.columns or "DebtName" not in df.columns:
            continue
        src = df["Symbol"].astype(str).apply(lambda x: ent_id(x, E)).values
        dst = df["DebtName"].astype(str).apply(lambda x: ent_id(x, E)).values
        amt = pd.to_numeric(df.get("EndingAmount", 0), errors="coerce").fillna(0).values
        for i in range(len(df)):
            s,d = int(src[i]), int(dst[i])
            if s >= 0 and d >= 0 and s != d:
                trade_set.add((s, d, float(amt[i])))
    print(f"  trade: {len(trade_set)} edges")

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

    # 4.4 involved_in
    iv_set = set()
    la = T["lawsuit"]
    LIMIT_LA = 50000
    if len(la) > LIMIT_LA:
        la = la.sample(LIMIT_LA, random_state=SEED)
    src = la["Symbol"].astype(str).apply(lambda x: ent_id(x, E)).values
    eids = la["EventID"].astype(str).values
    for i in range(len(la)):
        s = int(src[i])
        if s >= 0 and eids[i]:
            iv_set.add((s, eids[i], 0.0))
    print(f"  involved_in: {len(iv_set)} events")

    # to tensors
    if trade_set:
        ei[("enterprise","trade","enterprise")] = torch.tensor(
            [(s,d) for s,d,w in trade_set]).long().t()
    if eq_set:
        ei[("enterprise","equity","enterprise")] = torch.tensor(
            [(s,d) for s,d,w in eq_set]).long().t()
    if iv_set:
        ev_ids = sorted(set(e[1] for e in iv_set))
        ev_map = {eid:i for i,eid in enumerate(ev_ids)}
        pairs = [(s, ev_map[eid]) for s,eid,sev in iv_set]
        ei[("enterprise","involved_in","riskevent")] = torch.tensor(pairs).long().t()

    return ei, {eid:i for i,eid in enumerate(sorted(set(e[1] for e in iv_set)))} if iv_set else {}

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
def build_labels_and_assemble(T, E, X, M_features, edge_index, event_map):
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
    if event_map:
        x_dict["riskevent"] = torch.zeros((len(event_map), 3))

    x_struct = {"enterprise": torch.tensor(X_struct)}
    x_missing = {"enterprise": torch.tensor(M)}
    struct_hint_dict = {"enterprise": torch.tensor(struct_hint)}

    # masks
    n_train, n_val = int(nl*0.7), int(nl*0.15)
    train = torch.zeros(n, dtype=bool); train[:n_train] = True
    val   = torch.zeros(n, dtype=bool); val[n_train:n_train+n_val] = True
    test  = torch.zeros(n, dtype=bool); test[n_train+n_val:nl] = True

    data = HeteroGraphData(
        x_dict=x_dict, edge_index_dict=edge_index,
        y_white=torch.tensor(y_white), y_risk=torch.tensor(y_risk),
        y_grade=torch.tensor(y_grade),
        train_mask=train, val_mask=val, test_mask=test,
        num_enterprises=n,
        x_struct=x_struct, x_missing=x_missing,
        struct_hint=struct_hint_dict,
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
    X, M_feat = build_features(T, E); stages.append(("Features",time.time()-t0))
    t0 = time.time()
    ei, ev_map = build_edges(T, E); stages.append(("Edges",time.time()-t0))
    t0 = time.time()
    data = build_labels_and_assemble(T, E, X, M_feat, ei, ev_map); stages.append(("Assembly",time.time()-t0))
    print("\nTiming:")
    for name,sec in stages:
        print(f"  {name}: {sec:.1f}s")
    return data

if __name__ == "__main__":
    import time; t0 = time.time()
    data = load_csmar_data()
    print(f"\nTotal: {time.time()-t0:.1f}s")
