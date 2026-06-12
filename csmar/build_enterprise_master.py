"""
================================================================================
企业主表构建脚本 v3 — 真正全向量化
从 15 张 CSMAR CSV 中提取所有企业，生成 Excel + SQLite
zsh:1: parse error near `\n'  — 全 zero-copy，零 for 行循环
================================================================================
"""
import os, sys
import pandas as pd
import sqlite3

CSMAR_DIR = r"D:\Users\imguoyyy\PycharmProjects\WhiteList\csmar"
OUT_DIR = CSMAR_DIR  # 输出也放在 csmar 文件夹

def _stock_vec(s: pd.Series) -> pd.Series:
    s = s.astype(str).str.strip().str.replace('"', '', regex=False)
    s = s.str.replace(r'\..*$', '', regex=True)
    s = s[s.str.match(r'^\d+$')]  # 只保留纯数字代码
    return s.str.zfill(6)

def _clean_vec(s: pd.Series) -> pd.Series:
    return (s.astype(str)
            .str.replace("（", "(", regex=False)
            .str.replace("）", ")", regex=False)
            .str.replace(" ", "", regex=False)
            .str.replace("*", "", regex=False)
            .str.strip())

def _read(folder, fname):
    p = os.path.join(CSMAR_DIR, folder, fname)
    return pd.read_csv(p, encoding="utf-8-sig", low_memory=False, on_bad_lines="skip")

print("=" * 60)
print("  企业主表构建 v3（全向量化）")
print("=" * 60)

# ═══════════════════════════════════════════════════════════
# Step 1: 收集所有 (code, name, table, category) — 全用 DataFrame
# ═══════════════════════════════════════════════════════════
CSV_LIST = [
    ("偿债能力164658534", "FI_T1.csv", "Stkcd", "ShortName", "财务报表"),
    ("盈利能力164846481", "FI_T5.csv", "Stkcd", "ShortName", "财务报表"),
    ("经营能力165028388", "FI_T4.csv", "Stkcd", "ShortName", "财务报表"),
    ("发展能力165246963", "FI_T8.csv", "Stkcd", "ShortName", "财务报表"),
    ("现金流分析165410036", "FI_T6.csv", "Stkcd", "ShortName", "财务报表"),
    ("风险水平165544812", "FI_T7.csv", "Stkcd", "ShortName", "财务报表"),
    ("供应链金融总体情况表163329193", "SCF_Overview.csv", "Symbol", "ShortName", "供应链金融"),
    ("供应链金融业务统计表163124673", "SCF_BusinessStats.csv", "Symbol", "ShortName", "供应链金融"),
    ("商业信用及话语权表162729271", "SCF_TradeCreditPower.csv", "Symbol", "ShortName", "供应链金融"),
    ("应收账款主要欠款人信息表161752742", "SCF_ARMajorDebtors.csv", "Symbol", "ShortName", "供应链金融"),
    ("预付账款主要欠款人信息表162411085", "SCF_APMajorCreditors.csv", "Symbol", "ShortName", "供应链金融"),
    ("十大股东文件170142830", "HLD_Shareholders.csv", "Stkcd", None, "公司治理"),
    ("上市公司控制人文件170232148", "HLD_Contrshr.csv", "Stkcd", None, "公司治理"),
    ("诉讼仲裁明细表170517712", "LA_DETAIL.csv", "Symbol", "ShortName", "法律诉讼"),
]

df_list = []

for folder, fname, code_col, name_col, category in CSV_LIST:
    try:
        df = _read(folder, fname)
    except Exception as e:
        print(f"  [跳过] {folder}/{fname}: {e}")
        continue
    if code_col not in df.columns:
        print(f"  [跳过] {folder}/{fname}: 无 {code_col} 列")
        continue

    codes = _stock_vec(df[code_col])
    table_label = f"{folder}/{fname}"

    if name_col and name_col in df.columns:
        names = _clean_vec(df[name_col].reindex(codes.index))
        names = names.where(names.str.len() > 0, "")
    else:
        names = pd.Series("", index=codes.index)

    # 对齐索引
    common_idx = codes.index.intersection(names.index)
    codes = codes.loc[common_idx]
    names = names.loc[common_idx]
    valid = codes.str.len() >= 2
    codes = codes[valid]
    names = names[valid]

    if len(codes) == 0:
        continue

    temp = pd.DataFrame({
        "code": codes.values,
        "name": names.values,
        "table": table_label,
        "category": category,
    })
    df_list.append(temp)
    print(f"  {folder}: {len(temp)} 条")

# 合并所有
print(f"\n合并 {sum(len(d) for d in df_list)} 条 → ", end="")
df_all_pairs = pd.concat(df_list, ignore_index=True)
print(f"{len(df_all_pairs)} 条")

# ═══════════════════════════════════════════════════════════
# Step 2: groupby code → 企业主表（有代码部分）
# ═══════════════════════════════════════════════════════════
print("聚合...")
g = df_all_pairs.groupby("code")

df_master = g.agg(
    name_list=("name", lambda x: " / ".join(sorted(set(x[x != ""]))[:5])),
    categories=("category", lambda x: ", ".join(sorted(set(x)))),
    table_count=("table", "nunique"),
    tables=("table", lambda x: ", ".join(sorted(set(x)))),
).reset_index()

df_master.columns = ["stock_code", "name", "categories", "table_count", "tables"]
df_master["has_financials"] = df_master["categories"].str.contains("财务报表", na=False)
df_master["has_scf"] = df_master["categories"].str.contains("供应链金融", na=False)
df_master["has_governance"] = df_master["categories"].str.contains("公司治理", na=False)
df_master["has_lawsuit"] = df_master["categories"].str.contains("法律诉讼", na=False)
df_master["source"] = "stock_code"

existing_names = set()
for n in df_master["name"].dropna():
    for part in str(n).split(" / "):
        if part:
            existing_names.add(part.lower())

print(f"  有代码企业: {len(df_master)}")

# ═══════════════════════════════════════════════════════════
# Step 3: 交易对手方（DebtName，无代码）
# ═══════════════════════════════════════════════════════════
print("交易对手方...")
cp_parts = []
for folder, fname in [
    ("应收账款主要欠款人信息表161752742", "SCF_ARMajorDebtors.csv"),
    ("预付账款主要欠款人信息表162411085", "SCF_APMajorCreditors.csv"),
]:
    try:
        df = _read(folder, fname)
    except:
        continue
    if "DebtName" not in df.columns or "Symbol" not in df.columns:
        continue
    dns = _clean_vec(df["DebtName"])
    dns = dns.dropna()
    dns = dns[dns.str.len() > 1]
    syms = _stock_vec(df["Symbol"].reindex(dns.index))
    temp = pd.DataFrame({
        "name": dns.values,
        "linked_code": syms.values,
    })
    cp_parts.append(temp)

if cp_parts:
    df_cp = pd.concat(cp_parts, ignore_index=True)
    # groupby name
    cp_agg = df_cp.groupby("name").agg(
        linked_codes=("linked_code", lambda x: ", ".join(sorted(set(x[x != ""]))[:5])),
        link_count=("linked_code", "nunique"),
    ).reset_index()
    # 排除已有
    cp_new_rows = []
    for _, row in cp_agg.iterrows():
        if row["name"].lower() not in existing_names:
            cp_new_rows.append({
                "stock_code": "",
                "name": row["name"],
                "categories": "交易对手方",
                "table_count": 1,
                "tables": f"关联: {row['linked_codes']}",
                "has_financials": False, "has_scf": False,
                "has_governance": False, "has_lawsuit": False,
                "source": "trade_counterparty",
            })
    df_cp_master = pd.DataFrame(cp_new_rows)
    print(f"  新增: {len(df_cp_master)}")
else:
    df_cp_master = pd.DataFrame()

# ═══════════════════════════════════════════════════════════
# Step 4: 股东（无代码）
# ═══════════════════════════════════════════════════════════
print("股东...")
sh_parts = []
for fname in ["HLD_Shareholders.csv", "HLD_Shareholders1.csv"]:
    try:
        df = _read("十大股东文件170142830", fname)
    except:
        continue
    if "S0301a" not in df.columns or "Stkcd" not in df.columns:
        continue
    shn = _clean_vec(df["S0301a"])
    shn = shn.dropna()
    shn = shn[shn.str.len() > 1]
    stk = _stock_vec(df["Stkcd"].reindex(shn.index))
    temp = pd.DataFrame({
        "name": shn.values,
        "linked_code": stk.values,
    })
    sh_parts.append(temp)

if sh_parts:
    df_sh = pd.concat(sh_parts, ignore_index=True)
    sh_agg = df_sh.groupby("name").agg(
        linked_codes=("linked_code", lambda x: ", ".join(sorted(set(x[x != ""]))[:5])),
        link_count=("linked_code", "nunique"),
    ).reset_index()
    all_known = existing_names | set(df_cp_master["name"].str.lower() if len(df_cp_master) > 0 else [])
    sh_new_rows = []
    for _, row in sh_agg.iterrows():
        if row["name"].lower() not in all_known:
            sh_new_rows.append({
                "stock_code": "",
                "name": row["name"],
                "categories": "股东",
                "table_count": 1,
                "tables": f"持股: {row['linked_codes']}",
                "has_financials": False, "has_scf": False,
                "has_governance": False, "has_lawsuit": False,
                "source": "shareholder",
            })
    df_sh_master = pd.DataFrame(sh_new_rows)
    print(f"  新增: {len(df_sh_master)}")
else:
    df_sh_master = pd.DataFrame()

# ═══════════════════════════════════════════════════════════
# Step 5: 最终合并
# ═══════════════════════════════════════════════════════════
print("\n最终合并...")
df_final = pd.concat([df_master, df_cp_master, df_sh_master], ignore_index=True)
df_final.insert(0, "enterprise_id", range(len(df_final)))
df_final = df_final.sort_values(["source", "stock_code", "name"]).reset_index(drop=True)
df_final["enterprise_id"] = range(len(df_final))

print(f"  总企业数: {len(df_final)}")
print(f"  有股票代码: {(df_final['stock_code'] != '').sum()}")
print(f"  有财务报表: {df_final['has_financials'].sum()}")
print(f"  交易对手方: {(df_final['source'] == 'trade_counterparty').sum()}")
print(f"  股东: {(df_final['source'] == 'shareholder').sum()}")

# CSV（比 Excel 快，29 万行不卡，没有字符限制）
csv_path = os.path.join(OUT_DIR, "enterprise_master.csv")
df_final.to_csv(csv_path, index=False, encoding="utf-8-sig")
print(f"[OK] CSV → {csv_path}")

# 统计摘要单独存一个 CSV
summary_path = os.path.join(OUT_DIR, "enterprise_master_summary.csv")
pd.DataFrame([
    ("总企业数", len(df_final)),
    ("有股票代码", (df_final["stock_code"] != "").sum()),
    ("无股票代码", (df_final["stock_code"] == "").sum()),
    ("有财务报表", df_final["has_financials"].sum()),
    ("有供应链金融数据", df_final["has_scf"].sum()),
    ("有公司治理数据", df_final["has_governance"].sum()),
    ("有法律诉讼数据", df_final["has_lawsuit"].sum()),
    ("仅交易对手方", (df_final["source"] == "trade_counterparty").sum()),
    ("仅股东", (df_final["source"] == "shareholder").sum()),
    ("出现在多张表", (df_final["table_count"] > 1).sum()),
], columns=["指标", "数值"]).to_csv(summary_path, index=False, encoding="utf-8-sig")
print(f"[OK] 统计摘要 → {summary_path}")

# SQLite
db_path = os.path.join(OUT_DIR, "enterprise_master.db")
conn = sqlite3.connect(db_path)
df_final.to_sql("enterprise_master", conn, if_exists="replace", index=False)
for col in ["stock_code", "source", "has_financials"]:
    conn.execute(f"CREATE INDEX IF NOT EXISTS idx_{col} ON enterprise_master({col})")
conn.close()
print(f"[OK] SQLite → {db_path}")

print("\n完成！")
