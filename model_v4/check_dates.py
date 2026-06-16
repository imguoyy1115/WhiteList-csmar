"""检查 CSMAR 各表的日期范围"""
import os, pandas as pd

CSMAR = r"D:\Users\imguoyyy\PycharmProjects\WhiteList\csmar"
folders = sorted(os.listdir(CSMAR))

for folder in folders:
    path = os.path.join(CSMAR, folder)
    if not os.path.isdir(path):
        continue
    files = [f for f in os.listdir(path) if f.endswith('.csv')]
    if not files:
        continue
    fpath = os.path.join(path, files[0])
    try:
        df = pd.read_csv(fpath, encoding='utf-8-sig', low_memory=False, on_bad_lines='skip')
    except:
        continue

    # 找日期列
    date_cols = [c for c in df.columns if any(k in c.lower() for k in ['accper', 'enddate', 'date', 'declaredate', 'eventsigndate'])]
    if not date_cols:
        # 打印前8列名供参考
        print(f"{folder}: 无日期列, cols={list(df.columns)[:6]}")
        continue

    for dc in date_cols[:2]:
        d = pd.to_datetime(df[dc], errors='coerce')
        valid = d.dropna()
        if len(valid) > 0:
            print(f"{folder}: {dc} = {valid.min().date()} ~ {valid.max().date()}  ({len(df)} rows, {len(valid)} valid dates)")
        else:
            print(f"{folder}: {dc} = 无有效日期  ({len(df)} rows)")
