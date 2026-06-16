"""列出所有 CSMAR 表的全部字段"""
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
        df = pd.read_csv(fpath, encoding='utf-8-sig', low_memory=False, on_bad_lines='skip', nrows=1)
    except:
        continue
    print(f"\n{'='*70}")
    print(f"【{folder}】")
    print(f"  文件: {files[0]}")
    print(f"  总行数: (读取中...)")
    print(f"  字段数: {len(df.columns)}")
    print(f"  字段列表:")
    for i, c in enumerate(df.columns):
        print(f"    [{i:2d}] {c}")
