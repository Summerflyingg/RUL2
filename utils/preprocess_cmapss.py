# utils/preprocess_cmapss.py
import numpy as np, os, argparse

def load_txt(path):
    # NASA txt: 6 + 26 + 3 = 35 列，其中传感器 1~21，我们保留 14 个有效传感器（按常用列表）
    cols_keep = [  # 14 sensors used by most papers
        2, 3, 4, 7, 8, 9, 11, 12, 13, 15, 17, 20, 21, 23
    ]
    data = np.loadtxt(path)
    unit_ids = data[:, 0].astype(int)
    sensors = data[:, cols_keep]
    return unit_ids, sensors.astype(np.float32)

def main(raw_root, out_root, subset):
    train_path = os.path.join(raw_root, f"train_{subset}.txt")
    uid, sens = load_txt(train_path)
    for u in np.unique(uid):
        arr = sens[uid == u]  # (T,14)
        os.makedirs(os.path.join(out_root, subset), exist_ok=True)
        np.save(os.path.join(out_root, subset, f"unit{u:03d}.npy"), arr)
    print(f"{subset}: wrote {len(np.unique(uid))} unit files.")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--raw_root", default="data/raw")
    p.add_argument("--out_root", default="data")
    p.add_argument("--subsets", nargs="+", default=["FD004"])
    args = p.parse_args()
    for sub in args.subsets:
        main(args.raw_root, args.out_root, sub)
# python utils/preprocess_cmapss.py --raw_root data/raw --out_root data
