"""実データが無いときの動作確認用サンプル（data/sample/）を作る。値はすべて架空。"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.geo import LocalXY  # noqa: E402
from src.loaders import load_config, resolve  # noqa: E402


def main(seed: int = 0):
    cfg = load_config()
    out = resolve(cfg["paths"]["sample_dir"])
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    # 架空の街区：駅ポートの東南東 約500m に 18列×10行 の区画（172戸を使用）
    st = cfg["destinations"]["station"]["port"]
    proj = LocalXY(st["lat"], st["lon"])
    xs, ys = np.meshgrid(np.arange(18) * 16.0 + 350, np.arange(10) * 18.0 - 160)
    x, y = xs.ravel()[:172] + rng.normal(0, 2, 172), ys.ravel()[:172] + rng.normal(0, 2, 172)
    lat, lon = proj.to_latlon(x, y)
    pd.DataFrame({"house_id": np.arange(1, 173), "lat": lat.round(6), "lon": lon.round(6)}).to_csv(
        out / "houses.csv", index=False)

    # 架空アンケート（23件）：往路の時間帯別トリップ数
    rows = []
    for rid in range(1, 24):
        if rng.random() < 0.6:  # 通勤者
            rows.append((rid, int(rng.choice([6, 7, 7, 8])), "通勤", "駅", 1))
        if rng.random() < 0.15:
            rows.append((rid, 7, "通学", "駅", 1))
        for _ in range(rng.poisson(0.5)):
            rows.append((rid, int(rng.choice([10, 11, 14, 15, 17, 18])), "買物", "イオン", 1))
        if rng.random() < 0.1:
            rows.append((rid, int(rng.choice([9, 10])), "通院", "駅", 1))
    pd.DataFrame(rows, columns=["回答ID", "時間帯", "目的", "目的地", "トリップ数"]).to_csv(
        out / "survey.csv", index=False)
    print(f"サンプルを作成しました: {out}（架空データ）")


if __name__ == "__main__":
    main()
