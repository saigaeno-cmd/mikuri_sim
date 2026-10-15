"""ポート位置（最適解）の表と保存。"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .loaders import Network, resolve


def ports_table(scn, net: Network, cfg: dict) -> pd.DataFrame:
    """ポートごとの座標・台数・区画・カバー戸数（街区側の最適ポート＋駅・イオン）。"""
    t = scn.fleet.table.copy()
    nb = t["種別"] == "街区"
    cands = net.candidates.loc[scn.solution.ports]
    t.loc[nb, "候補ID"] = cands["cand_id"].to_numpy()
    t.loc[nb, "緯度"] = cands["lat"].to_numpy()
    t.loc[nb, "経度"] = cands["lon"].to_numpy()
    t.loc[nb, "カバー戸数"] = np.bincount(scn.solution.assign, minlength=len(scn.solution.ports))
    for key, d in cfg["destinations"].items():
        t.loc[t["種別"] == key, ["緯度", "経度"]] = [d["port"]["lat"], d["port"]["lon"]]
    t["種別"] = t["種別"].replace({"station": "駅", "aeon": "イオン"})
    t["出発トリップ/日"] = t["出発トリップ/日"].round(1)
    return t


def save_ports(table: pd.DataFrame, name: str, cfg: dict) -> tuple[Path, Path]:
    """data/output/<name>.csv と .geojson に保存し、パスを返す。"""
    out = resolve(cfg["paths"].get("output_dir", "data/output"))
    out.mkdir(parents=True, exist_ok=True)
    csv, gj = out / f"{name}.csv", out / f"{name}.geojson"
    table.to_csv(csv, index=False, encoding="utf-8-sig")
    feats = [{"type": "Feature",
              "geometry": {"type": "Point", "coordinates": [float(r["経度"]), float(r["緯度"])]},
              "properties": {k: (None if pd.isna(v) else (v.item() if hasattr(v, "item") else v))
                             for k, v in r.items() if k not in ("緯度", "経度")}}
             for r in table.to_dict("records")]
    gj.write_text(json.dumps({"type": "FeatureCollection", "features": feats}, ensure_ascii=False, indent=1),
                  encoding="utf-8")
    return csv, gj
