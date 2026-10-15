"""全ポート数（1〜max_ports）× 車種の最適ポート位置を data/output/ に書き出す。

使い方:
    python scripts/export_ports.py [--day 平日] [--conversion 0.3]
出力:
    data/output/optimal_ports_all.csv / .geojson   （車種・ポート数ごとの全ポート）
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.export import ports_table, save_ports  # noqa: E402
from src.loaders import load_config, load_network, load_survey, survey_path  # noqa: E402
from src.sensitivity import run_scenario  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--day", default="平日")
    ap.add_argument("--conversion", type=float, default=0.3)
    args = ap.parse_args()
    cfg = load_config()
    net = load_network(cfg)
    cfg["simulation"]["n_households"] = cfg["simulation"].get("n_households") or len(net.houses)
    survey = load_survey(survey_path(cfg)[0], cfg)
    purposes = cfg.get("survey_xlsx", {}).get("default_modes") if "day" in survey.columns else None
    opts = {"day": args.day if "day" in survey.columns else None}
    tables = []
    for v, vc in cfg["vehicles"].items():
        for k in range(1, cfg["simulation"]["max_ports"] + 1):
            scn = run_scenario(net, survey, cfg, conversion=args.conversion, purposes=purposes,
                               station_share=None, vehicle=v, k=k, opts=opts)
            t = ports_table(scn, net, cfg)
            t.insert(0, "車種", vc["label"])
            t.insert(1, "ポート数", k)
            tables.append(t)
    csv, gj = save_ports(pd.concat(tables, ignore_index=True), "optimal_ports_all", cfg)
    print(f"保存しました: {csv}\n            {gj}")


if __name__ == "__main__":
    main()
