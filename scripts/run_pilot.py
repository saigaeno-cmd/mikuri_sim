"""台数を固定した実証（既定10台）のシミュレーション。転換率5%刻みで、ポート数・車種の割合ごとの充足率と、
最良の組み合わせでの1年の結果を data/output/pilot/ に保存する。

使い方: python scripts/run_pilot.py [--vehicles 10] [--max-conversion 0.5] [--procs 4] [--districts ...]
"""
from __future__ import annotations

import argparse
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.loaders import load_config, load_network, load_survey, resolve, survey_path  # noqa: E402
from src.pilot import pilot_best, pilot_grid, pilot_year  # noqa: E402

G = {}


def _init(districts=None):
    cfg = load_config()
    net = load_network(cfg).subset(districts)
    cfg["simulation"]["n_households"] = cfg["simulation"].get("n_households") or len(net.houses)
    G.update(cfg=cfg, net=net, survey=load_survey(survey_path(cfg)[0], cfg))


def job(args):
    conv, n, n_rep = args
    cfg, net, sv = G["cfg"], G["net"], G["survey"]
    g = pilot_grid(net, cfg, sv, conv, n_total=n, n_rep=n_rep)
    b = pilot_best(g)
    ys, bm, cost = pilot_year(net, cfg, sv, conv, b)
    best = {**b.to_dict(), **{f"年_{k}": v for k, v in ys.items() if not isinstance(v, list)},
            "年間利用回数": cost["年間利用回数"], "年間総費用": cost["年間総費用"],
            **{f"月別充足率_{int(m)}": v for m, v in zip(bm["月"], bm["充足率"])}}
    return g, best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vehicles", type=int, default=10)
    ap.add_argument("--max-conversion", type=float, default=0.5)
    ap.add_argument("--n-rep", type=int, default=200)
    ap.add_argument("--procs", type=int, default=4)
    ap.add_argument("--districts", nargs="*", default=None)
    a = ap.parse_args()
    convs = [round(c, 2) for c in np.arange(0.05, a.max_conversion + 1e-9, 0.05)]
    with Pool(a.procs, initializer=_init, initargs=(a.districts,)) as pool:
        res = pool.map(job, [(c, a.vehicles, a.n_rep) for c in convs])
    cfg = load_config()
    out = resolve(cfg["paths"].get("output_dir", "data/output")) / "pilot"
    out.mkdir(parents=True, exist_ok=True)
    grid = pd.concat([g for g, _ in res], ignore_index=True)
    best = pd.DataFrame([b for _, b in res])
    grid.to_csv(out / f"pilot_{a.vehicles}_grid.csv", index=False, encoding="utf-8-sig")
    best.to_csv(out / f"pilot_{a.vehicles}_best.csv", index=False, encoding="utf-8-sig")
    (out / "meta.json").write_text(json.dumps({"vehicles": a.vehicles, "n_rep": a.n_rep,
                                               "districts": a.districts or load_network(cfg).districts},
                                              ensure_ascii=False), encoding="utf-8")
    cols = ["転換率", "ポート数", "自転車", "キックボード", "駅初期台数", "充足率", "利用希望", "利用", "断られた/日", "年_充足率",
            "年間利用回数", "年間総費用"]
    print(best[cols].round(3).to_string(index=False))
    print("保存:", out)


if __name__ == "__main__":
    main()
