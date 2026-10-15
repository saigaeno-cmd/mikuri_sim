"""全条件 × 10回のシミュレーション（バッチ）。結果は data/output/study/ に保存。

条件：需要シナリオ × 代替乗車（なし(i)/あり(ii)）× リバランス運用（a/b/c）× 駅ポート容量（18/25/32台）
各回（seed）で：
  1) 平日・晴れの1日を60回シミュレーションし、貸出失敗率 20%（充足率80%）・5%・1% 以下を満たす
     最小の「自転車＋キックボード」台数を探索
  2) 見つかった各案で1か月（平日・休日、気象は平年の出現日数から抽選）をシミュレーションし、年間費用を算出
使い方: python scripts/run_study.py [--reps 10] [--procs 4] [--districts 既存街区 新規街区] [--quick]
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.loaders import load_config, load_network, load_survey, resolve, survey_path  # noqa: E402
from src.plan import demand_hourly, fleet_for, make_ops, optimal_layout  # noqa: E402
from src.study import annual_cost, min_fleet, month_eval  # noqa: E402

G = {}


def _init(districts=None):
    cfg = load_config()
    net = load_network(cfg).subset(districts)
    cfg["simulation"]["n_households"] = cfg["simulation"].get("n_households") or len(net.houses)
    G.update(cfg=cfg, net=net, survey=load_survey(survey_path(cfg)[0], cfg))


def run_one(args):
    scen, sub, pattern, cap, seed, targets, n_rep = args
    cfg, net, sv = G["cfg"], G["net"], G["survey"]
    P = cfg["survey_xlsx"]["default_modes"]
    conv = cfg["demand_scenarios"][scen]
    es = cfg["fleet_mix"]["ebike_share"]
    oh, rh, sh = demand_hourly(sv, cfg, conv, P, "平日")
    oh2, rh2, _ = demand_hourly(sv, cfg, conv, P, "休日")
    L, ports, assign = optimal_layout(net, cfg, sh, 3, es, station_cap=cap)
    ops = make_ops(cfg, pattern=pattern, substitution=sub)
    t0 = time.time()
    res = min_fleet(L, ops, oh, rh, targets=targets, n_rep=n_rep, seed=seed)
    rows = []
    for tgt in targets:
        r = res[tgt]
        base = {"需要シナリオ": scen, "転換率": conv, "代替乗車": "あり(ii)" if sub else "なし(i)", "運用": pattern,
                "駅ポート容量": cap, "回": seed, "目標失敗率": tgt, "探索評価数": len(res["_cache"])}
        if r is None:
            rows.append({**base, "達成": False})
            continue
        F = fleet_for(L, r["自転車"], r["キックボード"], oh, rh, ops, seed=seed)
        summ, daily = month_eval(L, F, cfg, {"平日": (oh, rh), "休日": (oh2, rh2)}, pattern=pattern,
                                 substitution=sub, seed=seed)
        cost = annual_cost(cfg, F, L.k, daily, cap)
        rows.append({**base, "達成": True, "自転車": r["自転車"], "キックボード": r["キックボード"], "合計台数": r["合計"],
                     "平日晴れ失敗率": r["失敗率"], "駅初期台数": int(F.init[L.k].sum()),
                     **{f"月_{k}": v for k, v in summ.items() if not isinstance(v, list)},
                     **{f"費用_{k}": v for k, v in cost.items()}, "所要秒": time.time() - t0})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--procs", type=int, default=4)
    ap.add_argument("--districts", nargs="*", default=None, help="対象街区（例：既存街区 新規街区）。省略で全街区")
    ap.add_argument("--n-rep", type=int, default=40, help="最小台数探索で1条件あたりの日数")
    ap.add_argument("--quick", action="store_true", help="動作確認用（条件を絞る）")
    a = ap.parse_args()
    cfg = load_config()
    scen = list(cfg["demand_scenarios"])
    caps = list(cfg["station_port"]["capacity_cases"].values())
    grid = list(itertools.product(scen, (False, True), ("a", "b", "c"), caps))
    if a.quick:
        grid = grid[:2]
    targets = (0.20, 0.05, 0.01)
    jobs = [(s, sub, p, c, seed, targets, a.n_rep) for (s, sub, p, c) in grid for seed in range(a.reps)]
    out = resolve(cfg["paths"].get("output_dir", "data/output")) / "study"
    out.mkdir(parents=True, exist_ok=True)
    print(f"{len(grid)} 条件 × {a.reps} 回 = {len(jobs)} 件")
    rows, t0 = [], time.time()
    with Pool(a.procs, initializer=_init, initargs=(a.districts,)) as pool:
        for i, r in enumerate(pool.imap_unordered(run_one, jobs), 1):
            rows.extend(r)
            if i % 20 == 0 or i == len(jobs):
                pd.DataFrame(rows).to_csv(out / "results.csv", index=False, encoding="utf-8-sig")
                print(f"  {i}/{len(jobs)} 件  経過 {time.time() - t0:.0f} 秒", flush=True)
    pd.DataFrame(rows).to_csv(out / "results.csv", index=False, encoding="utf-8-sig")
    (out / "meta.json").write_text(json.dumps({"reps": a.reps, "n_rep": a.n_rep, "targets": targets,
                                               "districts": a.districts or load_network(cfg).districts,
                                               "conditions": len(grid)}, ensure_ascii=False), encoding="utf-8")
    print("保存:", out / "results.csv")


if __name__ == "__main__":
    main()
