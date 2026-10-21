"""運用ルールの追加検証。結果は data/output/study/ に保存（報告資料で使う）。

1) rebalance_buffer_test.csv：リバランス b・c で駅に残す台数（期待値＋1台／ポアソン95%点）と最小台数
2) full_return_test.csv     ：満車で返せない車両の扱い（区画外→次の作業で回収／運営者が回収に出る）
3) discourage_test.csv      ：利用離れの強さ α（0／0.5／1.0）と1年の結果

いずれも失敗率5%・運用a（1は a/b/c）・基本の代替乗車、需要は低位・高位。
使い方: python scripts/run_checks.py [--procs 4] [--reps 3] [--districts ...]
"""
from __future__ import annotations

import argparse
import copy
import sys
from multiprocessing import Pool
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.loaders import load_config, load_network, load_survey, resolve, survey_path  # noqa: E402
from src.plan import demand_hourly, fleet_for, make_ops, optimal_layout  # noqa: E402
from src.study import annual_cost, min_fleet, year_eval  # noqa: E402

G = {}
SCENS = ("低位（暫定）", "高位（暫定）")


def _init(districts=None):
    cfg = load_config()
    net = load_network(cfg).subset(districts)
    cfg["simulation"]["n_households"] = cfg["simulation"].get("n_households") or len(net.houses)
    G.update(cfg=cfg, net=net, survey=load_survey(survey_path(cfg)[0], cfg))


def _setup(scen, cap):
    cfg, net, sv = G["cfg"], G["net"], G["survey"]
    P = cfg["survey_xlsx"]["default_modes"]
    conv = cfg["demand_scenarios"][scen]
    oh, rh, sh = demand_hourly(sv, cfg, conv, P, "平日")
    oh2, rh2, _ = demand_hourly(sv, cfg, conv, P, "休日")
    L, _, _ = optimal_layout(net, cfg, sh, 3, cfg["fleet_mix"]["ebike_share"], station_cap=cap)
    return cfg, L, (oh, rh), {"平日": (oh, rh), "休日": (oh2, rh2)}


def job(args):
    kind, scen, cap, opt, seed = args
    cfg, L, (oh, rh), dem = _setup(scen, cap)
    sub = cfg["fleet_mix"]["substitution"]
    row = {"検証": kind, "需要シナリオ": scen, "駅ポート容量": cap, "回": seed}
    if kind == "buffer":
        pat, q, label = opt
        ops = make_ops(cfg, pattern=pat, substitution=sub, station_buffer_q=q)
        r = min_fleet(L, ops, oh, rh, targets=(0.05,), n_rep=60, seed=seed)[0.05]
        return {**row, "運用": pat, "駅に残す台数の基準": label, "合計台数": r and r["合計"]}
    if kind == "full_return":
        cfg = copy.deepcopy(cfg)
        cfg["rebalance"]["full_return"] = opt
        ops = make_ops(cfg, pattern="a", substitution=sub)
        r = min_fleet(L, ops, oh, rh, targets=(0.05,), n_rep=60, seed=seed)[0.05]
        if r is None:
            return {**row, "満車時の扱い": opt, "合計台数": None}
        F = fleet_for(L, r["自転車"], r["キックボード"], oh, rh, ops, seed=seed)
        ys, daily, _ = year_eval(L, F, cfg, dem, pattern="a", substitution=sub, seed=seed)
        c = annual_cost(cfg, F, L.k, daily, cap)
        return {**row, "満車時の扱い": opt, "合計台数": r["合計"], "返却失敗率": ys["返却失敗率"],
                "作業回数/日": ys["作業回数/日"], "年間総費用": c["年間総費用"]}
    if kind == "discourage":
        ops = make_ops(cfg, pattern="a", substitution=sub)
        r = min_fleet(L, ops, oh, rh, targets=(0.05,), n_rep=60, seed=seed)[0.05]
        if r is None:
            return {**row, "α": opt, "合計台数": None}
        cfg = copy.deepcopy(cfg)
        cfg.setdefault("behavior", {})["discourage_alpha"] = opt
        F = fleet_for(L, r["自転車"], r["キックボード"], oh, rh, ops, seed=seed)
        ys, daily, _ = year_eval(L, F, cfg, dem, pattern="a", substitution=sub, seed=seed)
        c = annual_cost(cfg, F, L.k, daily, cap)
        return {**row, "α": opt, "合計台数": r["合計"], "利用希望/日": ys["利用希望"], "充足率": ys["充足率"],
                "年間利用回数": c["年間利用回数"], "1トリップあたり費用": c["1トリップあたり費用"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--procs", type=int, default=4)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--districts", nargs="*", default=None)
    a = ap.parse_args()
    cfg = load_config()
    caps = list(cfg["station_port"]["capacity_cases"].values())
    base = cfg["station_port"]["max_slots"]
    buf = [("a", None, "—"), ("b", None, "期待値＋1台（指定どおり）"), ("b", 0.95, "ポアソン95%点"),
           ("c", 0.95, "ポアソン95%点")]
    jobs = [("buffer", s, base, o, r) for s in SCENS for o in buf for r in range(a.reps)]
    jobs += [("full_return", s, c, o, r) for s in SCENS for c in caps for o in ("outside", "collect")
             for r in range(a.reps)]
    jobs += [("discourage", s, base, o, r) for s in SCENS for o in (0.0, 0.5, 1.0) for r in range(a.reps)]
    print(f"{len(jobs)} 件")
    with Pool(a.procs, initializer=_init, initargs=(a.districts,)) as pool:
        rows = pool.map(job, jobs)
    out = resolve(cfg["paths"].get("output_dir", "data/output")) / "study"
    df = pd.DataFrame(rows)
    keys = {"buffer": ["需要シナリオ", "運用", "駅に残す台数の基準"], "full_return": ["需要シナリオ", "駅ポート容量", "満車時の扱い"],
            "discourage": ["需要シナリオ", "α"]}
    names = {"buffer": "rebalance_buffer_test.csv", "full_return": "full_return_test.csv",
             "discourage": "discourage_test.csv"}
    for k, cols in keys.items():
        x = df[df["検証"] == k].dropna(axis=1, how="all")
        num = [c for c in x.columns if c not in cols + ["検証", "回"] and pd.api.types.is_numeric_dtype(x[c])]
        res = x.groupby(cols, sort=False)[num].median().reset_index()
        res.to_csv(out / names[k], index=False, encoding="utf-8-sig")
        print("保存:", out / names[k])


if __name__ == "__main__":
    main()
