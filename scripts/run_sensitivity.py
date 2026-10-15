"""感度分析：最小台数と年間費用のトルネード図、推奨案が変わる境目。結果は data/output/study/ に保存。

基準：需要＝低位と高位の中間、(i) 代替乗車なし、駅ポート容量25台、手続き60秒、自転車比率60.3%、歩行4.8km/h、
      費用は config の値。目標＝貸出失敗率5%以下（平日・晴れの1日で探索）。
推奨案：運用 a/b/c のうち、目標を満たす最小台数での年間総費用が最も小さいもの。
使い方: python scripts/run_sensitivity.py [--procs 4] [--districts 既存街区 新規街区]
"""
from __future__ import annotations

import argparse
import copy
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.loaders import load_config, load_network, load_survey, resolve, survey_path  # noqa: E402
from src.plan import demand_hourly, fleet_for, make_ops, optimal_layout  # noqa: E402
from src.study import annual_cost, min_fleet, month_eval  # noqa: E402

TARGET = 0.05
G = {}


def base_values(cfg):
    sc = cfg["demand_scenarios"]
    low, high = sc["低位（暫定）"], sc["高位（暫定）"]
    return {"需要（転換率）": (low + high) / 2, "自転車の比率": cfg["fleet_mix"]["ebike_share"],
            "駅ポート容量": cfg["station_port"]["max_slots"], "手続き時間（秒）": cfg["procedure_sec"],
            "歩行速度（km/h）": cfg["walk_speed_kmh"]}


def ranges(cfg):
    sc = cfg["demand_scenarios"]
    caps = cfg["station_port"]["capacity_cases"]
    return {  # 台数に効く項目：(低, 高, 境目探索の格子)
        "需要（転換率）": (sc["低位（暫定）"], sc["高位（暫定）"], np.linspace(sc["低位（暫定）"], sc["高位（暫定）"], 5)),
        "自転車の比率": (0.50, 0.70, [0.50, 0.55, 0.603, 0.65, 0.70]),
        "駅ポート容量": (caps["区画①のみ"], caps["区画②も2列"], list(caps.values())),
        "手続き時間（秒）": (30, 120, [30, 60, 90, 120]),
        "歩行速度（km/h）": (4.0, 5.5, [4.0, 4.5, 4.8, 5.0, 5.5]),
    }


COST_ONLY = {"車両単価": ("price_scale", 0.7, 1.3, [0.7, 0.85, 1.0, 1.15, 1.3]),
             "ポート初期費用": ("port_scale", 0.5, 2.0, [0.5, 0.75, 1.0, 1.5, 2.0]),
             "軽トラック費": ("truck_scale", 0.5, 2.0, [0.5, 0.75, 1.0, 1.5, 2.0])}


def _init(districts=None):
    cfg = load_config()
    net = load_network(cfg).subset(districts)
    cfg["simulation"]["n_households"] = cfg["simulation"].get("n_households") or len(net.houses)
    G.update(cfg=cfg, net=net, survey=load_survey(survey_path(cfg)[0], cfg))


def run_point(args):
    """1つの条件・運用で：最小台数（目標5%）→ 1か月 → 費用（倍率1）。"""
    param, value, pattern, n_rep = args
    cfg = copy.deepcopy(G["cfg"])
    net, sv = G["net"], G["survey"]
    v = base_values(cfg)
    v[param] = value
    P = cfg["survey_xlsx"]["default_modes"]
    oh, rh, sh = demand_hourly(sv, cfg, v["需要（転換率）"], P, "平日")
    oh2, rh2, _ = demand_hourly(sv, cfg, v["需要（転換率）"], P, "休日")
    cap = int(v["駅ポート容量"])
    L, _, _ = optimal_layout(net, cfg, sh, 3, v["自転車の比率"], walk_kmh=v["歩行速度（km/h）"], station_cap=cap)
    kw = dict(pattern=pattern, substitution=False, ebike_share=v["自転車の比率"], procedure_sec=v["手続き時間（秒）"],
              walk_kmh=v["歩行速度（km/h）"])
    ops = make_ops(cfg, **kw)
    r = min_fleet(L, ops, oh, rh, targets=(TARGET,), n_rep=n_rep, seed=0)[TARGET]
    row = {"項目": param, "値": value, "運用": pattern}
    if r is None:
        return {**row, "達成": False}, None
    F = fleet_for(L, r["自転車"], r["キックボード"], oh, rh, ops)
    summ, daily = month_eval(L, F, cfg, {"平日": (oh, rh), "休日": (oh2, rh2)}, seed=0, **kw)
    cost = annual_cost(cfg, F, L.k, daily, cap)
    return ({**row, "達成": True, "自転車": r["自転車"], "キックボード": r["キックボード"], "合計台数": r["合計"],
             "失敗率": r["失敗率"], "月_充足率": summ["充足率"], "年間総費用": cost["年間総費用"]},
            (F, L.k, daily, cap))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--procs", type=int, default=4)
    ap.add_argument("--districts", nargs="*", default=None, help="対象街区（例：既存街区 新規街区）。省略で全街区")
    ap.add_argument("--n-rep", type=int, default=40)
    a = ap.parse_args()
    cfg = load_config()
    base = base_values(cfg)
    rg = ranges(cfg)
    # 各項目の格子（基準値を含む）と両端。基準点＝需要格子の中央（他の項目は基準値）
    pts = sorted({(p, round(float(x), 4)) for p, (lo, hi, grid) in rg.items() for x in [lo, hi, *grid]})
    jobs = [(p, x, pat, a.n_rep) for (p, x) in pts for pat in ("a", "b", "c")]
    print(f"{len(jobs)} 件の探索")
    t0 = time.time()
    rows, keep = [], {}
    with Pool(a.procs, initializer=_init, initargs=(a.districts,)) as pool:
        for i, (row, extra) in enumerate(pool.imap_unordered(run_point, jobs), 1):
            rows.append(row)
            if extra is not None:
                keep[(row["項目"], row["値"], row["運用"])] = extra
            if i % 10 == 0:
                print(f"  {i}/{len(jobs)}  {time.time() - t0:.0f}秒", flush=True)
    df = pd.DataFrame(rows)
    out = resolve(cfg["paths"].get("output_dir", "data/output")) / "study"
    out.mkdir(parents=True, exist_ok=True)
    # 費用だけに効く項目：基準点の各運用の結果に倍率を掛けて再計算
    b_d = round(float(base["需要（転換率）"]), 4)
    cost_rows = []
    for name, (kw, lo, hi, grid) in COST_ONLY.items():
        for x in sorted({lo, hi, *grid}):
            for pat in ("a", "b", "c"):
                ex = keep.get(("需要（転換率）", b_d, pat))
                if ex is None:
                    continue
                F, k, daily, cap = ex
                c = annual_cost(cfg, F, k, daily, cap, **{kw: x})
                nb, nk = F.n
                cost_rows.append({"項目": name, "値": x, "運用": pat, "達成": True, "自転車": nb, "キックボード": nk,
                                  "合計台数": nb + nk, "年間総費用": c["年間総費用"]})
    df = pd.concat([df, pd.DataFrame(cost_rows)], ignore_index=True)
    df.to_csv(out / "sensitivity.csv", index=False, encoding="utf-8-sig")
    print("保存:", out / "sensitivity.csv", f"（{time.time() - t0:.0f}秒）")


if __name__ == "__main__":
    main()
