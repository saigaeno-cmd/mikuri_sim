"""実証運用の利用記録を取り込み、シミュレーションの前提値と比べる（需要・車種比率・速度・失敗率）。

利用記録（CSV、1行＝1回の利用）。ひな形：data/templates/usage_log_template.csv
    貸出日時, 返却日時, 貸出ポート, 返却ポート, 車種, 利用者ID（任意）, 走行距離km（任意）, 代替乗車（任意：1/0）
    ポート名：住宅側ポートは自由な名前、駅は「駅西口ポート」、イオンは「モビリティステーション」（config の port_label）
    車種：電動アシスト自転車／電動キックボード（ebike / kickboard でも可）
断られた記録（任意、CSV）：日時, ポート, 希望車種   … アプリで「空きなし」になった操作のログ

出力（data/output/usage/）
    hourly_demand.csv   ：平日・休日 × 時間帯 × 方向（往路=住宅→駅・イオン／復路）の1日あたり利用回数
    calibration.csv     ：観測値とシミュレーション前提の比較（推定転換率・車種比率・平均速度・失敗率）
使い方: python scripts/import_usage_log.py 利用記録.csv [--refusals 断られた記録.csv] [--conversion 0.12]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.demand import build_demand  # noqa: E402
from src.loaders import load_config, load_network, load_survey, resolve, survey_path  # noqa: E402

TYPE = {"電動アシスト自転車": "ebike", "自転車": "ebike", "ebike": "ebike",
        "電動キックボード": "kickboard", "キックボード": "kickboard", "kickboard": "kickboard"}


def _read(path):
    for enc in ("utf-8-sig", "cp932"):
        try:
            return pd.read_csv(path, encoding=enc)
        except UnicodeDecodeError:
            continue
    raise ValueError(f"{path} の文字コードを判別できません")


def analyze(log: pd.DataFrame, cfg: dict, holidays, refusals: pd.DataFrame | None = None) -> dict:
    d = log.copy()
    d["貸出日時"] = pd.to_datetime(d["貸出日時"])
    d["返却日時"] = pd.to_datetime(d["返却日時"])
    d["車種"] = d["車種"].map(lambda v: TYPE.get(str(v).strip(), "unknown"))
    hol = {pd.Timestamp(h).date() for h in holidays}
    d["日"] = d["貸出日時"].dt.date
    d["曜日区分"] = d["貸出日時"].map(lambda t: "休日" if t.weekday() >= 5 or t.date() in hol else "平日")
    d["時"] = d["貸出日時"].dt.hour
    dest_ports = {cfg["destinations"][k]["port_label"]: k for k in cfg["destinations"]}
    d["方向"] = np.where(d["返却ポート"].isin(dest_ports), "往路", np.where(d["貸出ポート"].isin(dest_ports), "復路", "その他"))
    d["目的地"] = [dest_ports.get(b, dest_ports.get(a, "その他")) for a, b in zip(d["貸出ポート"], d["返却ポート"])]
    d["分"] = (d["返却日時"] - d["貸出日時"]).dt.total_seconds() / 60
    n_days = d.groupby("曜日区分")["日"].nunique()
    hourly = d.groupby(["曜日区分", "方向", "目的地", "時"]).size().rename("回").reset_index()
    hourly["回/日"] = hourly["回"] / hourly["曜日区分"].map(n_days)
    out = {"hourly": hourly, "days": n_days.to_dict(), "trips_per_day": (d.groupby("曜日区分").size() / n_days).to_dict()}
    known = d[d["車種"] != "unknown"]
    out["ebike_share_obs"] = float((known["車種"] == "ebike").mean()) if len(known) else np.nan
    if "走行距離km" in d and d["走行距離km"].notna().any():
        dd = d[(d["分"] > 0) & d["走行距離km"].notna()]
        out["speed_kmh"] = dd.groupby("車種").apply(lambda g: g["走行距離km"].sum() / (g["分"].sum() / 60)).to_dict()
    if "代替乗車" in d and d["代替乗車"].notna().any():
        out["substitution_rate"] = float(d["代替乗車"].fillna(0).astype(float).mean())
    if refusals is not None and len(refusals):
        r = refusals.copy()
        r["日時"] = pd.to_datetime(r["日時"])
        out["fail_rate"] = len(r) / (len(r) + len(d))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("log")
    ap.add_argument("--refusals", default=None)
    ap.add_argument("--conversion", type=float, default=None, help="比較する転換率（省略時は低位シナリオ）")
    a = ap.parse_args()
    cfg = load_config()
    net = load_network(cfg)
    cfg["simulation"]["n_households"] = cfg["simulation"].get("n_households") or len(net.houses)
    year = cfg["month"]["year"]
    res = analyze(_read(a.log), cfg, cfg["month"].get(f"holidays_{year}", []),
                  _read(a.refusals) if a.refusals else None)
    out = resolve(cfg["paths"].get("output_dir", "data/output")) / "usage"
    out.mkdir(parents=True, exist_ok=True)
    res["hourly"].to_csv(out / "hourly_demand.csv", index=False, encoding="utf-8-sig")

    sv = load_survey(survey_path(cfg)[0], cfg)
    P = cfg["survey_xlsx"]["default_modes"]
    conv = a.conversion or list(cfg["demand_scenarios"].values())[0]
    rows = []
    for day in ("平日", "休日"):
        if day not in res["trips_per_day"]:
            continue
        base = build_demand(sv, cfg, 1.0, P, None, day=day)["trips"].sum()        # 転換率100%のときの回/日
        obs = res["trips_per_day"][day]
        rows.append({"項目": f"利用回数/日（{day}）", "観測": round(obs, 1), "シミュレーション": round(base * conv, 1),
                     "備考": f"観測から推定した転換率 {obs / base:.1%}（シミュレーションは {conv:.0%}）" if base else ""})
    rows.append({"項目": "電動アシスト自転車の割合", "観測": round(res["ebike_share_obs"], 3),
                 "シミュレーション": cfg["fleet_mix"]["ebike_share"], "備考": "実際に乗った車種（代替乗車を含む）"})
    for t, v in (res.get("speed_kmh") or {}).items():
        rows.append({"項目": f"平均速度 km/h（{t}）", "観測": round(v, 1),
                     "シミュレーション": cfg["vehicles"].get(t, {}).get("speed_kmh"), "備考": "手続き時間を含む"})
    if "substitution_rate" in res:
        rows.append({"項目": "代替乗車率", "観測": round(res["substitution_rate"], 3), "シミュレーション": "",
                     "備考": "計算結果の代替乗車率と比べる"})
    if "fail_rate" in res:
        rows.append({"項目": "貸出失敗率", "観測": round(res["fail_rate"], 3), "シミュレーション": "",
                     "備考": "断られた操作 ÷（利用＋断られた操作）"})
    cal = pd.DataFrame(rows)
    cal.to_csv(out / "calibration.csv", index=False, encoding="utf-8-sig")
    print(f"観測日数：{res['days']}")
    print(cal.to_string(index=False))
    print("保存:", out / "hourly_demand.csv", "／", out / "calibration.csv")
    print("推定した転換率は、画面の「設定」→需要シナリオ（または config.yaml の demand_scenarios）に入れて"
          " python scripts/run_all.py を実行すると反映されます。")


if __name__ == "__main__":
    main()
