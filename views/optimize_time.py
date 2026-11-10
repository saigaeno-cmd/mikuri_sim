"""① ポート最適化（時間短縮）：徒歩より「徒歩＋モビリティ」の方が X 分以上短くなる住宅は必ず利用するとして、
その住宅が最も多くなるポート位置と、目標を満たす最小台数を求める。"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import streamlit as st
from streamlit_folium import st_folium

from src import timesave as ts
from src import viz
from src.demand import DESTS
from src.engine import EB, KB
from src.plan import demand_hourly, fleet_for, make_ops
from src.study import annual_cost, min_fleet, month_eval
from src.ui import fresh_cfg, header_warnings, load_or_stop, month_specs, ops_sidebar
from src.weather_data import weather_path

BASE_CFG, SURVEY, NET, FLAGS, DATA_KEY = load_or_stop("opt_time")
cfg = fresh_cfg(BASE_CFG)
P = cfg["survey_xlsx"]["default_modes"]
_, _, SH_SURVEY = demand_hourly(SURVEY, cfg, 1.0, P, "平日")

with st.sidebar:
    st.header("条件")
    st.subheader("時間短縮の基準")
    X = st.slider("短くなる時間（分以上）", 0.0, 10.0, 4.0, 0.5, key="ot_x", format="%.1f分",
                  help="徒歩だけで行くより、徒歩＋モビリティ（手続き・目的地側の徒歩を含む）の方がこの時間以上短くなる住宅は、"
                       "必ず利用するとみなします。")
    trips = st.number_input("対象住宅の利用（1日あたり往復）", 0.5, 5.0, 1.0, 0.5, key="ot_trips")
    st_share = st.slider("行き先：駅の割合", 0, 100, int(round(SH_SURVEY[0] * 100)), 1, format="%d%%", key="ot_share",
                         help="初期値はアンケート（トリップ加重）。残りはモビリティステーション（イオン）。") / 100
    share = np.array([st_share, 1 - st_share])
    ops_spec = ops_sidebar(cfg, key_prefix="ot_", show_period=False)
    st.subheader("最適化")
    k = st.slider("住宅側ポート数", 1, cfg["simulation"]["max_ports"], 3, key="ot_k")
    target_f = st.slider("目標充足率", 50, 99, round(cfg["service"]["target_fulfillment"] * 100), 1, format="%d%%",
                         key="ot_target") / 100

st.title("① ポート最適化（時間短縮）")
st.caption("徒歩だけより「徒歩＋モビリティ」の方が指定の時間以上短くなる住宅は、必ず利用するとみなします（需要の量は転換率ではなく"
           "この条件で決まります）。ポートは、その住宅が最も多くなる位置を選び、台数は選んだ1か月（平日・休日・天気）で"
           "目標を満たす最小台数を探します。")
header_warnings(NET, FLAGS)

WO, MOB = ts.time_tensors(NET, cfg, ops_spec.ebike_share, procedure_sec=ops_spec.procedure_sec)


@st.cache_data(show_spinner=False)
def cover_table(k, share_t, es, proc, data_key):
    rows = []
    for x in np.arange(0, 10.5, 0.5):
        S = ts.solve_max_cover(WO, MOB, k, x, np.array(share_t))
        q, _ = ts.coverage(WO, MOB, S, x, np.array(share_t))
        rows.append({"短縮（分以上）": float(x), "対象の住宅（駅）": int(q[:, 0].sum()), "対象の住宅（イオン）": int(q[:, 1].sum()),
                     "1日の往復（目安）": float((q * np.array(share_t)).sum())})
    return pd.DataFrame(rows)


S = ts.solve_max_cover(WO, MOB, k, X, share)
q, saving = ts.coverage(WO, MOB, S, X, share)
L, assign, w = ts.layout_for(NET, cfg, S, q, share, WO, MOB, station_cap=ops_spec.station_cap)
n_trip = {key: float((q[:, d] * share[d]).sum() * trips) for d, key in enumerate(DESTS)}
n_hh = int(q.any(1).sum())

c1, c2, c3 = st.columns(3)
c1.metric("対象の住宅", f"{n_hh} / {len(NET.houses)} 戸", help=f"駅 {int(q[:, 0].sum())}戸・イオン {int(q[:, 1].sum())}戸")
c2.metric("1日の利用（往復）", f"{sum(n_trip.values()):.1f} 往復",
          help=f"駅 {n_trip['station']:.1f}・イオン {n_trip['aeon']:.1f}（目的地の割合で按分）")
sv_q = saving[q]
c3.metric("短縮時間（対象の平均）", f"{sv_q.mean():.1f} 分" if sv_q.size else "—")

if sum(n_trip.values()) <= 0:
    st.warning(f"{X:.1f}分以上短くなる住宅がありません。短くなる時間を小さくしてください（下の表で目安を確認できます）。")
    st.dataframe(cover_table(k, tuple(share), ops_spec.ebike_share, ops_spec.procedure_sec, DATA_KEY),
                 hide_index=True, use_container_width=True)
    st.stop()

# ---------- 需要と台数 ----------
oh_wd, rh_wd = ts.demand_for(SURVEY, cfg, {k_: v for k_, v in n_trip.items()}, "平日")
oh_hd, rh_hd = ts.demand_for(SURVEY, cfg, {k_: v for k_, v in n_trip.items()}, "休日")
DEM = {"平日": (oh_wd, rh_wd), "休日": (oh_hd, rh_hd)}
OPS_KW = dict(pattern=ops_spec.pattern, substitution=ops_spec.substitution, ebike_share=ops_spec.ebike_share,
              procedure_sec=ops_spec.procedure_sec)
CAL, SPECS, SOURCE = month_specs(cfg, ops_spec.year, ops_spec.month, DEM, OPS_KW)
N_REP = len(SPECS) * ops_spec.n_rep
_wp = weather_path(cfg)
WKEY = f"{_wp.stat().st_mtime if _wp.exists() else 0}"
targets = tuple(sorted({round(1 - target_f, 4), 0.05, 0.01}, reverse=True))
key = json.dumps([X, trips, list(share), k, S, ops_spec.key(), DATA_KEY, WKEY, targets], default=str)


@st.cache_data(show_spinner="最小台数を探索中…（対象の住宅が多いと数分かかります）")
def search(key_):
    ops = make_ops(cfg, weather="晴れ", **OPS_KW)
    res = min_fleet(L, ops, oh_wd, rh_wd, targets=targets, n_rep=N_REP, seed=0, day_specs=SPECS)
    res.pop("_cache")
    return res


@st.cache_data(show_spinner="選んだ案を1か月で評価中…")
def evaluate(key_, nb, nk):
    ops = make_ops(cfg, weather="晴れ", **OPS_KW)
    F = fleet_for(L, nb, nk, oh_wd, rh_wd, ops)
    ms, daily = month_eval(L, F, cfg, DEM, month=ops_spec.month, seed=1, cal=CAL, **OPS_KW)
    return F.init, ms, daily, annual_cost(cfg, F, L.k, daily, ops_spec.station_cap)


res = search(key)
st.subheader("目標ごとの最小台数")
rows = [{"目標": f"充足率{1 - t:.0%}以上（失敗率{t:.0%}以下）", "自転車": r["自転車"] if r else None,
         "キックボード": r["キックボード"] if r else None, "合計": r["合計"] if r else None,
         "充足率（1か月）": 1 - r["失敗率"] if r else None, "_t": t} for t, r in res.items() if not str(t).startswith("_")]
tbl = pd.DataFrame(rows).drop(columns="_t")
for c_ in ("自転車", "キックボード", "合計"):
    tbl[c_] = tbl[c_].map(lambda v: "達成不可" if v is None or pd.isna(v) else f"{int(v)}")
tbl["充足率（1か月）"] = tbl["充足率（1か月）"].map(lambda v: "—" if v is None or pd.isna(v) else f"{v:.1%}")
st.dataframe(tbl, hide_index=True, use_container_width=True)
st.caption(f"探索の基準：{ops_spec.year}年{ops_spec.month}月の1か月（{SOURCE}）× {ops_spec.n_rep}回、"
           f"{'代替乗車あり' if ops_spec.substitution else '代替乗車なし'}・運用{ops_spec.pattern}・駅ポート{ops_spec.station_cap}台。")
# 駅ポートに同時に置かれる台数の目安（朝に駅へ着く累計 − 夕方に駅から出る累計 の最大）
need_st = float(np.max(np.cumsum(oh_wd[:, 0] - rh_wd[:, 0])))
if need_st > ops_spec.station_cap:
    st.warning(f"朝に駅へ着く車両が、最大で約{need_st:.0f}台同時に駅ポートに置かれます（容量{ops_spec.station_cap}台）。"
               "満車で返せない車両は夜まで使えないため、台数を増やしても充足率が上がりにくくなります。"
               f"サイドバーの「駅ポートの容量」を{int(np.ceil(need_st))}台程度まで増やすか、短くなる時間を大きくして比べてください。")
ok = [r for r in rows if r["合計"] is not None]
if not ok:
    st.error("どの目標も達成できません（探索した台数の範囲内）。")
    st.dataframe(cover_table(k, tuple(share), ops_spec.ebike_share, ops_spec.procedure_sec, DATA_KEY),
                 hide_index=True, use_container_width=True, height=260)
    st.stop()
choice = st.radio("地図に出す案", [r["目標"] for r in ok], horizontal=True, key="ot_choice")
sel = next(r for r in ok if r["目標"] == choice)
init, ms, daily, cost = evaluate(key, sel["自転車"], sel["キックボード"])

c1, c2, c3, c4 = st.columns(4)
c1.metric("台数", f"自{sel['自転車']}＋キ{sel['キックボード']}", help=f"合計 {sel['合計']} 台")
c2.metric("充足率（1か月）", f"{ms['充足率']:.1%}")
c3.metric("平均移動時間", f"{ms['平均移動時間(分/片道)']:.1f} 分/片道",
          delta=f"{ms['平均移動時間(分/片道)'] - ms['全員徒歩の平均(分/片道)']:+.1f} 分（全員徒歩比）", delta_color="inverse")
c4.metric("年間総費用", f"{cost['年間総費用'] / 1e4:,.0f} 万円")

# ---------- 地図 ----------
st.markdown("**ポートの位置・初期台数（色つき＝対象の住宅、灰色＝短縮が足りず利用しない住宅）**")
cand = NET.candidates.loc[S, ["lat", "lon"]].reset_index(drop=True)
main = cand.assign(label=[f"P{j + 1}" for j in range(L.k)], 初期台数=init[:L.k].sum(1), 必要区画=init[:L.k].sum(1),
                   内訳=[f"自{init[j, EB]}・キ{init[j, KB]}" for j in range(L.k)])
dest_tbl = pd.DataFrame([("駅", "station", int(init[L.k].sum()), 0, ops_spec.station_cap),
                         ("モビリティステーション", "aeon", 0, 0, None)], columns=["ポート", "種別", "初期台数", "必要区画", "収納"])
m = viz.port_map(NET.houses, NET.candidates, main, assign, dest_tbl, cfg, main_name="時間短縮の最適ポート")
st_folium(m, height=540, use_container_width=True, returned_objects=[], key="ot_map")

st.subheader("短くなる時間ごとの対象住宅（住宅側ポート数を固定して位置を選び直した場合）")
ct = cover_table(k, tuple(share), ops_spec.ebike_share, ops_spec.procedure_sec, DATA_KEY)
st.dataframe(ct.style.format({"短縮（分以上）": "{:.1f}", "1日の往復（目安）": "{:.1f}"}), hide_index=True,
             use_container_width=True, height=260)
st.caption(f"徒歩だけの平均：駅 {WO[:, 0].mean():.1f}分・イオン {WO[:, 1].mean():.1f}分。"
           f"モビリティの時間には、手続き {ops_spec.procedure_sec:.0f}秒と、目的地ポートから施設までの徒歩を含みます。")
