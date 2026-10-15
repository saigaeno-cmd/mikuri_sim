"""① ポート最適化：需要・気象・運用条件から、移動時間最短のポート位置と、目標を満たす最小の車両台数（自転車＋キックボード）を求める。"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import streamlit as st
from streamlit_folium import st_folium

from src import viz
from src.engine import EB, KB, simulate_days
from src.export import save_ports
from src.loaders import ROOT
from src.plan import demand_hourly, fleet_for, make_ops, optimal_layout
from src.study import annual_cost, min_fleet, month_eval
from src.ui import demand_sidebar, fresh_cfg, header_warnings, load_or_stop, ops_sidebar

BASE_CFG, SURVEY, NET, FLAGS, DATA_KEY = load_or_stop()
cfg = fresh_cfg(BASE_CFG)

with st.sidebar:
    st.header("条件")
    st.subheader("需要")
    spec = demand_sidebar(cfg, SURVEY, key_prefix="opt_")
    ops_spec = ops_sidebar(cfg, key_prefix="opt_")
    st.subheader("最適化")
    k = st.slider("住宅側ポート数", 1, cfg["simulation"]["max_ports"], 3, key="opt_k")
    target_f = st.slider("目標充足率", 50, 99, round(cfg["service"]["target_fulfillment"] * 100), 1, format="%d%%",
                         key="opt_target") / 100

st.title("① ポート最適化")
st.caption("住宅側ポートの位置は、車種比率で加重した（徒歩＋走行＋目的地側徒歩）の時間が最小になる地点（p-median）。"
           "台数は「自転車×キックボード」の組み合わせから、貸出失敗率が目標以下になる最小の合計台数を探します。")
header_warnings(NET, FLAGS)

P, conv = spec.purposes, spec.conversion
day = spec.opts.get("day") or "平日"
oh, rh, sh = demand_hourly(SURVEY, cfg, conv, P, day, spec.opts.get("hourly"), spec.station_share)
oh_wd, rh_wd, _ = demand_hourly(SURVEY, cfg, conv, P, "平日", None, spec.station_share)
oh_hd, rh_hd, _ = demand_hourly(SURVEY, cfg, conv, P, "休日", None, spec.station_share)
search_weather = ops_spec.weather if ops_spec.period == "1日" else "晴れ"
targets = tuple(sorted({round(1 - target_f, 4), 0.05, 0.01}, reverse=True))


@st.cache_data(show_spinner="最小台数を探索中…（自転車×キックボードの組み合わせごとにシミュレーション）")
def run_search(cfg_json, spec_key, ops_key, k, targets, _key):
    L, ports, assign = optimal_layout(NET, cfg, sh, k, ops_spec.ebike_share, station_cap=ops_spec.station_cap)
    ops = make_ops(cfg, weather=search_weather, pattern=ops_spec.pattern, substitution=ops_spec.substitution,
                   ebike_share=ops_spec.ebike_share, procedure_sec=ops_spec.procedure_sec)
    res = min_fleet(L, ops, oh, rh, targets=targets, n_rep=ops_spec.n_rep, seed=0)
    res.pop("_cache")
    return L, ports, assign, ops, res


@st.cache_data(show_spinner="選んだ案を評価中…")
def run_eval(cfg_json, spec_key, ops_key, k, nb, nk, _key, _L, _ops):
    F = fleet_for(_L, nb, nk, oh, rh, _ops)
    day_res = simulate_days(_L, F, _ops, oh, rh, n_rep=ops_spec.n_rep, seed=1)
    m_sum, daily = month_eval(_L, F, cfg, {"平日": (oh_wd, rh_wd), "休日": (oh_hd, rh_hd)}, pattern=ops_spec.pattern,
                              substitution=ops_spec.substitution, ebike_share=ops_spec.ebike_share,
                              procedure_sec=ops_spec.procedure_sec, month=ops_spec.month, seed=1)
    cost = annual_cost(cfg, F, _L.k, daily, ops_spec.station_cap)
    return F, day_res, m_sum, daily, cost


if oh.sum() <= 0:
    st.warning("この条件では利用希望がありません。")
    st.stop()
cfg_json = json.dumps(cfg, ensure_ascii=False, sort_keys=True, default=str)
L, ports, assign, ops, res = run_search(cfg_json, spec.key(), ops_spec.key(), k, targets, DATA_KEY)

# ---------- 目標ごとの最小台数 ----------
st.subheader("目標ごとの最小台数")
basis = (f"{day}・{search_weather}の1日 × {ops_spec.n_rep}回" if ops_spec.period == "1日"
         else f"平日・晴れの1日 × {ops_spec.n_rep}回")
rows = []
for t in targets:
    r = res[t]
    rows.append({"目標": f"充足率{1 - t:.0%}以上（失敗率{t:.0%}以下）",
                 "自転車": r["自転車"] if r else None, "キックボード": r["キックボード"] if r else None,
                 "合計": r["合計"] if r else None, "貸出失敗率": r["失敗率"] if r else None, "_t": t})
tbl = pd.DataFrame(rows)
st.dataframe(tbl.drop(columns="_t").style.format({"貸出失敗率": "{:.1%}", "自転車": "{:.0f}", "キックボード": "{:.0f}",
                                                 "合計": "{:.0f}"}, na_rep="達成不可"),
             hide_index=True, use_container_width=True)
st.caption(f"探索の基準：{basis}（{'(ii) 代替乗車あり' if ops_spec.substitution else '(i) 代替乗車なし'}・運用{ops_spec.pattern}・"
           f"駅ポート容量{ops_spec.station_cap}台・手続き{ops_spec.procedure_sec:.0f}秒）。達成不可＝探索範囲の台数では目標に届かない。")
ok = [r for r in rows if r["合計"] is not None]
if not ok:
    st.error("どの目標も達成できません（駅ポート容量や需要を確認してください）。")
    st.stop()
choice = st.radio("詳しく見る案", [r["目標"] for r in ok], horizontal=True, key="opt_choice")
sel = next(r for r in ok if r["目標"] == choice)
F, day_res, m_sum, daily, cost = run_eval(cfg_json, spec.key(), ops_spec.key(), k, sel["自転車"], sel["キックボード"],
                                          DATA_KEY, L, ops)
R = day_res if ops_spec.period == "1日" else m_sum
period_txt = (f"1日（{day}・{ops_spec.weather}）× {ops_spec.n_rep}回の平均" if ops_spec.period == "1日"
              else f"{cfg['month']['year']}年{ops_spec.month}月の1か月（平日・休日・気象を含む）")

# ---------- 選んだ案 ----------
c1, c2, c3, c4 = st.columns(4)
c1.metric("台数", f"自{sel['自転車']}＋キ{sel['キックボード']}", help=f"電動アシスト自転車＋電動キックボード、合計 {sel['合計']} 台")
c2.metric("充足率", f"{R['充足率']:.1%}", help=f"貸出失敗率 {R['貸出失敗率']:.1%}（{period_txt}）")
c3.metric("平均移動時間", f"{R['平均移動時間(分/片道)']:.1f} 分/片道",
          delta=f"{R['平均移動時間(分/片道)'] - R['全員徒歩の平均(分/片道)']:+.1f} 分（全員徒歩比）", delta_color="inverse")
c4.metric("年間総費用", f"{cost['年間総費用'] / 1e4:,.0f} 万円", help=f"1トリップあたり {cost['1トリップあたり費用']:,.0f} 円")
st.caption(f"{period_txt}：返却失敗率 {R['返却失敗率']:.1%}・代替乗車率 {R['代替乗車率']:.1%}・"
           f"駅ポート最大在庫 {R['駅ポート最大在庫']}台／容量{ops_spec.station_cap}台・リバランス {R['作業回数/日']:.1f}回/日"
           f"（{R['移動台数/日']:.1f}台）・電池交換 {R['電池交換/日']:.2f}回/日・稼働率 {R['稼働率(回/台/日)']:.2f}回/台/日")

caps_house = np.maximum(F.init[:L.k].sum(1), np.array(m_sum["ポート別最大駐車(95%点)"][:L.k]))
main = NET.candidates.loc[list(ports), ["lat", "lon"]].reset_index(drop=True).assign(
    label=[f"P{j + 1}" for j in range(L.k)], 初期台数=F.init[:L.k].sum(1), 必要区画=caps_house, 収納=caps_house,
    内訳=[f"自{F.init[j, EB]}・キ{F.init[j, KB]}" for j in range(L.k)])
dest_tbl = pd.DataFrame([("駅", "station", int(F.init[L.k].sum()), 0, ops_spec.station_cap),
                         ("モビリティステーション", "aeon", 0, 0, None)],
                        columns=["ポート", "種別", "初期台数", "必要区画", "収納"])
st.markdown("**ポートの位置・初期台数（自＝電動アシスト自転車、キ＝電動キックボード）・収納可能台数**")
m = viz.port_map(NET.houses, NET.candidates, main, assign, dest_tbl, cfg, main_name="推奨ポート")
st_folium(m, height=560, use_container_width=True, returned_objects=[], key="opt_map")
st.caption(f"駅西口ポートの初期台数：自転車{F.init[L.k, EB]}台・キックボード{F.init[L.k, KB]}台（毎晩この配置に戻す）。"
           "住宅側の収納可能台数＝初期台数と1か月の最大駐車台数（95%点）の大きい方。")

a, b = st.columns([3, 2])
with a:
    st.markdown("**年間費用の内訳**")
    keys = ["車両（償却）", "車両維持", "ポート（償却）", "用地", "人件費（リバランス）", "軽トラック"]
    st.dataframe(pd.DataFrame({"項目": keys + ["年間総費用"], "円/年": [cost[x] for x in keys + ["年間総費用"]]})
                 .style.format({"円/年": "{:,.0f}"}), hide_index=True, use_container_width=True)
    bm = cfg["cost"]["benchmarks"]
    st.caption(f"1台あたり {cost['1台あたり年間費用'] / 1e4:.1f} 万円/年（目安：中野区 {bm['中野区シェアサイクル区負担（2020年度）']}）・"
               f"運営費に占める再配置費用 {cost['運営費に占める再配置費用']:.0%}（目安：{bm['再配置費用の割合（国交省資料）']}）・"
               f"稼働率 {cost['稼働率(回/台/日)']:.2f} 回/台/日（目安：{bm['国内の稼働率']}）")
with b:
    st.markdown("**時間帯別の利用希望と充足**")
    hr = np.array(R["_hourly"])
    if hr.size:
        st.plotly_chart(viz.hourly_service(pd.DataFrame({"時": np.arange(24), "利用希望": hr[:, 0], "利用": hr[:, 1]})),
                        use_container_width=True)

if ops_spec.period == "1か月":
    st.subheader(f"{cfg['month']['year']}年{ops_spec.month}月の日別結果")
    show = daily[["日付", "曜日区分", "気象", "利用希望", "利用", "充足率", "返却失敗/日", "代替乗車率", "駅ポート最大在庫",
                  "作業時間(分/日)"]]
    st.dataframe(show.style.format({"利用希望": "{:.0f}", "利用": "{:.0f}", "充足率": "{:.0%}", "代替乗車率": "{:.0%}",
                                    "返却失敗/日": "{:.0f}", "作業時間(分/日)": "{:.0f}"}),
                 hide_index=True, use_container_width=True, height=300)

# ---------- 運用パターンの比較 ----------
st.subheader("リバランス運用の比較（a/b/c）")
if st.button("運用 a・b・c を同じ目標で比較する（1〜2分）"):
    t = sel["_t"]
    comp = []
    for pat in ("a", "b", "c"):
        ops_p = make_ops(cfg, weather=search_weather, pattern=pat, substitution=ops_spec.substitution,
                         ebike_share=ops_spec.ebike_share, procedure_sec=ops_spec.procedure_sec)
        r = min_fleet(L, ops_p, oh, rh, targets=(t,), n_rep=ops_spec.n_rep, seed=0)[t]
        if r is None:
            comp.append({"運用": pat, "台数": "達成不可"})
            continue
        Fp = fleet_for(L, r["自転車"], r["キックボード"], oh, rh, ops_p)
        ms, dl = month_eval(L, Fp, cfg, {"平日": (oh_wd, rh_wd), "休日": (oh_hd, rh_hd)}, pattern=pat,
                            substitution=ops_spec.substitution, ebike_share=ops_spec.ebike_share,
                            procedure_sec=ops_spec.procedure_sec, month=ops_spec.month, seed=1)
        cp = annual_cost(cfg, Fp, L.k, dl, ops_spec.station_cap)
        comp.append({"運用": pat, "台数": f"{r['自転車']}+{r['キックボード']}={r['合計']}", "駅ポート最大在庫": ms["駅ポート最大在庫"],
                     "作業回数/日": ms["作業回数/日"], "移動台数/日": ms["移動台数/日"], "年間総費用(万円)": cp["年間総費用"] / 1e4,
                     "充足率(1か月)": ms["充足率"]})
    st.dataframe(pd.DataFrame(comp).style.format({"作業回数/日": "{:.1f}", "移動台数/日": "{:.1f}", "年間総費用(万円)": "{:,.0f}",
                                                  "充足率(1か月)": "{:.1%}"}, na_rep="-"), hide_index=True, use_container_width=True)

if st.button("この案を data/output に保存"):
    out = main.drop(columns=["必要区画"]).assign(自転車=F.init[:L.k, EB], キックボード=F.init[:L.k, KB])
    st_port = cfg["destinations"]["station"]["port"]
    out = pd.concat([out, pd.DataFrame([{"label": "駅西口ポート", "lat": st_port["lat"], "lon": st_port["lon"],
                                         "初期台数": int(F.init[L.k].sum()), "自転車": int(F.init[L.k, EB]),
                                         "キックボード": int(F.init[L.k, KB]), "収納": ops_spec.station_cap}])],
                    ignore_index=True)
    csv, gj = save_ports(out.rename(columns={"label": "name", "lat": "緯度", "lon": "経度"}),
                         f"optimized_mix_{ops_spec.pattern}_{'ii' if ops_spec.substitution else 'i'}_conv{round(conv * 100)}", cfg)
    st.success(f"保存しました：`{csv.relative_to(ROOT)}` と `{gj.relative_to(ROOT)}`")
