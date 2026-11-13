"""① ポート最適配置（2026-11 改訂）：目的地（南栗橋駅・イオンスタイル・公園・エニタイム）と対象車両を選び、
住宅からポートまでの徒歩・短くなる時間の条件を満たす住宅の利用が最も多くなるポート位置と、目標充足率を満たす最小台数を求める。
固定条件（config の opt_time）：全街区230戸・夜間1回の再配置・ポート上限5台（駅は必要容量を算出）・手続き30秒・
9〜11月の3か月・経路別の実測速度・目的地ごとに1日1往復・用途ごとの使える車種。"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import streamlit as st
from streamlit_folium import st_folium

from src import opt_multi as om
from src import viz
from src.engine import TYPES, fleet_label
from src.ui import header_warnings, hourly_fig, load_or_stop

BASE_CFG, SURVEY, NET, FLAGS, DATA_KEY = load_or_stop("opt_time", all_districts=True)
cfg = BASE_CFG
OC = cfg["opt_time"]
ss = st.session_state

with st.sidebar:
    st.header("目的地")
    dests = [d for d in om.DEST_ALL if st.checkbox(om.DEST_LABEL[d], True, key=f"ot2_d_{d}")]
    st.header("対象車両")
    vehicles = [t for t in TYPES if st.checkbox(om.TYPE_LABEL[t], True, key=f"ot2_v_{t}")]
    st.header("目標充足率")
    target = st.slider("目標充足率", 50, 99, round(cfg["service"]["target_fulfillment"] * 100), 1, format="%d%%",
                       key="ot2_target", label_visibility="collapsed") / 100
    st.header("ポート数上限")
    k_max = st.slider("ポート数上限", 1, cfg["simulation"]["max_ports"], 3, key="ot2_k", label_visibility="collapsed",
                      help="住宅側ポートの数の上限。条件を満たす住宅が使わないポートは置きません。")
    st.header("車両上限")
    max_by = []
    for t in TYPES:
        v = st.number_input(f"{om.TYPE_LABEL[t]}（台）", min_value=0, max_value=300, value=None, step=1,
                            key=f"ot2_max_{t}", placeholder="上限なし",
                            disabled=t not in vehicles,
                            help=f"{om.TYPE_LABEL[t]}の上限台数。空欄なら上限なし。")
        max_by.append(None if (v is None or t not in vehicles) else int(v))
    st.caption("空欄＝上限なし。上限の範囲で目標充足率を満たす最小台数を探します。")
    st.header("需要発生条件")
    A = st.slider("① 住宅からポートまでの徒歩（分以内）", 0.5, 10.0, 3.0, 0.5, key="ot2_a", format="%.1f分",
                  help="出発地点（住宅）から住宅側ポートまで、この時間以内で歩ける住宅だけが利用します。")
    X = st.slider("② 短くなる時間（分以上）", 0.0, 10.0, 4.0, 0.5, key="ot2_x", format="%.1f分",
                  help="住宅から目的地までの合計移動時間（徒歩＋手続き＋走行＋ポートから施設までの徒歩）が、"
                       "徒歩だけよりこの時間以上短くなる住宅は、その目的地へ1日1往復します。")
    run = st.button("RUN", type="primary", use_container_width=True)
    with st.expander("固定の条件"):
        st.markdown(
            f"- 対象街区：既存街区＋新規街区（{len(NET.houses)}戸）\n"
            f"- 運用：夜間に1回、初期配置に戻す\n"
            f"- ポートの台数上限：{OC['port_cap']}台（駅ポートは上限なしで必要容量を算出）\n"
            f"- 手続き時間：{OC['procedure_sec']}秒\n"
            f"- 検証期間：{'・'.join(f'{m}月' for m in OC['months'])}の3か月\n"
            f"- 走行速度：走行経路ごとの実測値\n"
            f"- 対象住宅の利用：目的地ごとに1日{OC['trips']:g}往復\n"
            "- 利用用途と使える車種：\n"
            + "\n".join(f"  - {p['id']}. {p['label']}：{'・'.join(om.TYPE_LABEL[v] for v in p['vehicles'])}"
                        + ("（休日のみ）" if p["days"] == "休日" else "")
                        + (f"（{p['per_day']:g}{'回' if p.get('one_way') else '往復'}/日で固定）" if "per_day" in p else "")
                        for p in OC["purposes"]))

st.title("① ポート最適配置")
header_warnings(NET, FLAGS)

params = dict(dests=dests, vehicles=vehicles, target=target, k_max=int(k_max), A=float(A), X=float(X),
              max_by_type=max_by if any(x is not None for x in max_by) else None)
pkey = json.dumps([params, DATA_KEY, OC], ensure_ascii=False, default=str)


@st.cache_data(show_spinner="計算中…（3か月分のシミュレーションで最小台数を探します。数分かかることがあります）")
def compute(pkey_):
    return om.compute(NET, cfg, SURVEY, **json.loads(pkey_)[0])


if not dests:
    st.warning("目的地を1つ以上選んでください。")
    st.stop()
if not vehicles:
    st.warning("対象車両を1つ以上選んでください。")
    st.stop()
act = om.active_purposes(cfg, dests, vehicles)
if not act:
    st.warning("選んだ目的地・車両で使える利用用途がありません（例：公園は電動アシスト自転車だけ）。")
    st.stop()
if run:
    ss.ot2_key = pkey
if ss.get("ot2_key") != pkey:
    st.info("左で条件を選び、**RUN** を押してください。"
            + ("（条件が変わりました）" if ss.get("ot2_key") else ""))
    st.caption("この条件で使う利用用途：" + "、".join(f"{p['id']}. {p['label']}" for p in act))
    st.stop()
R = compute(pkey)
if R.get("error"):
    st.warning(R["error"])
    st.stop()

ms, sel, L = R.get("ms"), R.get("sel"), R["L"]
st.subheader("最適ポート配置")
n_hol = int((R["cal"]["曜日区分"] == "休日").sum())
st.caption(f"検証期間：{'・'.join(f'{m}月' for m in OC['months'])}（平日{len(R['cal']) - n_hol}日・休日{n_hol}日、{R['source']}）。"
           f"対象の住宅 {R['n_hh']}/{R['n_houses']}戸。")
ref = R.get("ref")
if sel is None:
    if ref and ref.get("上限"):
        lim = "・".join(f"{om.TYPE_LABEL[t]}{c}台" for t, c in zip(TYPES, R["max_by_type"]) if c is not None)
        st.error(f"車両上限（{lim}）の範囲では、目標充足率{target:.0%}を満たせません。"
                 f"下の結果は、上限の台数（上限の無い車種はポートを上限まで埋めた台数）の {ref['合計']}台での充足率です。")
    else:
        st.error(f"目標充足率{target:.0%}は達成できません。ポートの台数上限（{OC['port_cap']}台）までポートを埋めても、"
                 f"充足率は最大 {ms['充足率']:.1%} です。ポート数上限を増やすか、需要発生条件・目標充足率を見直してください。"
                 f"下の結果は、すべてのポートを上限まで埋めた {ref['合計']}台での参考値です。")
    sel = ref
c1, c2, c3, c4 = st.columns(4)
c1.metric("ポート数", f"{L.k}", border=True,
          help="住宅側ポートの数（上限以内）。このほかに目的地側のポート（駅・イオン・公園・エニタイム）を使います。")
c2.metric("駅ポート必要容量", f"{R['station_need']} 台", border=True,
          help="3か月の駅ポートの最大駐車台数（日ごとの最大の95%点）。駅ポートは上限なしで計算。")
c3.metric("充足率", f"{ms['充足率']:.0%}" if not ref or ref.get("上限") else f"最大 {ms['充足率']:.0%}", border=True,
          help=f"{ms['充足率']:.1%}（目標 {target:.0%}以上、3か月）")
c4.metric("平均短縮移動時間", f"{R['saving']:.1f} 分", border=True,
          help=f"徒歩だけ {ms['全員徒歩の平均(分/片道)']:.1f}分 → 平均 {ms['平均移動時間(分/片道)']:.1f}分（片道、利用できなかった人は徒歩）")
with st.container(border=True):
    v1, v2, v3, v4 = st.columns(4)
    v1.metric("合計車両台数", f"{sel['合計']} 台" + ("（上限）" if ref and ref.get("上限") else "（参考）" if ref else ""))
    v2.metric(om.TYPE_LABEL["ebike"], f"{sel['自転車']} 台")
    v3.metric(om.TYPE_LABEL["kickboard"], f"{sel['キックボード']} 台")
    v4.metric(om.TYPE_LABEL["trans"], f"{sel.get('4輪', 0)} 台")

init = R["init"]
left, right = st.columns([3, 2], gap="medium")
with left:
    st.subheader("ポート場所")
    cand = R["cands"].iloc[R["S"]].reset_index(drop=True)
    main = cand[["lat", "lon"]].assign(label=[f"P{j + 1}" for j in range(L.k)], 初期台数=init[:L.k].sum(1),
                                       必要区画=OC["port_cap"], 収納=OC["port_cap"],
                                       内訳=[fleet_label(init[j]) for j in range(L.k)])
    dest_tbl = pd.DataFrame([("駅", "station", int(init[L.k].sum()), 0, None),
                             ("モビリティステーション", "aeon", int(init[L.k + 1].sum()), 0, OC["port_cap"])],
                            columns=["ポート", "種別", "初期台数", "必要区画", "収納"])
    extra = {f"{om.DEST_LABEL[d]}ポート": pd.DataFrame([{"label": f"{om.DEST_LABEL[d]}ポート",
                                                        **OC["dest_ports"][d]}])
             for d in ("park", "anytime") if d in R["dests"]}
    m = viz.port_map(NET.houses, NET.candidates.iloc[0:0], main, R["assign"], dest_tbl, cfg, main_name="最適ポート",
                     square=True, overlays=extra or None)
    st_folium(m, height=540, use_container_width=True, returned_objects=[], key="ot2_map")
    st.caption("色つき＝対象の住宅（色は使うポート）、灰色＝条件を満たさず利用しない住宅。点線の丸＝公園・エニタイムのポート。"
               "ラベル：自＝電動アシスト自転車、キ＝電動キックボード、4輪＝4輪モビリティの初期台数")
with right:
    st.markdown("**時間帯別の利用希望と充足**")
    fig = hourly_fig(ms["_hourly"])
    if fig is not None:
        st.plotly_chart(fig, use_container_width=True)
    st.markdown("**ポートごと**")
    peaks = ms["ポート別最大駐車(95%点)"]
    names = [f"P{j + 1}" for j in range(L.k)] + ["駅西口ポート", "モビリティステーション"] + \
        [f"{om.DEST_LABEL[d]}ポート" for d in R["dests"] if d not in ("station", "aeon")]
    cover = list(np.bincount(R["assign"][R["assign"] >= 0], minlength=L.k)) + [None] * (len(names) - L.k)
    caps = [("上限なし" if not np.isfinite(c) else f"{int(c)}") for c in L.caps]
    st.dataframe(pd.DataFrame({"ポート": names, "自転車": init[:, 0], "キックボード": init[:, 1], "4輪": init[:, 2],
                               "収納": caps, "最大駐車(95%点)": peaks[:len(names)],
                               "カバー戸数": [("—" if c is None else str(int(c))) for c in cover]}),
                 hide_index=True, use_container_width=True)
    st.caption("台数は毎晩戻す初期配置。最大駐車が収納と同じなら、満車で返せないことがあります。")

with st.expander("詳細（利用用途ごと・費用）"):
    pur = ms.get("_用途別") or {}
    rows = []
    for p in R["purposes"]:
        r_ = pur.get(p["label"])
        rows.append({"用途": f"{p['id']}. {p['label']}", "使える車種": "・".join(om.TYPE_LABEL[v] for v in p["vehicles"]
                                                                              if v in vehicles),
                     "対象の住宅": R["n_cover"].get(p["label"], "—"),
                     "利用希望（回/日）": r_["利用希望/日"] if r_ else 0.0, "利用（回/日）": r_["利用/日"] if r_ else 0.0,
                     "充足率": r_["充足率"] if r_ else None})
    st.dataframe(pd.DataFrame(rows).style.format({"利用希望（回/日）": "{:.1f}", "利用（回/日）": "{:.1f}", "充足率": "{:.1%}"},
                                                 na_rep="—"), hide_index=True, use_container_width=True)
    st.caption("回/日は3か月の1日平均（片道ごと、休日のみの用途は平日を0として平均）。")
    d1, d2, d3 = st.columns(3)
    d1.metric("年間総費用（目安）", f"{R['cost']['年間総費用'] / 1e4:,.0f} 万円")
    d2.metric("返却失敗率", f"{ms['返却失敗率']:.1%}")
    d3.metric("代替乗車率", f"{ms['代替乗車率']:.1%}")
