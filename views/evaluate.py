"""② ポート配置評価（2026-11 改訂）：地図をダブルクリックしてポート候補地と容量を置き、RUN で
候補地の中から「ポート数上限」以内の組み合わせを選び（すべて容量まで埋めたときの充足率が最大のもの）、
目標充足率を満たす最小台数を求める。

固定条件（config の opt_time）：全街区230戸・夜間1回の再配置・手続き30秒・9〜11月の3か月・経路別の実測速度・
用途ごとの使える車種。需要：住宅すべてが目的地ごとに1日1往復 × 転換率（低位12%・高位24%）。時間帯は需要表の形。"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import streamlit as st
from streamlit_folium import st_folium

from src import opt_multi as om
from src import viz
from src.engine import TYPES, fleet_label
from src.portmap import portmap
from src.ui import header_warnings, hourly_fig, load_or_stop

BASE_CFG, SURVEY, NET, FLAGS, DATA_KEY = load_or_stop("eval", all_districts=True)
cfg = BASE_CFG
OC = cfg["opt_time"]
ss = st.session_state
ss.setdefault("ev3_cands", [])
ss.setdefault("ev3_ver", 0)
ss.setdefault("ev3_nonce", None)
SCEN = {k: v for k, v in cfg.get("demand_scenarios", {}).items() if k.startswith(("低位", "高位"))}

with st.sidebar:
    st.header("目的地")
    dests = [d for d in om.DEST_ALL if st.checkbox(om.DEST_LABEL[d], True, key=f"ev3_d_{d}")]
    st.header("対象車両")
    vehicles = [t for t in TYPES if st.checkbox(om.TYPE_LABEL[t], True, key=f"ev3_v_{t}")]
    st.header("目標充足率")
    target = st.slider("目標充足率", 50, 99, round(cfg["service"]["target_fulfillment"] * 100), 1, format="%d%%",
                       key="ev3_target", label_visibility="collapsed") / 100
    st.header("ポート数上限")
    k_max = st.slider("ポート数上限", 1, cfg["simulation"]["max_ports"], 3, key="ev3_k", label_visibility="collapsed",
                      help="候補地の中から、この数までの住宅側ポートを選びます。")
    st.header("需要設定")
    scen = st.selectbox("需要シナリオ", list(SCEN), key="ev3_scen", format_func=lambda n: f"{n}（転換率 {SCEN[n]:.0%}）")
    conv = float(SCEN[scen])
    st.header("目的地側ポートの容量")
    c1, c2 = st.columns(2)
    dcap = {"station": c1.number_input("南栗橋駅", 1, 300, int(cfg["station_port"]["max_slots"]), key="ev3_c_station"),
            "aeon": c2.number_input("イオンスタイル", 1, 100, int(OC["port_cap"]), key="ev3_c_aeon"),
            "park": c1.number_input("公園", 1, 100, int(OC["port_cap"]), key="ev3_c_park"),
            "anytime": c2.number_input("エニタイム", 1, 100, int(OC["port_cap"]), key="ev3_c_anytime")}
    st.caption("駅ポートは「必要容量」（上限なしで回したときの最大駐車）も表示します。")
    with st.expander("固定の条件"):
        st.markdown(
            f"- 対象街区：既存街区＋新規街区（{len(NET.houses)}戸）\n"
            f"- 運用：夜間に1回、初期配置に戻す\n"
            f"- 手続き時間：{OC['procedure_sec']}秒\n"
            f"- 検証期間：{'・'.join(f'{m}月' for m in OC['months'])}の3か月\n"
            f"- 走行速度：走行経路ごとの実測値\n"
            f"- 需要：住宅すべてが目的地ごとに1日1往復 × 転換率（時間帯は需要表の形）\n"
            "- 利用用途と使える車種：\n"
            + "\n".join(f"  - {p['id']}. {p['label']}：{'・'.join(om.TYPE_LABEL[v] for v in p['vehicles'])}"
                        + ("（休日のみ）" if p["days"] == "休日" else "")
                        + (f"（{p['per_day']:g}{'回' if p.get('one_way') else '往復'}/日で固定）" if "per_day" in p else "")
                        for p in OC["purposes"]))

st.title("② ポート配置評価")
header_warnings(NET, FLAGS)

# ---------- ポート候補地入力 ----------
st.subheader("ポート候補地入力")
dest_pts = []
for d, col in (("station", "#1f3a93"), ("aeon", "#e07a1f"), ("park", "#2e7d32"), ("anytime", "#7b1fa2")):
    p = cfg["destinations"][d].get("port") or OC["dest_ports"][d]
    dest_pts.append({"lat": p["lat"], "lon": p["lon"], "label": f"{om.DEST_LABEL[d]}ポート", "color": col})
h = NET.houses
center = [float(h["lat"].mean()) + 0.0008, float(h["lon"].mean()) + 0.0012]
val = portmap(houses=h[["lat", "lon"]].round(6).values.tolist(), dests=dest_pts, cands=ss.ev3_cands, ver=ss.ev3_ver,
              center=center, zoom=17, height=520, default_cap=int(OC["port_cap"]), key="ev3_portmap")
if val and val.get("ver") == ss.ev3_ver and val.get("nonce") != ss.ev3_nonce:
    ss.ev3_nonce = val["nonce"]
    ss.ev3_cands = val["cands"]
cands = pd.DataFrame(ss.ev3_cands, columns=["name", "lat", "lon", "cap"])
b0, b1, b2 = st.columns([4, 1, 1])
b0.caption(f"候補地 {len(cands)} か所" + ("：" + "、".join(f"{r.name}（容量{r.cap}）" for r in cands.itertuples())
                                          if len(cands) else "（地図をダブルクリックして置いてください）"))
if b1.button("CLEAR", use_container_width=True):
    ss.ev3_cands, ss.ev3_ver, ss.ev3_key = [], ss.ev3_ver + 1, None
    st.rerun()
run = b2.button("RUN", type="primary", use_container_width=True, disabled=not len(cands))
st.divider()

if not dests or not vehicles:
    st.warning("目的地と対象車両を1つ以上選んでください。")
    st.stop()
act = om.active_purposes(cfg, dests, vehicles)
if not act:
    st.warning("選んだ目的地・車両で使える利用用途がありません（例：公園は電動アシスト自転車だけ）。")
    st.stop()
params = dict(dests=dests, vehicles=vehicles, target=target, k_max=int(k_max), conversion=conv,
              dest_caps={k: float(v) for k, v in dcap.items()})
pkey = json.dumps([params, cands.to_dict("records"), DATA_KEY, OC], ensure_ascii=False, default=str)


@st.cache_data(show_spinner="候補地の組み合わせを比べ、最小台数を探しています…（数十秒〜数分）")
def compute(pkey_):
    p, cs = json.loads(pkey_)[:2]
    return om.evaluate_candidates(NET, cfg, SURVEY, pd.DataFrame(cs), **p)


if run:
    ss.ev3_key = pkey
st.subheader("ポート選定結果")
if ss.get("ev3_key") != pkey:
    st.info("候補地を置き、条件を選んで **RUN** を押してください。" + ("（条件か候補地が変わりました）" if ss.get("ev3_key") else ""))
    st.stop()
R = compute(pkey)
if R.get("error"):
    st.warning(R["error"])
    st.stop()

ms, sel, ref, L = R["ms"], R.get("sel"), R.get("ref"), R["L"]
n_hol = int((R["cal"]["曜日区分"] == "休日").sum())
st.caption(f"選んだポート：{'・'.join(R['chosen'])}（{R['method']}）。検証期間：{'・'.join(f'{m}月' for m in OC['months'])}"
           f"（平日{len(R['cal']) - n_hol}日・休日{n_hol}日、{R['source']}）。需要：{scen}（転換率{conv:.0%}）。")
if sel is None:
    st.error(f"目標充足率{target:.0%}は達成できません。選んだポートを容量まで埋めても、充足率は最大 {ms['充足率']:.1%} です。"
             f"候補地・容量・ポート数上限を見直すか、目標充足率を下げてください。下の結果は、容量まで埋めた {ref['合計']}台での参考値です。")
    sel = ref
c1, c2, c3, c4 = st.columns(4)
c1.metric("ポート数", f"{L.k}", border=True, help="選んだ住宅側ポートの数。このほかに目的地側のポートを使います。")
c2.metric("駅ポート必要容量", f"{R['station_need']} 台", border=True,
          help=f"上限なしで回したときの3か月の最大駐車台数（95%点）。計算に使った駅ポートの容量は{dcap['station']}台。")
c3.metric("充足率", f"{ms['充足率']:.0%}" if not ref else f"最大 {ms['充足率']:.0%}", border=True,
          help=f"{ms['充足率']:.1%}（目標 {target:.0%}以上、3か月）")
c4.metric("平均短縮移動時間", f"{R['saving']:.1f} 分", border=True,
          help=f"徒歩だけ {ms['全員徒歩の平均(分/片道)']:.1f}分 → 平均 {ms['平均移動時間(分/片道)']:.1f}分（片道、利用できなかった人は徒歩）")
with st.container(border=True):
    v1, v2, v3, v4 = st.columns(4)
    v1.metric("合計車両台数", f"{sel['合計']} 台" + ("（参考）" if ref else ""))
    v2.metric(om.TYPE_LABEL["ebike"], f"{sel['自転車']} 台")
    v3.metric(om.TYPE_LABEL["kickboard"], f"{sel['キックボード']} 台")
    v4.metric(om.TYPE_LABEL["trans"], f"{sel.get('4輪', 0)} 台")

init = R["init"]
cs = R["cands"].iloc[R["S"]].reset_index(drop=True)
left, right = st.columns([3, 2], gap="medium")
with left:
    st.subheader("選定ポート場所")
    main = cs[["lat", "lon"]].assign(label=list(cs["name"]), 初期台数=init[:L.k].sum(1), 必要区画=L.caps[:L.k].astype(int),
                                     収納=L.caps[:L.k].astype(int), 内訳=[fleet_label(init[j]) for j in range(L.k)])
    dest_tbl = pd.DataFrame([("駅", "station", int(init[L.k].sum()), 0, dcap["station"]),
                             ("モビリティステーション", "aeon", int(init[L.k + 1].sum()), 0, dcap["aeon"])],
                            columns=["ポート", "種別", "初期台数", "必要区画", "収納"])
    others = cands[~cands["name"].isin(R["chosen"])].rename(columns={"name": "label"})
    overlays = {f"{om.DEST_LABEL[d]}ポート": pd.DataFrame([{"label": f"{om.DEST_LABEL[d]}ポート（容量{dcap[d]}）",
                                                          **OC["dest_ports"][d]}])
                for d in ("park", "anytime") if d in R["dests"]}
    m = viz.port_map(NET.houses, NET.candidates.iloc[0:0], main, R["assign"], dest_tbl, cfg, main_name="選定ポート",
                     square=True, overlays=overlays or None)
    st_folium(m, height=500, use_container_width=True, returned_objects=[], key="ev3_resmap")
    st.caption("色つき＝住宅（色は使うポート）。点線の丸＝公園・エニタイムのポート。"
               "ラベル：自＝電動アシスト自転車、キ＝電動キックボード、4輪＝4輪モビリティの初期台数 / 収納"
               + (f"。選ばれなかった候補地：{'、'.join(others['label'])}" if len(others) else ""))
with right:
    st.markdown("**時間帯別の利用希望と充足**")
    fig = hourly_fig(ms["_hourly"])
    if fig is not None:
        st.plotly_chart(fig, use_container_width=True)
    st.markdown("**ポートごと**")
    peaks = ms["ポート別最大駐車(95%点)"]
    names = list(cs["name"]) + ["駅西口ポート", "モビリティステーション"] + \
        [f"{om.DEST_LABEL[d]}ポート" for d in R["dests"] if d not in ("station", "aeon")]
    cover = list(np.bincount(R["assign"][R["assign"] >= 0], minlength=L.k)) + [None] * (len(names) - L.k)
    st.dataframe(pd.DataFrame({"ポート": names, "自転車": init[:, 0], "キックボード": init[:, 1], "4輪": init[:, 2],
                               "収納": [("上限なし" if not np.isfinite(c) else f"{int(c)}") for c in L.caps],
                               "最大駐車(95%点)": peaks[:len(names)],
                               "カバー戸数": [("—" if c is None else str(int(c))) for c in cover]}),
                 hide_index=True, use_container_width=True)
    st.caption("台数は毎晩戻す初期配置。最大駐車が収納と同じなら、満車で返せないことがあります。")

with st.expander("詳細（利用用途ごと・組み合わせの比較・費用）"):
    pur = ms.get("_用途別") or {}
    rows = []
    for p in R["purposes"]:
        r_ = pur.get(p["label"])
        rows.append({"用途": f"{p['id']}. {p['label']}",
                     "使える車種": "・".join(om.TYPE_LABEL[v] for v in p["vehicles"] if v in vehicles),
                     "利用希望（回/日）": r_["利用希望/日"] if r_ else 0.0, "利用（回/日）": r_["利用/日"] if r_ else 0.0,
                     "充足率": r_["充足率"] if r_ else None})
    st.dataframe(pd.DataFrame(rows).style.format({"利用希望（回/日）": "{:.1f}", "利用（回/日）": "{:.1f}", "充足率": "{:.1%}"},
                                                 na_rep="—"), hide_index=True, use_container_width=True)
    st.markdown("**候補地の組み合わせ（容量まで埋めたときの充足率の順）**")
    st.dataframe(R["table"].head(20).style.format({"充足率（容量まで埋めたとき）": "{:.1%}", "平均移動時間(分/片道)": "{:.1f}"}),
                 hide_index=True, use_container_width=True)
    d1, d2, d3 = st.columns(3)
    d1.metric("年間総費用（目安）", f"{R['cost']['年間総費用'] / 1e4:,.0f} 万円")
    d2.metric("返却失敗率", f"{ms['返却失敗率']:.1%}")
    d3.metric("代替乗車率", f"{ms['代替乗車率']:.1%}")
