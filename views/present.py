"""発表スライド：修論発表などで画面をそのまま投影できる6画面（16:9）。PNG・PPTで保存もできる。"""
from __future__ import annotations

import pandas as pd
import streamlit as st

from src import present as pr
from src.loaders import ROOT
from src.pilot import pilot_best, pilot_grid
from src.ui import load_or_stop, month_sidebar
from src.weather_data import weather_path

BASE_CFG, SURVEY, NET, FLAGS, DATA_KEY = load_or_stop("present")
cfg = BASE_CFG
ss = st.session_state
ss.setdefault("pr_i", 0)
ss.setdefault("pr_present", False)

with st.sidebar:
    st.header("発表の条件")
    conv = st.select_slider("利用転換率", [round(c * 100) for c in cfg["simulation"]["conversion_options"]], 10,
                            format_func=lambda v: f"{v}%", key="pr_conv") / 100
    year, month, passes = month_sidebar(cfg, "pr_")
    cap = st.number_input("駅ポートの容量（台）", 1, 200, int(cfg["station_port"]["max_slots"]), key="pr_cap")
    sub = st.radio("希望車種が空いていないとき", ["(ii) ほかの車種に乗る", "(i) 利用しない"],
                   index=0 if cfg["fleet_mix"]["substitution"] else 1, key="pr_sub").startswith("(ii)")
    k = st.slider("住宅側ポート数", 1, cfg["simulation"]["max_ports"], 3, key="pr_k")
    target = st.slider("③の目標充足率", 50, 99, 80, 1, format="%d%%", key="pr_target") / 100
    n_pilot = st.number_input("⑤ 実証の台数", 1, 60, 10, key="pr_np")
    if st.button("発表モードにする（メニューを隠す）", type="primary", use_container_width=True):
        ss.pr_present = True
        st.rerun()
show = ss.pr_present

if show:
    st.markdown("""<style>
      [data-testid="stSidebar"], [data-testid="stHeader"], [data-testid="stToolbar"] {display:none !important;}
      .block-container {padding: 0.6rem 1rem 0 1rem !important; max-width: 100% !important;}
      [data-testid="stImage"] img {max-height: calc(100vh - 70px); width: auto !important; margin: 0 auto; display: block;}
    </style>""", unsafe_allow_html=True)

_wp = weather_path(cfg)
WKEY = _wp.stat().st_mtime if _wp.exists() else 0


@st.cache_data(show_spinner="スライドの計算中…（初回は20〜40秒）")
def calc(conv, year, month, passes, cap, sub, k, target, data_key, wkey):
    return pr.compute(NET, cfg, SURVEY, conversion=conv, year=year, month=month, station_cap=cap, substitution=sub,
                      k=k, target=target, passes=passes)


@st.cache_data(show_spinner="実証の計算中…")
def calc_pilot(n_total, sub, cap, data_key):
    rows = []
    for c in [x / 100 for x in range(5, 55, 5)]:
        g = pilot_grid(NET, cfg, SURVEY, c, n_total=n_total, n_rep=60, station_cap=cap, substitution=sub)
        rows.append(pilot_best(g))
    return pd.DataFrame(rows)


def sens_table():
    p = ROOT / "data" / "output" / "study" / "sensitivity.csv"
    return pd.read_csv(p) if p.exists() else None


C = calc(conv, int(year), int(month), int(passes), int(cap), sub, int(k), target, DATA_KEY, WKEY)


@st.cache_data(show_spinner=False)
def slide_png(i, conv, year, month, passes, cap, sub, k, target, n_pilot, data_key, wkey):
    if i == 0:
        f = pr.fig_area(C)
    elif i == 1:
        f = pr.fig_demand(C)
    elif i == 2:
        f = pr.fig_optimal(C)
    elif i == 3:
        f = pr.fig_curve(C)
    elif i == 4:
        pl = calc_pilot(int(n_pilot), sub, int(cap), data_key)
        b0 = pl.iloc[0]
        note = (f"条件：実証{int(n_pilot)}台（住宅側{int(b0['ポート数'])}か所・自転車{int(b0['自転車'])}・"
                f"キックボード{int(b0['キックボード'])}・4輪{int(b0.get('4輪', 0))}）／平日・晴れ／{'代替乗車あり' if sub else '代替乗車なし'}／"
                f"駅ポート{int(cap)}台")
        f = pr.fig_pilot(pl, int(n_pilot), note)
    else:
        f = pr.fig_sensitivity(sens_table())
    return pr.to_png(f)


args = (conv, int(year), int(month), int(passes), int(cap), sub, int(k), target, int(n_pilot), DATA_KEY, WKEY)
if ss.get("pr_args") != args:                 # 条件が変わったら作ったPPTは捨てる
    ss.pr_args, ss.pr_pptx = args, None
i = ss.pr_i
st.image(slide_png(i, *args), use_container_width=True)

# ---------- 操作（スライドの下） ----------
c1, c2, c3, c4, c5 = st.columns([1, 4, 1, 2, 2])
if c1.button("◀ 前", disabled=i == 0, use_container_width=True):
    ss.pr_i = i - 1
    st.rerun()
pick = c2.radio("画面", range(6), index=i, horizontal=True, label_visibility="collapsed",
                format_func=lambda j: f"{j + 1} {pr.TITLES[j]}")
if pick != i:
    ss.pr_i = pick
    st.rerun()
if c3.button("次 ▶", disabled=i == 5, use_container_width=True):
    ss.pr_i = i + 1
    st.rerun()
c4.download_button("この画面をPNGで保存", slide_png(i, *args), file_name=f"slide_{i + 1}_{pr.TITLES[i]}.png",
                   mime="image/png", use_container_width=True)
if c5.button("6画面をPPTにする", use_container_width=True):
    with st.spinner("6画面を作成中…"):
        ss.pr_pptx = pr.to_pptx([slide_png(j, *args) for j in range(6)])
if ss.get("pr_pptx"):
    c5.download_button("PPTを保存", ss.pr_pptx, file_name=f"発表スライド_転換率{round(conv * 100)}%_{year}年{month}月.pptx",
                       mime="application/vnd.openxmlformats-officedocument.presentationml.presentation",
                       use_container_width=True)
if show:
    if st.button("発表モードを終える（メニューを表示）"):
        ss.pr_present = False
        st.rerun()
else:
    st.caption("条件は左のサイドバーで変えられます。「発表モードにする」を押すと、メニューを隠してスライドだけを表示します（下のボタンで戻せます）。"
               "⑥ 感度分析は一括計算（scripts/run_all.py）の結果を使います。")
