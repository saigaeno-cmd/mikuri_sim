"""各ページで共通の読込・サイドバー部品。"""
from __future__ import annotations

import copy
import json
from dataclasses import dataclass

import numpy as np
import streamlit as st

from .demand import base_hourly, build_demand, dest_shares
from .loaders import file_hash, houses_path, load_config, load_houses_all, load_network, load_survey, survey_path


@st.cache_resource
def load_all():
    """設定・アンケート・事前計算結果（道路網の行列）を読むだけ。"""
    cfg = load_config()
    houses_files, houses_sample = houses_path(cfg)
    survey_file, survey_sample = survey_path(cfg)
    survey = load_survey(survey_file, cfg)
    net = load_network(cfg)
    stale = net.meta.get("input_hash") != file_hash(*houses_files)
    load_houses_all(cfg)  # 列の検証
    flags = {"houses_sample": houses_sample, "survey_sample": survey_sample, "stale": stale}
    return cfg, survey, net, flags, file_hash(*houses_files, survey_file)


@st.cache_resource
def _subset(_net, key, districts: tuple):
    return _net.subset(districts)


def district_sidebar(net, page: str) -> tuple:
    """対象街区（複数あるときだけ表示）。選択は全ページ共通（ページごとのウィジェットに引き継ぐ）。"""
    names = net.districts
    if len(names) <= 1:
        return tuple(names)
    counts = net.houses["街区"].value_counts()
    prev = [n for n in st.session_state.get("_districts", names) if n in names] or names
    sel = st.sidebar.multiselect("対象街区", names, default=prev, key=f"districts_{page}",
                                 format_func=lambda n: f"{n}（{counts[n]}戸）",
                                 help="シミュレーションに含める街区。需要は対象街区の戸数で計算します。")
    if not sel:
        st.sidebar.warning("街区を1つ以上選んでください。全街区で計算します。")
        sel = names
    else:
        st.session_state._districts = list(sel)
    st.sidebar.caption(f"対象：{sum(counts[n] for n in sel)}戸")
    return tuple(n for n in names if n in sel)


def load_or_stop(page: str = "main"):
    """（設定, アンケート, 対象街区の Network, フラグ, データキー）。設定の n_households は対象街区の戸数にする。"""
    try:
        cfg, survey, net, flags, key = load_all()
    except (FileNotFoundError, ValueError) as e:
        st.error(str(e))
        st.stop()
    sel = district_sidebar(net, page)
    net = _subset(net, key, sel)
    cfg = copy.deepcopy(cfg)
    if not cfg["simulation"].get("n_households"):
        cfg["simulation"]["n_households"] = len(net.houses)
    return cfg, survey, net, flags, f"{key}|{','.join(sel)}"


def header_warnings(net, flags):
    if flags["survey_sample"]:
        st.warning("アンケート（data/raw/survey.xlsx）が無いため、**需要は架空のサンプルアンケート** で計算しています。"
                   "台数・需要の数値は参考値です。", icon="⚠️")
    if flags["houses_sample"]:
        st.warning("住宅データが無いため **架空の住宅配置** で表示しています。", icon="⚠️")
    if net.meta.get("source") == "synthetic":
        st.warning("道路網は **合成格子（動作確認用）** です。実解析は `python scripts/precompute_network.py` "
                   "で作り直してください。", icon="⚠️")
    if flags["stale"]:
        st.error("住宅データが事前計算時から変わっています。precompute_network.py を再実行してください。")
    if net.Wn is None:
        st.error("事前計算結果が古い形式です。`python scripts/precompute_network.py` を再実行してください。")
        st.stop()


@dataclass
class DemandSpec:
    """サイドバーで指定した需要条件。"""
    conversion: float
    purposes: list[str]
    station_share: float | None
    opts: dict                    # day, hourly

    def build(self, survey, cfg, multiplier: float = 1.0):
        return build_demand(survey, cfg, self.conversion * multiplier, self.purposes, self.station_share, **self.opts)

    def key(self) -> str:
        return json.dumps([self.conversion, self.purposes, self.station_share, self.opts], ensure_ascii=False)


def vehicle_radio(cfg, key="vehicle"):
    keys = list(cfg["vehicles"])
    return st.radio("車種", keys, format_func=lambda v: cfg["vehicles"][v]["label"], horizontal=True, key=key,
                    help="需要は同じ条件で、走行速度などの違いによる差を比較します。")


def demand_sidebar(cfg, survey, *, key_prefix: str = "") -> DemandSpec:
    """転換率・曜日・用途・時間帯別需要・目的地比率。st.sidebar の中で呼ぶ。"""
    sim = cfg["simulation"]
    scen = cfg.get("demand_scenarios", {})
    mode = st.radio("需要の指定", ["シナリオ", "転換率（10%刻み）"], horizontal=True, key=f"{key_prefix}dmode",
                    help="シナリオ：低位・高位（暫定値。都の資料で確定予定）と旧シナリオ（一律転換率10%・30%）")
    if mode == "シナリオ" and scen:
        name = st.selectbox("需要シナリオ", list(scen), key=f"{key_prefix}scen",
                            format_func=lambda n: f"{n}（転換率 {scen[n]:.0%}）")
        conv_pct = round(scen[name] * 100)
    else:
        opts = [round(c * 100) for c in sim.get("conversion_options", sim["conversion_rates"])]
        conv_pct = st.select_slider("利用転換率", options=opts, value=opts[min(2, len(opts) - 1)],
                                    format_func=lambda v: f"{v}%", key=f"{key_prefix}conv")
    by_mode = "day" in survey.columns  # 2023アンケート（交通手段別・曜日別）
    day = st.radio("曜日", ["平日", "休日"], horizontal=True, help="アンケートの月〜金／土日の平均",
                   key=f"{key_prefix}day") if by_mode else None
    all_purposes = sorted(survey.loc[survey["trips"] > 0, "purpose"].unique())
    default_p = [p for p in cfg.get("survey_xlsx", {}).get("default_modes", []) if p in all_purposes] or all_purposes
    purposes = st.multiselect(
        "利用用途（転換対象の交通手段）" if by_mode else "利用用途", all_purposes,
        default=default_p if by_mode else all_purposes, key=f"{key_prefix}purposes",
        help="アンケートには目的の設問が無いため、現在の交通手段で区分します。既定はアンケートで「利用」と印のある手段。"
        if by_mode else None)
    if not purposes:
        st.warning("用途を1つ以上選んでください。")
        st.stop()

    with st.expander("時間帯別の需要（編集可）", expanded=False):
        st.caption("街区全体・転換前のトリップ数/日。初期値はアンケート（÷回答23世帯×戸数）。"
                   "編集すると、その時間帯の合計をこの値に置き換えます（内訳の比率は維持）。")
        base = base_hourly(survey, cfg, purposes, day)
        edited = st.data_editor(
            base, key=f"{key_prefix}hourly_{day}_{'_'.join(purposes)}", hide_index=True, use_container_width=True,
            disabled=["時"], num_rows="fixed",
            column_config={"時": st.column_config.NumberColumn("時", format="%d時", width=48),
                           "往路": st.column_config.NumberColumn("往路", min_value=0.0, step=0.5, format="%.1f",
                                                                width=72, help="自宅→施設"),
                           "復路": st.column_config.NumberColumn("復路", min_value=0.0, step=0.5, format="%.1f",
                                                                width=72, help="施設→自宅")})
        edited_flag = not np.allclose(edited[["往路", "復路"]].to_numpy(dtype=float),
                                      base[["往路", "復路"]].to_numpy(dtype=float))
        st.caption(f"合計：往路 {edited['往路'].sum():.1f} ／ 復路 {edited['復路'].sum():.1f}"
                   + ("（**手入力値を使用中**）" if edited_flag else "（アンケート値）"))
    hourly = tuple((int(h), float(o), float(r)) for h, o, r in edited[["時", "往路", "復路"]].itertuples(index=False)) \
        if edited_flag else None
    opts = {"day": day, "hourly": hourly}

    st.markdown("**目的地の比率**")
    use_survey = st.checkbox("アンケートの比率を使う（用途別）", value=True, key=f"{key_prefix}use_share",
                             help="2023アンケートは目的地の設問が無いため、徒歩＋電車＝駅、徒歩＝イオン と仮定"
                             if by_mode else None)
    survey_share = float(dest_shares(build_demand(survey, cfg, 1.0, purposes, None, **opts))[0])
    if use_survey:
        station_share = None
        st.caption(f"駅 {survey_share:.0%} ／ イオン {1 - survey_share:.0%}（トリップ加重）")
    else:
        station_share = st.slider("駅の割合", 0, 100, int(round(survey_share * 10) * 10), step=10,
                                  format="%d%%", key=f"{key_prefix}share") / 100
        st.caption(f"駅 {station_share:.0%} ／ イオン {1 - station_share:.0%}")
    spec = DemandSpec(conv_pct / 100, list(purposes), station_share, opts)
    spec.survey_share = survey_share  # 表示用
    return spec


def fresh_cfg(base_cfg):
    return copy.deepcopy(base_cfg)


@dataclass
class OpsSpec:
    """運用・車両の条件（新モデル）。"""
    ebike_share: float
    substitution: bool
    pattern: str
    station_cap: int
    procedure_sec: float
    period: str           # "1日" / "1か月"
    weather: str          # 1日のときの気象
    month: int
    n_rep: int

    def key(self) -> str:
        return json.dumps(self.__dict__, ensure_ascii=False, sort_keys=True)


def ops_sidebar(cfg, *, key_prefix: str = "", show_period: bool = True) -> OpsSpec:
    """車両（混在比率・代替乗車）・運用（リバランス・駅ポート容量・手続き時間）・充電・期間。cfg の battery を上書きする。"""
    fm = cfg["fleet_mix"]
    st.subheader("車両")
    es = st.slider("電動アシスト自転車を希望する人の割合", 30, 90, round(fm["ebike_share"] * 100), 1, format="%d%%",
                   key=f"{key_prefix}es", help="別アンケート Q16 より 60.3%（残りは電動キックボード）") / 100
    sub = st.radio("希望車種が空いていないとき", ["(i) 利用しない", "(ii) もう一方の車種に乗る"],
                   index=1 if fm["substitution"] else 0, key=f"{key_prefix}sub").startswith("(ii)")
    st.subheader("運用")
    pats = cfg["rebalance"]["patterns"]
    pattern = st.radio("リバランス", list(pats), format_func=lambda p: f"{p}：{pats[p]}", key=f"{key_prefix}pat")
    cap = st.number_input("駅ポートの容量（台）", 1, 200, int(cfg["station_port"]["max_slots"]), key=f"{key_prefix}cap",
                          help="基本25台（区画①18＋区画②1列7）。区画①のみ18台、区画②も2列なら32台")
    proc = st.slider("手続き時間（秒/回）", 0, 180, int(cfg["procedure_sec"]), 10, key=f"{key_prefix}proc")
    with st.expander("電動キックボードの充電（仮定値）"):
        b = cfg["battery"]["kickboard"]
        b["range_km"] = st.number_input("航続距離 km", 5.0, 100.0, float(b["range_km"]), 5.0, key=f"{key_prefix}rng")
        b["swap_below"] = st.slider("交換・充電を始める残量", 0.05, 0.5, float(b["swap_below"]), 0.05,
                                    key=f"{key_prefix}swb")
        b["out_hours"] = st.number_input("充電・交換で使えない時間（時間）", 0.5, 24.0, float(b["out_hours"]), 0.5,
                                         key=f"{key_prefix}outh")
        b["nightly_charge"] = st.checkbox("夜間の再配置時に毎晩充電する", value=bool(b["nightly_charge"]),
                                          key=f"{key_prefix}nchg")
    period, weather, month = "1日", "晴れ", int(cfg["month"]["month"])
    if show_period:
        st.subheader("期間・気象")
        period = st.radio("シミュレーション期間", ["1日", "1か月"], horizontal=True, key=f"{key_prefix}period",
                          help="1か月：平日・休日（土日祝）を含め、気象は平年の出現日数から日ごとに抽選。毎晩初期配置に戻す")
        if period == "1日":
            weather = st.selectbox("気象条件", list(cfg["weather"]), key=f"{key_prefix}w1")
        else:
            month = st.selectbox("対象月（{}年）".format(cfg["month"]["year"]), list(range(1, 13)), index=month - 1,
                                 format_func=lambda m: f"{m}月", key=f"{key_prefix}month")
    n_rep = st.select_slider("シミュレーション回数（1日モードの日数）", [20, 40, 60, 100, 200], 60, key=f"{key_prefix}nrep")
    return OpsSpec(es, sub, pattern, int(cap), float(proc), period, weather, int(month), int(n_rep))
