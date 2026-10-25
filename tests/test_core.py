import itertools

import numpy as np
import pandas as pd
import pytest

from src.demand import build_demand, hourly_matrix
from src.fleet import simulate_fleet
from src.loaders import ROOT, load_config
from src.location import solve_pmedian

CFG = load_config()


def survey():
    return pd.DataFrame({
        "respondent_id": [1, 2, 3, 4],
        "hour": [7, 7, 10, 15],
        "purpose": ["通勤", "通勤", "買物", "買物"],
        "destination": ["station", "station", "aeon", "aeon"],
        "trips": [1, 1, 1, 1],
    })


def cfg(n_resp=4):
    c = {**CFG, "simulation": {**CFG["simulation"], "n_respondents": n_resp, "n_households": 172}}
    return c


def test_demand_total_scales_with_conversion():
    for conv in (0.1, 0.3, 0.5):
        d = build_demand(survey(), cfg(), conv)
        # 往路 4トリップ/4世帯 × 172戸 × 転換率、復路も同数
        assert d["trips"].sum() == pytest.approx(2 * 172 * conv)


def test_return_trip_shifted_by_dwell():
    d = build_demand(survey(), cfg(), 0.3, purposes=["通勤"])
    assert set(d.loc[d["direction"] == "return", "hour"]) == {7 + CFG["dwell_hours"]["通勤"]}


def test_station_share_override():
    d = build_demand(survey(), cfg(), 1.0, station_share=0.25)
    tot = d.groupby("dest")["trips"].sum()
    assert tot["station"] / tot.sum() == pytest.approx(0.25)


def test_zero_trip_respondents_count_in_denominator():
    d_small = build_demand(survey(), cfg(n_resp=4), 1.0)
    d_big = build_demand(survey(), cfg(n_resp=8), 1.0)
    assert d_big["trips"].sum() == pytest.approx(d_small["trips"].sum() / 2)


def test_fleet_monotone_in_conversion():
    totals = []
    for conv in (0.1, 0.3, 0.5):
        d = build_demand(survey(), cfg(), conv)
        r = simulate_fleet(hourly_matrix(d, "outbound"), hourly_matrix(d, "return"),
                           np.array([0.5, 0.5]), 0.95)
        totals.append(r.total)
    assert totals == sorted(totals) and totals[0] < totals[-1]


def test_fleet_deterministic_hand_calc():
    # 7時に10台出発→駅、17時に10台戻る（ばらつき無し: service_level=0.5 → z=0）
    out = np.zeros((24, 2)); out[7, 0] = 10
    ret = np.zeros((24, 2)); ret[17, 0] = 10
    r = simulate_fleet(out, ret, np.array([1.0]), 0.5)
    t = r.table.set_index("ポート")
    assert t.loc["P1", "初期台数"] == 10
    assert t.loc["駅", "初期台数"] == 0
    assert t.loc["駅", "必要区画"] == 10
    assert r.total == 10


def brute_force(C, k):
    best = min(itertools.combinations(range(C.shape[1]), k), key=lambda S: C[:, S].min(1).sum())
    return C[:, best].min(1).sum()


def milp(C, k):
    pulp = pytest.importorskip("pulp")  # テスト専用（厳密解との照合）。無い環境ではこの照合だけ省略
    n, m = C.shape
    prob = pulp.LpProblem("pmed", pulp.LpMinimize)
    y = [pulp.LpVariable(f"y{j}", cat="Binary") for j in range(m)]
    x = [[pulp.LpVariable(f"x{i}_{j}", lowBound=0) for j in range(m)] for i in range(n)]
    prob += pulp.lpSum(C[i, j] * x[i][j] for i in range(n) for j in range(m))
    for i in range(n):
        prob += pulp.lpSum(x[i]) == 1
        for j in range(m):
            prob += x[i][j] <= y[j]
    prob += pulp.lpSum(y) == k
    prob.solve(pulp.PULP_CBC_CMD(msg=False))
    return pulp.value(prob.objective)


@pytest.mark.parametrize("seed", range(5))
def test_pmedian_matches_exact(seed):
    rng = np.random.default_rng(seed)
    pts, cand = rng.random((40, 2)), rng.random((12, 2))
    C = np.linalg.norm(pts[:, None] - cand[None], axis=2) + rng.random(12) * 0.3
    for k in (1, 2, 3, 4):
        sol = solve_pmedian(C, k)
        assert len(set(sol.ports)) == k
        assert sol.objective == pytest.approx(brute_force(C, k), rel=1e-9)
    try:
        exact = milp(C, 3)
    except Exception as e:  # pulp が使えない環境（例：Python 3.14 rc 版）
        pytest.skip(f"pulp を使えないため厳密解との照合を省略: {e}")
    assert solve_pmedian(C, 3).objective == pytest.approx(exact, rel=1e-6)


def test_max_conversion_within_cap():
    from src.sensitivity import max_conversion_within, station_slots
    s, c = survey(), cfg()
    cmax = max_conversion_within(s, c, 10, purposes=None, station_share=None)
    assert 0 < cmax < 1
    assert station_slots(s, c, conversion=cmax, purposes=None, station_share=None) <= 10
    assert station_slots(s, c, conversion=min(1, cmax + 0.02), purposes=None, station_share=None) > 10


def test_load_houses_districts():
    from src.loaders import houses_path, load_houses_all
    paths, is_sample = houses_path(CFG)
    if is_sample:
        pytest.skip("実データなし")
    h = load_houses_all(CFG)
    assert h["街区"].value_counts().to_dict() == {"既存街区": 171, "新規街区": 59}
    assert h["house_id"].is_unique and len(h) == 230
    new = h[h["街区"] == "新規街区"]
    assert sorted(new["区画"]) == list(range(1, 60))                 # lot_no を区画番号として保持
    assert h["lat"].between(36.10, 36.12).all() and h["lon"].between(139.70, 139.72).all()


def test_network_subset_districts():
    net = _real_net()
    if "街区" not in net.houses:
        pytest.skip("事前計算が古い")
    sub = net.subset(["新規街区"])
    assert len(sub.houses) == 59 and sub.W.shape == (59, net.W.shape[1]) and sub.Wn.shape[0] == 59
    assert (sub.houses["街区"] == "新規街区").all()
    assert net.subset(None) is net and net.subset(net.districts) is net


def test_survey_xlsx_weekday_totals():
    from src.loaders import load_survey, resolve
    p = resolve("data/raw/survey.xlsx")
    if not p.exists():
        pytest.skip("アンケートxlsxなし")
    s = load_survey(p, CFG)
    wk = s[s["day"] == "平日"].groupby(["purpose", "direction"])["trips"].sum()
    # シート「平日徒歩平均」（徒歩＋徒歩＋電車）の合計：午前12、午後12
    assert wk[("徒歩＋電車", "outbound")] + wk[("徒歩", "outbound")] == pytest.approx(12)
    assert wk[("徒歩＋電車", "return")] + wk[("徒歩", "return")] == pytest.approx(12)
    assert set(s["hour"]) <= set(range(5, 23))


def test_hourly_override():
    s = survey().assign(direction=["outbound", "outbound", "outbound", "return"])
    c = cfg()
    hourly = ((7, 10.0, 0.0), (15, 0.0, 4.0), (20, 0.0, 6.0))
    d = build_demand(s, c, 0.5, hourly=hourly)
    g = d.groupby(["hour", "direction"])["trips"].sum()
    assert g[(7, "outbound")] == pytest.approx(5.0)
    assert g[(15, "return")] == pytest.approx(2.0)
    assert g[(20, "return")] == pytest.approx(3.0)        # アンケートに無い時間帯は手入力として追加
    assert (10, "outbound") not in g.index                 # 0 にした時間帯は消える
    # 7時の内訳（駅のみ）の比率は保たれる
    assert set(d.loc[(d["hour"] == 7), "dest"]) == {"station"}


def _real_net():
    from src.loaders import load_network
    try:
        net = load_network(CFG)
    except FileNotFoundError:
        pytest.skip("事前計算結果なし")
    if net.Wn is None:
        pytest.skip("全ノード行列なし（precompute を再実行）")
    return net


def test_parse_ports_formats():
    from src.custom_ports import parse_ports
    csv = "名称,緯度,経度\nA,36.1110,139.7080\nB,36.1105,139.7090\n".encode("cp932")
    df = parse_ports("ports.csv", csv)
    assert list(df["name"]) == ["A", "B"] and df["lat"].iloc[0] == pytest.approx(36.111)
    gj = b'{"type":"FeatureCollection","features":[{"type":"Feature","properties":{"name":"X"},' \
         b'"geometry":{"type":"Point","coordinates":[139.708,36.111]}}]}'
    df = parse_ports("p.geojson", gj)
    assert df.iloc[0]["name"] == "X" and df.iloc[0]["lon"] == pytest.approx(139.708)
    from src.loaders import resolve
    gpx = resolve("data/raw/ride_ST3.gpx")
    if gpx.exists():
        df = parse_ports("r.gpx", gpx.read_bytes())       # wpt なし → トラック始点
        assert len(df) == 1 and df.iloc[0]["lat"] == pytest.approx(36.113153)
    with pytest.raises(ValueError):
        parse_ports("p.csv", b"a,b\n1,2\n")


def test_evaluate_ports_matches_optimal():
    from src.custom_ports import evaluate_ports, snap_ports
    from src.sensitivity import run_scenario
    net = _real_net()
    c = cfg(n_resp=4)
    c["simulation"]["n_households"] = len(net.houses)
    kw = dict(conversion=0.3, purposes=None, station_share=None, vehicle="ebike")
    opt = run_scenario(net, survey(), c, k=3, **kw)
    ports = net.candidates.loc[opt.solution.ports, ["lat", "lon"]].reset_index(drop=True).assign(name=["a", "b", "c"])
    ev = evaluate_ports(net, survey(), c, snap_ports(net, ports), **kw)
    assert ev.total_min == pytest.approx(opt.total_min, rel=1e-6)
    assert ev.fleet.total == opt.fleet.total
    # 最適でない配置（1か所を遠くへ）は所要時間が増える
    bad = ports.copy(); bad.loc[0, ["lat", "lon"]] = [36.1120, 139.7125]
    assert evaluate_ports(net, survey(), c, snap_ports(net, bad), **kw).total_min > opt.total_min


# ---------- 運用シミュレーション（混在フリート） ----------
from src.engine import (EB, KB, Fleet, Layout, Ops, _truck, allocate_fleet, month_calendar,  # noqa: E402
                        simulate_day, simulate_days)


def _L(station_cap=np.inf):
    return Layout(k=1, share=np.array([1.0]), walk_m=np.array([100.0]), ride_m=np.array([[600.0, 200.0]]),
                  egress_m=np.array([50.0, 50.0]), alt_m=np.array([[900.0, 300.0]]),
                  caps=np.array([np.inf, station_cap, np.inf]))


def _O(**kw):
    base = dict(walk_kmh=4.8, ride_kmh=(12.0, 12.0), mix=(1.0, 0.0), demand_mult=(1.0, 1.0), procedure_min=0.0,
                kb_range_km=None)
    base.update(kw)
    return Ops(**base)


def _F(house=(0, 0), station=(0, 0)):
    return Fleet(np.array([house, station, (0, 0)]))


def _dem(out7=8.0, ret18=0.0, dest=0):
    out = np.zeros((24, 2)); out[7, dest] = out7
    ret = np.zeros((24, 2)); ret[18, dest] = ret18
    return out, ret


def test_engine_all_served_and_times():
    out, ret = _dem(8.0, 8.0)
    r = simulate_days(_L(), _F((1000, 0), (1000, 0)), _O(), out, ret, n_rep=200, seed=0)
    assert r["貸出失敗率"] == 0 and r["返却失敗率"] == 0
    vw, vr = 4.8 * 1000 / 60, 12 * 1000 / 60
    assert r["平均移動時間(分/片道)"] == pytest.approx(100 / vw + 600 / vr + 50 / vw)
    assert r["利用希望"] == pytest.approx(16, rel=0.05)          # 片道利用：往路8＋帰り（別の人）8


def test_engine_zero_fleet_all_walk():
    out, ret = _dem(8.0, 8.0)
    r = simulate_days(_L(), _F(), _O(), out, ret, n_rep=100, seed=0)
    assert r["充足率"] == 0 and r["平均移動時間(分/片道)"] == pytest.approx(r["全員徒歩の平均(分/片道)"])


def test_engine_matches_poisson_expectation():
    from scipy.stats import poisson
    out, ret = _dem(8.0, 0.0)
    r = simulate_days(_L(), _F((6, 0)), _O(), out, ret, n_rep=3000, seed=1)
    expect = sum(min(x, 6) * poisson.pmf(x, 8) for x in range(60))
    assert r["利用"] == pytest.approx(expect, rel=0.03)


def test_engine_station_capacity_return_failure():
    out, ret = _dem(20.0, 20.0)
    r = simulate_days(_L(station_cap=10), _F((100, 0)), _O(), out, ret, n_rep=300, seed=0)
    # 駅に着いた車両のうち10台を超える分は返却失敗（区画外）→ 夕方の帰りに使えるのは10台まで
    assert r["返却失敗/日"] == pytest.approx(10, abs=0.6)
    served_ret = r["利用"] - 20
    assert served_ret <= 10 + 0.5


def test_engine_substitution_policy():
    out, ret = _dem(5.0, 0.0)
    f = _F((0, 50))                                            # キックボードしか無い・全員が自転車を希望
    r1 = simulate_days(_L(), f, _O(substitution=False), out, ret, n_rep=100, seed=0)
    r2 = simulate_days(_L(), f, _O(substitution=True), out, ret, n_rep=100, seed=0)
    assert r1["充足率"] == 0
    assert r2["充足率"] == 1 and r2["代替乗車率"] == 1


def test_engine_round_trip_pairs_returns():
    out, ret = _dem(6.0, 6.0, dest=1)                          # イオン：往復
    r = simulate_days(_L(), _F((100, 0)), _O(), out, ret, n_rep=200, seed=0)
    assert r["利用希望"] == pytest.approx(12, rel=0.05) and r["充足率"] == 1


def test_engine_kickboard_charging_downtime():
    out, ret = _dem(5.0, 0.0)
    o = _O(mix=(0.0, 1.0), kb_range_km=1.0, kb_swap_below=0.5, kb_out_hours=24)
    r = simulate_days(_L(), _F((0, 1)), o, out, ret, n_rep=200, seed=0)
    # 1台・1回乗ると残量0.4km<0.5km で交換待ち（その日は使えない）→ 1日に1回だけ
    assert r["利用"] <= 1.0 + 1e-9 and r["電池交換/日"] == pytest.approx(r["利用"])


def test_engine_night_reset_and_truck_time():
    out, ret = _dem(8.0, 0.0)
    rng = np.random.default_rng(0)
    f = _F((10, 0))
    st, S = simulate_day(_L(), f, _O(), out, ret, rng)
    assert [S.count(p) for p in range(3)] == [10, 0, 0]       # 終電後に初期配置へ
    assert st.ops and st.ops[-1][1] == st.served_out           # 駅に行った台数だけ戻す
    o = _O()
    assert _truck(o, 4, 8) == (30 + 3 * 12 + 10 * 1, 2)        # 自転車4台＋キックボード8台＝2往復


def test_allocate_fleet_station_and_cap():
    out, ret = _dem(10.0, 10.0)
    F = allocate_fleet(_L(station_cap=4), _O(), 20, 0, out, ret, n_samp=500)
    assert F.init.sum() == 20 and F.init[1].sum() <= 4 and F.init[1].sum() > 0


def test_month_calendar_holidays():
    cal = month_calendar(2026, 11, CFG["month"]["holidays_2026"], CFG["climate"][11], np.random.default_rng(0))
    assert len(cal) == 30 and (cal["曜日区分"] == "休日").sum() == 11   # 土日9日＋文化の日・勤労感謝の日


def test_min_fleet_real_data():
    from src.plan import demand_hourly, make_ops, optimal_layout
    from src.study import min_fleet
    net = _real_net()
    c = {**CFG, "simulation": {**CFG["simulation"], "n_households": len(net.houses)}}
    from src.loaders import load_survey, survey_path
    sv = load_survey(survey_path(c)[0], c)
    oh, rh, sh = demand_hourly(sv, c, 0.1, c["survey_xlsx"]["default_modes"], "平日")
    L, _, _ = optimal_layout(net, c, sh, 3, 0.603)
    r = min_fleet(L, make_ops(c), oh, rh, targets=(0.2,), n_rep=20, seed=0)[0.2]
    assert r is not None and r["失敗率"] <= 0.2 and r["合計"] == r["自転車"] + r["キックボード"]


def test_rebalance_b_station_buffer():
    """b：9:30に駅の在庫を「夕方の期待需要＋1台」（または指定分位点）まで減らし、住宅側の不足へ戻す。"""
    out, ret = _dem(20.0, 6.0)
    res = {}
    for q in (None, 0.95):
        st, S = simulate_day(_L(), _F((40, 0)), _O(pattern="b", station_buffer_q=q), out, ret, np.random.default_rng(1))
        res[q] = ([o for o in st.ops if o[0] < 12][0][1], st.served_out)
    moved, at_station = res[None]
    assert moved == at_station - (6 + 1)
    assert res[0.95][0] == at_station - 10                      # ポアソン(6)の95%点＝10台を残す


def test_full_return_collect():
    """満車で返せない車両：outside は夜まで区画外、collect は回収作業が記録され住宅側へ戻る。"""
    out, ret = _dem(10.0, 8.0)
    f = _F((20, 0))
    st_o, S_o = simulate_day(_L(station_cap=3), f, _O(), out, ret, np.random.default_rng(3))
    st_c, S_c = simulate_day(_L(station_cap=3), f, _O(full_return="collect", collect_delay_min=30), out, ret,
                             np.random.default_rng(3))
    assert st_o.return_fail > 0 and st_c.return_fail == st_o.return_fail
    day_ops = [o for o in st_c.ops if o[0] < 24]
    assert day_ops and day_ops[0][1] >= 1                    # 日中に回収作業がある
    assert not [o for o in st_o.ops if o[0] < 24]            # outside は日中の作業なし


def test_discourage_lowers_next_day_demand():
    from src.engine import month_calendar, simulate_month
    out, ret = _dem(20.0, 0.0)
    cal = month_calendar(2026, 11, CFG["month"]["holidays_2026"], CFG["climate"][11], np.random.default_rng(0))
    cal["曜日区分"], cal["気象"] = "平日", "晴れ"
    ops = _O()
    f = _F((5, 0))                                           # 台数が足りず失敗が多い
    s0, d0, _ = simulate_month(_L(), f, lambda w: ops, lambda dt, w: (out, ret), cal, seed=1, discourage=0.0)
    s1, d1, _ = simulate_month(_L(), f, lambda w: ops, lambda dt, w: (out, ret), cal, seed=1, discourage=1.0)
    assert (d0["需要の倍率（利用離れ）"] == 1).all() and d1["需要の倍率（利用離れ）"].iloc[1:].mean() < 0.95
    assert s1["利用希望"] < s0["利用希望"]


def test_year_eval_and_cost():
    from src.plan import demand_hourly, fleet_for, make_ops, optimal_layout
    from src.study import annual_cost, year_eval
    from src.loaders import load_survey, survey_path
    net = _real_net()
    c = {**CFG, "simulation": {**CFG["simulation"], "n_households": len(net.houses)}}
    sv = load_survey(survey_path(c)[0], c)
    P = c["survey_xlsx"]["default_modes"]
    oh, rh, sh = demand_hourly(sv, c, 0.12, P, "平日")
    oh2, rh2, _ = demand_hourly(sv, c, 0.12, P, "休日")
    L, _, _ = optimal_layout(net, c, sh, 3, 0.603)
    F = fleet_for(L, 12, 8, oh, rh, make_ops(c))
    summ, daily, by_month = year_eval(L, F, c, {"平日": (oh, rh), "休日": (oh2, rh2)}, seed=0)
    assert len(daily) == 365 and len(by_month) == 12
    assert by_month["利用希望"].std() > 0                      # 月ごとに曜日・気象が違う
    cost = annual_cost(c, F, L.k, daily, 25)
    assert cost["人件費（電池交換）"] >= 0 and abs(cost["年間利用回数"] - daily["利用"].sum()) < 1e-6
    assert cost["年間総費用"] == pytest.approx(sum(cost[k] for k in ("車両（償却）", "車両維持", "ポート（償却）", "用地",
                                                                "人件費（リバランス）", "軽トラック", "人件費（電池交換）")))


def test_import_usage_log_and_survey_template():
    import importlib.util
    spec = importlib.util.spec_from_file_location("iul", ROOT / "scripts" / "import_usage_log.py")
    iul = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(iul)
    log = pd.read_csv(ROOT / "data" / "templates" / "usage_log_template.csv")
    r = iul.analyze(log, CFG, CFG["month"]["holidays_2026"])
    assert r["days"] == {"平日": 1, "休日": 1}                     # 11/3 は文化の日
    assert r["trips_per_day"]["平日"] == 2 and abs(r["ebike_share_obs"] - 1 / 3) < 1e-9
    h = r["hourly"]
    assert set(h["方向"]) == {"往路", "復路"} and set(h["目的地"]) == {"station", "aeon"}
    # 調査票ひな形（1行＝1回の外出）をそのまま読める
    from src.loaders import load_survey
    sv = pd.read_excel(ROOT / "data" / "templates" / "survey_template.xlsx", sheet_name="回答")
    p = ROOT / "data" / "templates" / "_tmp_survey.csv"
    sv.to_csv(p, index=False)
    try:
        s = load_survey(p, CFG)
    finally:
        p.unlink()
    assert list(s["destination"]) == ["station", "aeon", "other"] and (s["trips"] == 1).all()


def test_pilot_grid_fixed_fleet():
    from src.loaders import load_survey, survey_path
    from src.pilot import pilot_best, pilot_grid
    net = _real_net()
    c = {**CFG, "simulation": {**CFG["simulation"], "n_households": len(net.houses)}}
    sv = load_survey(survey_path(c)[0], c)
    g = pilot_grid(net, c, sv, 0.3, n_total=10, ks=(1, 2), n_rep=20)
    assert len(g) == 2 * 11 and ((g["自転車"] + g["キックボード"]) == 10).all()
    b = pilot_best(g)
    assert b["利用"] <= 10 * 2 + 1e-9                          # 10台×朝夕1往復で頭打ち（夜に初期配置へ戻す）
    assert b["断られた/日"] > 0
