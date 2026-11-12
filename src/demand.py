"""アンケート → 街区全体の時間帯別・目的地別トリップ（期待値）。"""
from __future__ import annotations

import numpy as np
import pandas as pd

HOURS = np.arange(24)
DESTS = ["station", "aeon"]


def survey_station_share(survey: pd.DataFrame, cfg: dict) -> dict[str, float]:
    """用途ごとの駅の割合。アンケートに目的地があればそこから、無ければ config の既定値。"""
    default = cfg["default_station_share_by_purpose"]
    shares = {}
    for p, g in survey.groupby("purpose"):
        if "destination" in g.columns:
            st = g.loc[g["destination"] == "station", "trips"].sum()
            ae = g.loc[g["destination"] == "aeon", "trips"].sum()
            if st + ae > 0:
                shares[p] = float(st / (st + ae))
                continue
        shares[p] = float(default.get(p, 0.5))
    return shares


def build_demand(
    survey: pd.DataFrame,
    cfg: dict,
    conversion: float,
    purposes: list[str] | None = None,
    station_share: float | None = None,
    day: str | None = None,
    hourly: tuple | None = None,
) -> pd.DataFrame:
    """列: hour, purpose, dest, direction(outbound=自宅→目的地 / return=目的地→自宅), trips。

    trips = 戸数 × (アンケートの時間帯別トリップ ÷ 回答世帯数) × 転換率 × 目的地比率
    station_share を与えると全用途の駅の割合をその値で上書きする。
    day: アンケートに曜日区分（平日/休日）がある場合の絞り込み。
    hourly: ((時, 往路, 復路), ...) 街区全体・転換前のトリップ数/日。与えると時間帯ごとの合計をこの値に置き換える。

    2026-11：config の paths.demand_table（街区230戸の時間帯別・目的地別の移動需要、公的統計ベース）があれば、
    アンケートの代わりにそれを使う（trips＝表の値 × 戸数/230 × 転換率。用途の絞り込みは使わない）。
    """
    tab = demand_table(cfg)
    if tab is not None:
        return _from_table(tab, cfg, conversion, station_share, hourly)
    if day is not None and "day" in survey.columns:
        survey = survey[survey["day"] == day]
    sim = cfg["simulation"]
    n_resp = sim.get("n_respondents") or survey["respondent_id"].nunique()
    scale = sim["n_households"] / n_resp * conversion

    s = survey[survey["trips"] > 0]
    if purposes is not None:
        s = s[s["purpose"].isin(purposes)]
    if "destination" in s.columns:
        s = s[s["destination"] != "other"]  # 対象2施設以外の目的地は除外
    if "direction" not in s.columns:
        s = s.assign(direction="outbound")
    shares = survey_station_share(survey, cfg)

    base = s.groupby(["hour", "purpose", "direction"], as_index=False)["trips"].sum()
    base["trips"] *= scale
    rows = []
    for r in base.itertuples(index=False):
        sh = station_share if station_share is not None else shares.get(r.purpose, 0.5)
        for dest, w in (("station", sh), ("aeon", 1 - sh)):
            if w > 0:
                rows.append((r.hour, r.purpose, dest, r.direction, r.trips * w))
    out = pd.DataFrame(rows, columns=["hour", "purpose", "dest", "direction", "trips"])

    if "direction" not in survey.columns and len(out):
        # 往路のみ → 用途別の滞在時間後に同数の復路を生成（23時で打ち切り）
        dwell = cfg["dwell_hours"]
        ret = out.copy()
        ret["hour"] = [min(23, h + int(dwell.get(p, cfg["default_dwell_hours"])))
                       for h, p in zip(ret["hour"], ret["purpose"])]
        ret["direction"] = "return"
        out = pd.concat([out, ret], ignore_index=True)
    if hourly is not None:
        out = _override_hourly(out, hourly, conversion, station_share)
    return out


# ---------------------------------------------------------------- 2026-11：時間帯別・目的地別の需要表
TABLE_HOUSEHOLDS = 230         # 表の前提の戸数（既存171＋新規59）


def demand_table(cfg) -> pd.DataFrame | None:
    """paths.demand_table の表（hour, destination, direction, trips）。無ければ None。"""
    from .loaders import resolve
    p = (cfg.get("paths") or {}).get("demand_table")
    if not p:
        return None
    path = resolve(p)
    if not path.exists():
        return None
    return _read_table(str(path), path.stat().st_mtime)


_TAB_CACHE: dict = {}


def _read_table(path: str, mtime: float) -> pd.DataFrame:
    key = (path, mtime)
    if key not in _TAB_CACHE:
        _TAB_CACHE[key] = pd.read_csv(path)
    return _TAB_CACHE[key]


def table_hourly(cfg, dest: str, conversion: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """需要表の目的地 dest の (往路 (24,), 復路 (24,))。× 戸数/230 × 転換率。"""
    tab = demand_table(cfg)
    o, r = np.zeros(24), np.zeros(24)
    if tab is None:
        return o, r
    sc = cfg["simulation"].get("n_households", TABLE_HOUSEHOLDS) / TABLE_HOUSEHOLDS * conversion
    t = tab[tab["destination"] == dest]
    for row in t.itertuples(index=False):
        (o if row.direction == "outbound" else r)[int(row.hour)] += row.trips * sc
    return o, r


def _from_table(tab, cfg, conversion, station_share, hourly) -> pd.DataFrame:
    sc = cfg["simulation"].get("n_households", TABLE_HOUSEHOLDS) / TABLE_HOUSEHOLDS * conversion
    t = tab[tab["destination"].isin(DESTS)]
    out = pd.DataFrame({"hour": t["hour"].astype(int), "purpose": "需要表", "dest": t["destination"],
                        "direction": t["direction"], "trips": t["trips"] * sc})
    if station_share is not None and len(out):          # 駅の割合を指定：合計は保ち、目的地の比率だけ変える
        tot = out.groupby("dest")["trips"].sum()
        for d, w in (("station", station_share), ("aeon", 1 - station_share)):
            if tot.get(d, 0) > 0:
                out.loc[out["dest"] == d, "trips"] *= w * tot.sum() / tot[d]
    if hourly is not None:
        out = _override_hourly(out, hourly, conversion, station_share)
    return out.reset_index(drop=True)


def _override_hourly(out: pd.DataFrame, hourly: tuple, conversion: float,
                     station_share: float | None) -> pd.DataFrame:
    """時間帯・方向ごとの合計を指定値（×転換率）に合わせる。内訳（目的地・用途）の比率は保つ。"""
    if station_share is None:
        tot = out.groupby("dest")["trips"].sum()
        station_share = float(tot.get("station", 0) / tot.sum()) if tot.sum() > 0 else 0.5
    rows = [out.iloc[0:0]]
    for h, ob, rt in hourly:
        for direction, target in (("outbound", ob), ("return", rt)):
            target = float(target) * conversion
            cur = out[(out["hour"] == h) & (out["direction"] == direction)]
            if target <= 0:
                continue
            if cur["trips"].sum() > 0:
                rows.append(cur.assign(trips=cur["trips"] * target / cur["trips"].sum()))
            else:
                rows.append(pd.DataFrame([(h, "手入力", d, direction, target * w)
                                          for d, w in (("station", station_share), ("aeon", 1 - station_share)) if w > 0],
                                         columns=out.columns))
    return pd.concat(rows, ignore_index=True)


def base_hourly(survey: pd.DataFrame, cfg: dict, purposes: list[str] | None, day: str | None,
                hours=range(5, 23)) -> pd.DataFrame:
    """アンケートから求めた街区全体・転換前の時間帯別トリップ数/日（サイドバー編集の初期値）。"""
    d = build_demand(survey, cfg, 1.0, purposes, None, day=day)
    p = d.pivot_table(index="hour", columns="direction", values="trips", aggfunc="sum")
    p = p.reindex(index=list(hours), columns=["outbound", "return"]).fillna(0.0)
    return pd.DataFrame({"時": p.index, "往路": p["outbound"].round(1).to_numpy(), "復路": p["return"].round(1).to_numpy()})


def dest_shares(demand: pd.DataFrame) -> np.ndarray:
    """トリップ数で重み付けした目的地比率（DESTS 順）。"""
    tot = demand.groupby("dest")["trips"].sum().reindex(DESTS, fill_value=0.0).to_numpy()
    return tot / tot.sum() if tot.sum() > 0 else np.array([0.5, 0.5])


def hourly_matrix(demand: pd.DataFrame, direction: str) -> np.ndarray:
    """(24, n_dest) の時間帯別トリップ数。"""
    d = demand[demand["direction"] == direction]
    m = d.pivot_table(index="hour", columns="dest", values="trips", aggfunc="sum")
    return m.reindex(index=HOURS, columns=DESTS, fill_value=0.0).fillna(0.0).to_numpy()
