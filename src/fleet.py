"""時間帯別の在庫シミュレーションによる必要台数（再配置なし）。

各ポートで1日の始めに置く台数 S0 は、
    不足量 D_t = (t時までの累積出発) − (t-1時までの累積到着)
が最大になる時点で車両が足りる台数とする（同じ時間帯の到着は出発に使えない＝安全側）。
往路と復路は同じ利用者の往復なので、出発・到着を独立に扱うとばらつきを過大に見積もる。
そこで「その時点で出払っている台数（＝往路を終え復路をまだ終えていない往復の数）」を
ポアソン分布とみなし、サービス水準の分位点で台数を決める：
    S0 = max_t PoissonPPF(サービス水準, max(D_t, 0))
必要区画数（最大在庫）は S0 ＋ 純流入の最大値（同じくポアソン分位点）。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import poisson

from .demand import DESTS


@dataclass
class FleetResult:
    table: pd.DataFrame  # port, kind, 初期台数, 必要区画, 出発トリップ/日
    total: int
    station_slots: int


def _quantile(mean: float, sl: float) -> int:
    return int(poisson.ppf(sl, mean)) if mean > 1e-9 else 0


def _port_need(dep: np.ndarray, arr: np.ndarray, sl: float) -> tuple[int, int]:
    if dep.sum() <= 0 and arr.sum() <= 0:
        return 0, 0
    cdep = np.cumsum(dep)
    carr = np.cumsum(arr)
    carr_prev = np.concatenate([[0.0], carr[:-1]])
    s0 = _quantile(max(0.0, float(np.max(cdep - carr_prev))), sl)
    slots = max(s0, s0 + _quantile(max(0.0, float(np.max(carr - cdep))), sl))
    return s0, slots


def simulate_fleet(out_h: np.ndarray, ret_h: np.ndarray, port_share: np.ndarray,
                   service_level: float, port_labels: list[str] | None = None) -> FleetResult:
    """out_h, ret_h: (24, n_dest) 往路・復路トリップ。port_share: 街区ポートごとの需要割合（合計1）。"""
    out_tot = out_h.sum(1)
    ret_tot = ret_h.sum(1)
    rows = []
    for j, q in enumerate(port_share):
        s0, slots = _port_need(q * out_tot, q * ret_tot, service_level)
        label = port_labels[j] if port_labels else f"P{j + 1}"
        rows.append((label, "街区", s0, slots, float(q * out_tot.sum())))
    dest_label = {"station": "駅", "aeon": "イオン"}
    for d, key in enumerate(DESTS):
        s0, slots = _port_need(ret_h[:, d], out_h[:, d], service_level)
        rows.append((dest_label[key], key, s0, slots, float(ret_h[:, d].sum())))
    table = pd.DataFrame(rows, columns=["ポート", "種別", "初期台数", "必要区画", "出発トリップ/日"])
    st = table.loc[table["種別"] == "station", "必要区画"]
    return FleetResult(table=table, total=int(table["初期台数"].sum()),
                       station_slots=int(st.iloc[0]) if len(st) else 0)
