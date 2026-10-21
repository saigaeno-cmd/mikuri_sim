"""混在フリート（電動アシスト自転車＋電動キックボード）の運用シミュレーション（1日／1か月）。

ポート：街区側 0..k-1、駅西口ポート k、モビリティステーション（イオン）k+1。
需要（すべて期待値、件/日、時間帯別）
- 住宅→目的地d（往路）：ポートj・時間帯h・希望車種t ごとに ポアソン(住宅割合_j × 往路[h,d] × 車種比率_t × 気象倍率_t)。
- 目的地が「片道利用」（駅）：帰りは別の利用者として、ポアソン(住宅割合_j × 復路[h,d] × 車種比率_t × 気象倍率_t) が
  目的地ポートで発生し、自宅側ポート j へ向かう。
- 目的地が「往復利用」（イオン）：往路で乗った人が、復路需要の時間分布（到着以降）から抽選した時刻に、
  同じ車種を希望して戻る。往路で乗れなかった人は往復とも徒歩。
貸出：希望車種の空き車両（キックボードは充電・交換中を除き、電池残量の多いもの）。無ければ
  (i) 利用しない（貸出失敗）／(ii) もう一方の車種に乗る（代替乗車。無ければ貸出失敗）。
返却：到着ポートが満車なら「返却失敗」。車両は区画外に一時駐車し、次のリバランス作業で運営者が回収する
  （それまで貸出には使えない）。
電池：キックボードは走行距離だけ残量が減り、交換基準を下回ると一定時間（out_hours）貸出不可→満充電に戻る。
リバランス：毎晩（終電後）初期配置に戻す（a）。b は 9:30、c はさらに 15:30 に在庫を調整する。
移動時間：利用＝徒歩(自宅⇔ポート)＋手続き＋走行＋徒歩(ポート⇔施設)、貸出失敗＝徒歩のみ。
"""
from __future__ import annotations

import dataclasses
import heapq
import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.stats import poisson

TYPES = ("ebike", "kickboard")
EB, KB = 0, 1
STATION, AEON = 0, 1          # 目的地 index（DESTS 順）
_ARR, _REB, _REQ = 0, 1, 2    # 同時刻の処理順：到着 → 作業 → 利用希望


@dataclass
class Layout:
    """配置（距離は m）。walk/alt は各ポートに割り当てた住宅の平均。"""
    k: int
    share: np.ndarray          # (k,) 住宅の割合
    walk_m: np.ndarray         # (k,) 自宅→ポート
    ride_m: np.ndarray         # (k, 2) ポート→目的地ポート
    egress_m: np.ndarray       # (2,) 目的地ポート→施設
    alt_m: np.ndarray          # (k, 2) 徒歩のみで自宅→施設
    caps: np.ndarray           # (k+2,) 収納可能台数（np.inf=制限なし）
    labels: list = field(default_factory=list)


@dataclass
class Ops:
    """運用・行動の条件。"""
    walk_kmh: float
    ride_kmh: tuple            # (ebike, kickboard)
    mix: tuple                 # 希望車種の比率 (ebike, kickboard)
    demand_mult: tuple         # 気象による需要倍率 (ebike, kickboard)
    substitution: bool = False
    procedure_min: float = 1.0
    trip_type: tuple = ("one_way", "round_trip")   # (駅, イオン)
    pattern: str = "a"
    reb_times: tuple = (9.5, 15.5)
    truck_cap: tuple = (4, 8)                       # (ebike, kickboard)
    base_min: float = 30.0
    per_vehicle_min: float = 3.0
    extra_trip_min: float = 10.0
    kb_range_km: float | None = 30.0
    kb_swap_below: float = 0.2
    kb_out_hours: float = 4.0
    nightly_charge: bool = False
    station_buffer_q: float | None = None   # 駅に残す台数：None＝期待値＋1台、数値＝ポアソン分布のその分位点
    full_return: str = "outside"            # 満車で返せないとき："outside"＝区画外に置き次の作業で回収／"collect"＝運営者が回収に出る
    collect_delay_min: float = 60.0         # "collect" のとき、最初の返却失敗から回収するまでの分


@dataclass
class Fleet:
    """初期配置（毎晩この状態に戻す）：ポートごとの車種別台数 (k+2, 2)。行＝街区ポート…、駅、モビリティステーション。"""
    init: np.ndarray

    @property
    def n(self) -> tuple:
        return int(self.init[:, EB].sum()), int(self.init[:, KB].sum())


def _hourly_counts_need(L: Layout, ops, out_h, ret_h, n_samp: int, rng) -> np.ndarray:
    """需要だけをサンプルして、各場所・車種で「その日を通して不足しないために朝に必要な台数」の分布を作る。
    戻り値 (n_samp, k+2, 2)。時間帯単位で、同じ時間帯の到着は出発に使えないとみなす（安全側）。"""
    k = L.k
    sp, ap = k, k + 1
    lam_t = [ops.mix[t] * ops.demand_mult[t] for t in (EB, KB)]
    need = np.zeros((n_samp, k + 2, 2))
    for t in (EB, KB):
        dep = np.zeros((n_samp, k + 2, 24))
        arr = np.zeros((n_samp, k + 2, 24))
        for d, port in ((STATION, sp), (AEON, ap)):
            lam = L.share[:, None] * out_h[None, :, d] * lam_t[t]               # (k, 24)
            c = rng.poisson(lam, size=(n_samp,) + lam.shape)
            dep[:, :k] += c
            arr[:, port] += c.sum(1)
            if ops.trip_type[d] == "one_way":
                lamr = L.share[:, None] * ret_h[None, :, d] * lam_t[t]
                r = rng.poisson(lamr, size=(n_samp,) + lamr.shape)
                dep[:, port] += r.sum(1)
                arr[:, :k] += r
            else:   # 往復：往路で乗った車両で戻るので、目的地側の在庫は要らない（街区側には夕方戻る）
                w = ret_h[:, d] / ret_h[:, d].sum() if ret_h[:, d].sum() > 0 else np.zeros(24)
                arr[:, :k] += c.sum(2)[:, :, None] * w[None, None, :]
        cdep = np.cumsum(dep, 2)
        carr = np.concatenate([np.zeros((n_samp, k + 2, 1)), np.cumsum(arr, 2)[:, :, :-1]], 2)
        need[:, :, t] = np.maximum((cdep - carr).max(2), 0)
    return need


def allocate_fleet(L: Layout, ops, n_eb: int, n_kb: int, out_h, ret_h, n_samp: int = 1500, seed: int = 0) -> Fleet:
    """自転車 n_eb 台・キックボード n_kb 台を、街区ポートと駅ポートに配る（毎晩の初期配置）。
    各場所・車種の必要台数の分布 R に対し、「1台増やすと不足しなくなる確率 P(R ≥ s+1)」が最大の所へ1台ずつ配る
    （分離可能な凹関数の貪欲法）。駅ポートの容量は超えない。"""
    rng = np.random.default_rng(seed)
    need = _hourly_counts_need(L, ops, out_h, ret_h, n_samp, rng)          # (n, k+2, 2)
    init = np.zeros((L.k + 2, 2), dtype=int)
    left = {EB: int(n_eb), KB: int(n_kb)}
    while left[EB] + left[KB] > 0:
        best, best_gain = None, -1.0
        for t in (EB, KB):
            if left[t] == 0:
                continue
            for p in range(L.k + 1):                                       # モビリティステーションには置かない
                if init[p].sum() >= L.caps[p]:
                    continue
                g = float((need[:, p, t] >= init[p, t] + 1).mean())
                if g > best_gain:
                    best, best_gain = (p, t), g
        if best is None:
            break
        if best_gain <= 0:  # どこも不足しない：住宅割合に応じて街区側へ
            t = EB if left[EB] else KB
            p = int(np.argmax(L.share * 1.0 - init[:L.k].sum(1) / max(init.sum(), 1)))
            best = (p, t)
        init[best] += 1
        left[best[1]] -= 1
    return Fleet(init)


def allocate_types(n_eb: int, n_kb: int, share: np.ndarray, peak_by_type: tuple) -> Fleet:
    """（旧）街区ポートだけに配る簡易配分。互換のため残す。"""
    k = len(share)
    init = np.zeros((k + 2, 2), dtype=int)
    for t, n in ((EB, n_eb), (KB, n_kb)):
        m = share * max(peak_by_type[t], 1e-6)
        for _ in range(int(n)):
            gain = poisson.sf(init[:k, t], m)
            gain[share <= 0] = -1
            init[int(np.argmax(gain)), t] += 1
    return Fleet(init)


def peak_outflow(out_h: np.ndarray, ret_h: np.ndarray) -> float:
    cdep = np.cumsum(out_h.sum(1))
    carr = np.concatenate([[0.0], np.cumsum(ret_h.sum(1))[:-1]])
    return float(max(0.0, np.max(cdep - carr)))


# ---------------------------------------------------------------- 状態
class _State:
    def __init__(self, layout: Layout, fleet: Fleet, ops: Ops, rng, soc0=None):
        self.L, self.O = layout, ops
        self.nport = layout.k + 2
        types, home = [], []
        for j in range(fleet.init.shape[0]):
            for t in (EB, KB):
                for _ in range(int(fleet.init[j, t])):
                    types.append(t)
                    home.append(j)
        self.vtype = np.array(types, dtype=int)
        self.home = np.array(home, dtype=int)
        self.nv = len(types)
        self.loc = self.home.copy()            # ポート番号／-1 走行中／-2 区画外
        rng_km = ops.kb_range_km or np.inf
        if soc0 is None:
            soc0 = np.where(self.vtype == KB, rng.uniform(0.3, 1.0, self.nv) * rng_km, np.inf)
        self.soc = np.asarray(soc0, dtype=float).copy()
        self.ready_at = np.zeros(self.nv)       # 充電・交換で使えない時刻まで
        self._rebuild()

    def _rebuild(self):
        self.at = [[[] for _ in range(2)] for _ in range(self.nport)]
        for v in range(self.nv):
            if self.loc[v] >= 0:
                self.at[self.loc[v]][self.vtype[v]].append(v)

    def count(self, p):
        return len(self.at[p][0]) + len(self.at[p][1])

    def take(self, p, t, now, need_km):
        best, best_soc = None, -1.0
        for v in self.at[p][t]:
            if self.ready_at[v] <= now and self.soc[v] >= need_km and self.soc[v] > best_soc:
                best, best_soc = v, self.soc[v]
        if best is not None:
            self.at[p][t].remove(best)
            self.loc[best] = -1
        return best

    def available(self, p, t, now):
        return sum(1 for v in self.at[p][t] if self.ready_at[v] <= now)

    def put(self, v, p):
        self.loc[v] = p
        self.at[p][self.vtype[v]].append(v)

    def remove(self, v):
        p = self.loc[v]
        if p >= 0:
            self.at[p][self.vtype[v]].remove(v)


# ---------------------------------------------------------------- 1日
@dataclass
class DayStats:
    req_out: int = 0
    req_ret: int = 0
    served_out: int = 0
    served_ret: int = 0
    fail_out: int = 0
    fail_ret: int = 0
    substitutions: int = 0
    return_fail: int = 0
    time_min: float = 0.0
    walk_only_min: float = 0.0
    legs: int = 0
    station_peak: int = 0
    rides_by_type: list = field(default_factory=lambda: [0, 0])
    ops: list = field(default_factory=list)        # [(時刻, 移動台数, 作業分, トラック往復数)]
    hourly: np.ndarray = field(default_factory=lambda: np.zeros((24, 4)))  # 希望・利用・失敗・返却失敗
    port_fail: np.ndarray | None = None
    port_peak: np.ndarray | None = None       # ポートごとの最大駐車台数（用地＝区画数の算定に使う）
    swaps: int = 0


def _truck(ops: Ops, n_eb: int, n_kb: int) -> tuple[float, int]:
    n = n_eb + n_kb
    if n == 0:
        return 0.0, 0
    trips = max(1, math.ceil(n_kb / ops.truck_cap[KB] + n_eb / ops.truck_cap[EB] - 1e-9))
    return ops.base_min + ops.per_vehicle_min * n + ops.extra_trip_min * (trips - 1), trips


def _expected_after(arr24: np.ndarray, t_h: float) -> float:
    """時間帯別の期待値（件/時）のうち、t_h 時以降の合計。"""
    h0 = int(t_h)
    return float(arr24[h0] * (h0 + 1 - t_h) + arr24[h0 + 1:].sum())


def _collect_outside(S: _State) -> list:
    """区画外の車両を住宅側ポートの不足（初期配置との差）へ戻す。戻した台数 [自転車, キックボード]。"""
    k = S.L.k
    moved = [0, 0]
    outside = [v for v in range(S.nv) if S.loc[v] == -2]
    house_cnt = np.array([[len(S.at[j][t]) for t in (EB, KB)] for j in range(k)])
    for v in outside:
        t = S.vtype[v]
        j = int(np.argmax(S.init_pt[:k, t] - house_cnt[:, t]))
        S.loc[v] = -3
        S.put(v, j)
        house_cnt[j, t] += 1
        moved[t] += 1
    return moved


def _rebalance(S: _State, st: DayStats, now: float, kind: str, out_h, ret_h, lam_scale):
    """kind: 'b'（9:30）/'c'（15:30）。区画外の車両もこの時に回収する。"""
    L, O = S.L, S.O
    k, sp = L.k, L.k
    moved = [0, 0]

    def move(v, p):
        S.remove(v)
        S.put(v, p)
        moved[S.vtype[v]] += 1

    # 区画外の回収（住宅側の不足へ）
    moved = _collect_outside(S)
    house_cnt = np.array([[len(S.at[j][t]) for t in (EB, KB)] for j in range(k)])
    t_h = now / 60.0
    ret_station = _expected_after(ret_h[:, STATION] * lam_scale, t_h) if O.trip_type[STATION] == "one_way" else 0.0
    if O.station_buffer_q is None:
        target_s = math.ceil(ret_station) + 1
    else:
        target_s = int(poisson.ppf(O.station_buffer_q, max(ret_station, 1e-9)))
    stock_s = S.count(sp)
    if kind == "b":
        excess = stock_s - target_s
        deficit = np.maximum(S.init_pt[:k] - house_cnt, 0)
        while excess > 0 and deficit.sum() > 0:
            j, t = np.unravel_index(int(np.argmax(deficit)), deficit.shape)
            cand = S.at[sp][t] or S.at[sp][1 - t]
            if not cand:
                break
            v = cand[0]
            if S.count(j) >= L.caps[j]:
                deficit[j, :] = 0
                continue
            move(v, j)
            deficit[j, t] -= 1
            excess -= 1
    else:  # c：住宅側（イオン往復用）と駅（帰宅用）をそろえる
        aeon_after = _expected_after(out_h[:, AEON] * lam_scale, t_h)
        tgt_house = np.ceil(L.share * aeon_after).astype(int) + 1
        cur_house = np.array([S.count(j) for j in range(k)])
        if stock_s > target_s:
            need = np.maximum(tgt_house - cur_house, 0)
            n_move = stock_s - target_s
            while n_move > 0 and need.sum() > 0:
                j = int(np.argmax(need))
                cand = S.at[sp][EB] + S.at[sp][KB]
                if not cand or S.count(j) >= L.caps[j]:
                    need[j] = 0
                    continue
                move(cand[0], j)
                need[j] -= 1
                n_move -= 1
        elif stock_s < target_s:
            surplus = np.maximum(cur_house - tgt_house, 0)
            n_move = min(target_s - stock_s, int(L.caps[sp] - stock_s) if np.isfinite(L.caps[sp]) else 10**9)
            while n_move > 0 and surplus.sum() > 0:
                j = int(np.argmax(surplus))
                cand = S.at[j][EB] + S.at[j][KB]
                if not cand:
                    surplus[j] = 0
                    continue
                move(cand[0], sp)
                surplus[j] -= 1
                n_move -= 1
    minutes, trips = _truck(O, moved[EB], moved[KB])
    if minutes > 0:
        st.ops.append((now / 60.0, moved[EB] + moved[KB], minutes, trips))


def _night_reset(S: _State, st: DayStats):
    """終電後：全車両を初期配置へ戻す。すでに初期配置の位置にある車両は動かさず、それ以外を移動台数とする。"""
    O = S.O
    want = S.init_pt.copy()
    keep = np.zeros(S.nv, dtype=bool)
    for v in range(S.nv):
        p, t = S.loc[v], S.vtype[v]
        if 0 <= p < want.shape[0] and want[p, t] > 0:
            want[p, t] -= 1
            keep[v] = True
    moved = [0, 0]
    for v in np.flatnonzero(~keep):
        t = S.vtype[v]
        j = int(np.argmax(want[:, t]))
        want[j, t] -= 1
        S.loc[v] = j
        moved[t] += 1
    S._rebuild()
    if O.nightly_charge and O.kb_range_km:
        S.soc[S.vtype == KB] = O.kb_range_km
        S.ready_at[:] = 0
    minutes, trips = _truck(O, moved[EB], moved[KB])
    if minutes > 0:
        st.ops.append((24.0, moved[EB] + moved[KB], minutes, trips))


def simulate_day(layout: Layout, fleet: Fleet, ops: Ops, out_h: np.ndarray, ret_h: np.ndarray,
                 rng: np.random.Generator, state: _State | None = None) -> tuple[DayStats, _State]:
    """1日分。state を渡すと前日の状態（電池・充電中）を引き継ぐ（位置は毎晩初期配置に戻る）。"""
    L, O = layout, ops
    S = state if state is not None else _State(L, fleet, O, rng)
    S.init_pt = fleet.init.copy()
    S.collect_pending = False
    # 前日からの充電待ちは時刻を1日分進める
    S.ready_at = np.maximum(S.ready_at - 24 * 60, 0)
    st = DayStats(port_fail=np.zeros(L.k + 2))
    st.port_peak = np.array([S.count(p) for p in range(L.k + 2)], dtype=float)
    st.station_peak = S.count(L.k)
    k, sp, ap = L.k, L.k, L.k + 1
    dport = (sp, ap)
    vw = O.walk_kmh * 1000 / 60
    vr = [x * 1000 / 60 for x in O.ride_kmh]
    lam_t = [O.mix[t] * O.demand_mult[t] for t in (EB, KB)]
    lam_scale = sum(lam_t)
    ev, seq = [], 0

    def push(t, prio, item):
        nonlocal seq
        heapq.heappush(ev, (t, prio, seq, item))
        seq += 1

    # 利用希望の発生（ポアソン。発生したマスだけをまとめて時刻に展開する）
    def gen(kind, lam, d, t):
        cnt = rng.poisson(lam)
        jj, hh = np.nonzero(cnt)
        if not len(jj):
            return
        n = cnt[jj, hh]
        x = np.repeat(hh, n) * 60 + rng.random(int(n.sum())) * 60
        for xi, ji in zip(x.tolist(), np.repeat(jj, n).tolist()):
            push(xi, _REQ, (kind, ji, d, t))

    for d in (STATION, AEON):
        for t in (EB, KB):
            gen("out", np.outer(L.share, out_h[:, d] * lam_t[t]), d, t)                 # (k, 24)
            if O.trip_type[d] == "one_way":
                gen("ret", np.outer(L.share, ret_h[:, d] * lam_t[t]), d, t)
    if O.pattern in ("b", "c"):
        push(O.reb_times[0] * 60, _REB, ("reb", "b"))
    if O.pattern == "c":
        push(O.reb_times[1] * 60, _REB, ("reb", "c"))
    rng_km = O.kb_range_km or np.inf

    def rent(p, t, now, km):
        v = S.take(p, t, now, km)
        sub = False
        if v is None and O.substitution:
            v = S.take(p, 1 - t, now, km)
            sub = v is not None
        return v, sub

    def arrive(v, p, now, km):
        S.soc[v] -= km
        if S.vtype[v] == KB and S.soc[v] < O.kb_swap_below * rng_km:
            S.ready_at[v] = now + O.kb_out_hours * 60
            S.soc[v] = rng_km
            st.swaps += 1
        if S.count(p) >= L.caps[p]:
            S.loc[v] = -2
            if O.full_return == "collect" and not S.collect_pending:
                S.collect_pending = True
                push(now + O.collect_delay_min, _REB, ("col",))
            st.return_fail += 1
            st.port_fail[p] += 1
            st.hourly[min(int(now // 60), 23), 3] += 1
        else:
            S.put(v, p)
            st.port_peak[p] = max(st.port_peak[p], S.count(p))
            if p == sp:
                st.station_peak = max(st.station_peak, S.count(sp))

    while ev:
        now, prio, _, item = heapq.heappop(ev)
        kind = item[0]
        if kind == "arr":
            _, v, p, km = item
            arrive(v, p, now, km)
            continue
        if kind == "col":                       # 運営者による回収（満車で返せなかった車両）
            moved = _collect_outside(S)
            S.collect_pending = False
            minutes, trips = _truck(O, moved[EB], moved[KB])
            if minutes > 0:
                st.ops.append((now / 60.0, sum(moved), minutes, trips))
            for p in range(L.k + 2):
                st.port_peak[p] = max(st.port_peak[p], S.count(p))
            continue
        if kind == "reb":
            _rebalance(S, st, now, item[1], out_h, ret_h, lam_scale)
            st.station_peak = max(st.station_peak, S.count(sp))
            for p in range(L.k + 2):
                st.port_peak[p] = max(st.port_peak[p], S.count(p))
            continue
        _, j, d, t = item
        h = min(int(now // 60), 23)
        st.hourly[h, 0] += 1
        st.legs += 1
        alt = L.alt_m[j, d] / vw
        st.walk_only_min += alt
        km = L.ride_m[j, d] / 1000
        if kind == "out":
            st.req_out += 1
            v, sub = rent(j, t, now, km)
            if v is None:
                st.fail_out += 1
                st.hourly[h, 2] += 1
                st.time_min += alt
                if O.trip_type[d] == "round_trip":   # 帰りも徒歩
                    st.legs += 1
                    st.time_min += alt
                    st.walk_only_min += alt
                continue
            st.served_out += 1
            st.hourly[h, 1] += 1
            st.substitutions += sub
            st.rides_by_type[S.vtype[v]] += 1
            ride = O.procedure_min + L.ride_m[j, d] / vr[S.vtype[v]]
            st.time_min += L.walk_m[j] / vw + ride + L.egress_m[d] / vw
            push(now + ride, _ARR, ("arr", v, dport[d], km))
            if O.trip_type[d] == "round_trip":
                w = ret_h[:, d]
                h0 = min(int((now + ride) // 60), 23)
                ww = w[h0:]
                if ww.sum() > 0:
                    hr = h0 + rng.choice(len(ww), p=ww / ww.sum())
                    lo = max(now + ride, hr * 60.0)
                    tr = lo + rng.random() * max(hr * 60.0 + 60 - lo, 0.0)
                    push(tr, _REQ, ("ret", j, d, t))
                else:                                  # 帰りの需要が無い時間帯 → 徒歩で帰る
                    st.legs += 1
                    st.time_min += alt
                    st.walk_only_min += alt
        else:  # 復路の利用希望（目的地ポート → 自宅側ポート j）
            st.req_ret += 1
            v, sub = rent(dport[d], t, now, km)
            if v is None:
                st.fail_ret += 1
                st.hourly[h, 2] += 1
                st.time_min += alt
                continue
            st.served_ret += 1
            st.hourly[h, 1] += 1
            st.substitutions += sub
            st.rides_by_type[S.vtype[v]] += 1
            ride = O.procedure_min + L.ride_m[j, d] / vr[S.vtype[v]]
            st.time_min += L.egress_m[d] / vw + ride + L.walk_m[j] / vw
            push(now + ride, _ARR, ("arr", v, j, km))
    _night_reset(S, st)
    return st, S


# ---------------------------------------------------------------- 集計
def summarize(days: list[DayStats], fleet: Fleet) -> dict:
    g = lambda a: float(sum(getattr(d, a) for d in days))
    req = g("req_out") + g("req_ret")
    served = g("served_out") + g("served_ret")
    n_days = max(len(days), 1)
    ops = [o for d in days for o in d.ops]
    n_veh = max(sum(fleet.n), 1)
    peaks = np.array([d.port_peak for d in days]) if days else np.zeros((1, 1))
    return {
        "日数": len(days),
        "ポート別最大駐車(95%点)": np.ceil(np.quantile(peaks, 0.95, axis=0)).astype(int).tolist(),
        "利用希望": req / n_days,
        "利用": served / n_days,
        "貸出失敗率": 1 - served / req if req > 0 else 0.0,
        "充足率": served / req if req > 0 else 1.0,
        "返却失敗率": g("return_fail") / served if served > 0 else 0.0,
        "代替乗車率": g("substitutions") / served if served > 0 else 0.0,
        "返却失敗/日": g("return_fail") / n_days,
        "平均移動時間(分/片道)": g("time_min") / g("legs") if g("legs") else 0.0,
        "全員徒歩の平均(分/片道)": g("walk_only_min") / g("legs") if g("legs") else 0.0,
        "合計移動時間(人・分/日)": g("time_min") / n_days,
        "駅ポート最大在庫": int(max((d.station_peak for d in days), default=0)),
        "作業回数/日": len(ops) / n_days,
        "移動台数/日": sum(o[1] for o in ops) / n_days,
        "作業時間(分/日)": sum(o[2] for o in ops) / n_days,
        "トラック往復/日": sum(o[3] for o in ops) / n_days,
        "電池交換/日": g("swaps") / n_days,
        "稼働率(回/台/日)": served / n_days / n_veh,
        "_hourly": (np.mean([d.hourly for d in days], axis=0).tolist() if days else []),
    }


def simulate_days(layout, fleet, ops, out_h, ret_h, n_rep=100, seed=0) -> dict:
    """同じ条件の1日を n_rep 回（独立）。最小台数の探索に使う。"""
    # 日 i の乱数は [seed, i] で決める（最小台数の逐次判定と同じ乱数列になる）
    days = [simulate_day(layout, fleet, ops, out_h, ret_h, np.random.default_rng([seed, i]))[0] for i in range(n_rep)]
    return summarize(days, fleet)


# ---------------------------------------------------------------- 1か月
def month_calendar(year: int, month: int, holidays, climate_row: dict, rng) -> pd.DataFrame:
    """日ごとの曜日区分と気象。気象は月の平年出現日数から日ごとに独立に抽選（優先：雨＞小雨＞強風＞猛暑＞寒冷）。"""
    import calendar
    import datetime as dt
    nd = calendar.monthrange(year, month)[1]
    hol = {pd.Timestamp(h).date() for h in holidays}
    p_rain10 = climate_row["rain10"] / nd
    p_light = max(climate_row["rain1"] - climate_row["rain10"], 0) / nd
    rows = []
    for day in range(1, nd + 1):
        date = dt.date(year, month, day)
        dtype = "休日" if date.weekday() >= 5 or date in hol else "平日"
        u = rng.random()
        if u < p_rain10:
            w = "雨"
        elif u < p_rain10 + p_light:
            w = "小雨"
        elif rng.random() < climate_row["wind"] / nd:
            w = "強風"
        elif rng.random() < climate_row["hot"] / nd:
            w = "猛暑"
        elif rng.random() < climate_row["cold"] / nd:
            w = "寒冷"
        else:
            w = "晴れ"
        rows.append((date, dtype, w))
    return pd.DataFrame(rows, columns=["日付", "曜日区分", "気象"])


def simulate_month(layout: Layout, fleet: Fleet, ops_for, demand_for, cal: pd.DataFrame, seed: int = 0,
                   discourage: float = 0.0, state=None, mult0: float = 1.0):
    """ops_for(気象)->Ops、demand_for(曜日区分, 気象)->(out_h, ret_h)。電池・充電状態は日をまたいで引き継ぐ。
    discourage：利用離れの強さ α。前日に借りられなかった割合 f に対し、翌日の利用希望を (1−α·f) 倍にする。"""
    rng = np.random.default_rng(seed)
    days, rows = [], []
    mult = mult0
    for r in cal.itertuples(index=False):
        ops = ops_for(r.気象)
        if mult < 1.0:
            ops = dataclasses.replace(ops, demand_mult=tuple(x * mult for x in ops.demand_mult))
        out_h, ret_h = demand_for(r.曜日区分, r.気象)
        st, state = simulate_day(layout, fleet, ops, out_h, ret_h, rng, state)
        days.append(st)
        one = summarize([st], fleet)
        rows.append({"日付": r.日付, "曜日区分": r.曜日区分, "気象": r.気象, "需要の倍率（利用離れ）": mult, **one})
        req = st.req_out + st.req_ret
        if discourage > 0 and req > 0:
            mult = max(0.0, 1.0 - discourage * (st.fail_out + st.fail_ret) / req)
    daily = pd.DataFrame(rows)
    return summarize(days, fleet), daily, days
