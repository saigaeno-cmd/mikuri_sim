"""気象条件による需要・速度の補正（倍率は config.yaml の weather。仮定値）。"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Weather:
    name: str
    demand: float      # 需要倍率（転換率に掛ける）
    walk: float        # 徒歩速度の倍率
    ride: float        # 走行速度の倍率

    def apply(self, cfg: dict, vehicle: str) -> tuple[float, float]:
        """(徒歩速度 km/h, 走行速度 km/h)"""
        return cfg["walk_speed_kmh"] * self.walk, cfg["vehicles"][vehicle]["speed_kmh"] * self.ride


def get_weather(cfg: dict, name: str, vehicle: str) -> Weather:
    w = cfg["weather"][name]
    pick = (lambda v: float(v[vehicle]) if isinstance(v, dict) else float(v))
    return Weather(name, pick(w["demand"]), float(w["walk"]), pick(w["ride"]))
