"""南栗橋地域の実際の日別天気（data/raw/weather_minamikurihashi.csv）を読み、日ごとの気象区分にする。

ファイルの形式（どれでも可）：
- 正規化した CSV：日付, 降水量mm, 最高気温, 最低気温, 最大風速（m/s）
- Open-Meteo（scripts/fetch_weather.py が保存する形式）：time, precipitation_sum, temperature_2m_max, ...
- 気象庁「過去の気象データ・ダウンロード」の CSV（久喜アメダスなど。Shift_JIS・見出し複数行）
ファイルが無い月は、平年値（config の climate）から日ごとに抽選する（従来の方法）。
"""
from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pandas as pd

from .loaders import resolve

COLS = ["日付", "降水量mm", "最高気温", "最低気温", "最大風速"]


def weather_path(cfg) -> Path:
    return resolve(cfg["paths"].get("weather_daily") or "data/raw/weather_minamikurihashi.csv")


def _read_text(p: Path) -> str:
    raw = p.read_bytes()
    for enc in ("utf-8-sig", "cp932"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    raise ValueError(f"{p} の文字コードを判別できません")


def _from_jma(text: str) -> pd.DataFrame:
    """気象庁ダウンロード CSV：先頭に地点名などの行があり、見出しは「年月日」で始まる行。"""
    lines = text.splitlines()
    i = next(n for n, line in enumerate(lines) if line.startswith("年月日"))
    head = lines[i].split(",")
    body = [line.split(",") for line in lines[i + 1:] if line[:1].isdigit()]
    df = pd.DataFrame(body)

    def col(key):
        for j, h in enumerate(head):
            if key in h and j < df.shape[1]:
                return pd.to_numeric(df[j], errors="coerce")
        return pd.Series(np.nan, index=df.index)

    return pd.DataFrame({"日付": pd.to_datetime(df[0]), "降水量mm": col("降水量の合計"), "最高気温": col("最高気温"),
                         "最低気温": col("最低気温"), "最大風速": col("最大風速")})


def load_daily_weather(cfg) -> pd.DataFrame | None:
    p = weather_path(cfg)
    if not p.exists():
        return None
    text = _read_text(p)
    if text.lstrip().startswith(("ダウンロードした時刻", "年月日")) or "\n年月日" in text[:2000]:
        df = _from_jma(text)
    else:
        d = pd.read_csv(io.StringIO(text))
        if "time" in d:                                   # Open-Meteo
            d = d.rename(columns={"time": "日付"})
            ren = {}
            for c in d.columns:
                if c.startswith("precipitation_sum"):
                    ren[c] = "降水量mm"
                elif c.startswith("temperature_2m_max"):
                    ren[c] = "最高気温"
                elif c.startswith("temperature_2m_min"):
                    ren[c] = "最低気温"
                elif c.startswith("wind_speed_10m_max"):
                    ren[c] = "最大風速"
            d = d.rename(columns=ren)
        df = d
        df["日付"] = pd.to_datetime(df["日付"])
    for c in COLS[1:]:
        if c not in df:
            df[c] = np.nan
    return df[COLS].dropna(subset=["日付"]).sort_values("日付").reset_index(drop=True)


def classify(row, th: dict) -> str:
    """日別の値 → 気象区分（config の weather のキー）。優先：雨＞小雨＞強風＞猛暑＞寒冷＞晴れ。"""
    pr, tx, tn, ws = row["降水量mm"], row["最高気温"], row["最低気温"], row["最大風速"]
    if pd.notna(pr) and pr >= th.get("rain_heavy_mm", 10):
        return "雨"
    if pd.notna(pr) and pr >= th.get("rain_mm", 1):
        return "小雨"
    if pd.notna(ws) and ws >= th.get("wind_ms", 10):
        return "強風"
    if pd.notna(tx) and tx >= th.get("hot_c", 35):
        return "猛暑"
    if pd.notna(tn) and tn < th.get("cold_c", 0):
        return "寒冷"
    return "晴れ"


def available_months(cfg) -> list[tuple[int, int]]:
    """日別データがそろっている（年, 月）。新しい順。"""
    d = load_daily_weather(cfg)
    if d is None or d.empty:
        return []
    g = d.groupby([d["日付"].dt.year, d["日付"].dt.month]).size()
    full = [(int(y), int(m)) for (y, m), n in g.items() if n >= pd.Period(f"{y}-{m:02d}").days_in_month]
    return sorted(full, reverse=True)


def holidays_for(cfg, year: int) -> list:
    return list(cfg["month"].get(f"holidays_{year}", []))


def actual_calendar(cfg, year: int, month: int) -> pd.DataFrame | None:
    """実際の天気による1か月の暦（日付・曜日区分・気象・各値）。データが無ければ None。"""
    d = load_daily_weather(cfg)
    if d is None:
        return None
    m = d[(d["日付"].dt.year == year) & (d["日付"].dt.month == month)].copy()
    if len(m) < pd.Period(f"{year}-{month:02d}").days_in_month:
        return None
    th = cfg.get("weather_thresholds", {})
    hol = {pd.Timestamp(h).date() for h in holidays_for(cfg, year)}
    m["日付"] = m["日付"].dt.date
    m["曜日区分"] = [("休日" if dt.weekday() >= 5 or dt in hol else "平日") for dt in m["日付"]]
    m["気象"] = m.apply(lambda r: classify(r, th), axis=1)
    return m[["日付", "曜日区分", "気象", "降水量mm", "最高気温", "最低気温", "最大風速"]].reset_index(drop=True)
