"""南栗橋地域の実際の日別天気を取得して data/raw/weather_minamikurihashi.csv に保存する（Open-Meteo の過去の気象データ）。

使い方: python scripts/fetch_weather.py [--start 2024-01-01] [--end 2025-12-31]
インターネットに接続できるPCで実行する。気象庁の久喜アメダスのCSV（過去の気象データ・ダウンロード）を
同じファイル名で置いてもよい（src/weather_data.py がどちらも読める）。
"""
from __future__ import annotations

import argparse
import datetime as dt
import sys
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.loaders import load_config  # noqa: E402
from src.weather_data import weather_path  # noqa: E402


def main():
    today = dt.date.today()
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default=f"{today.year - 2}-01-01")
    ap.add_argument("--end", default=str(today - dt.timedelta(days=7)))
    a = ap.parse_args()
    cfg = load_config()
    p = cfg["destinations"]["station"]["port"]            # 南栗橋駅の付近
    q = urllib.parse.urlencode({
        "latitude": p["lat"], "longitude": p["lon"], "start_date": a.start, "end_date": a.end,
        "daily": "precipitation_sum,temperature_2m_max,temperature_2m_min,wind_speed_10m_max",
        "timezone": "Asia/Tokyo", "wind_speed_unit": "ms", "format": "csv"})
    url = f"https://archive-api.open-meteo.com/v1/archive?{q}"
    with urllib.request.urlopen(url, timeout=60) as r:
        text = r.read().decode("utf-8")
    lines = text.splitlines()
    i = next(n for n, line in enumerate(lines) if line.startswith("time"))   # 先頭の地点情報を飛ばす
    out = weather_path(cfg)
    out.write_text("\n".join(lines[i:]) + "\n", encoding="utf-8")
    print(f"保存: {out}（{len(lines) - i - 1}日分、{a.start}〜{a.end}）")


if __name__ == "__main__":
    main()
