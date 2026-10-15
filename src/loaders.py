"""CSV・事前計算結果の読込と検証。"""
from __future__ import annotations

import dataclasses
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent.parent


def load_config(path: Path | str = ROOT / "config.yaml") -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def houses_files(cfg: dict) -> tuple[list[tuple[str, Path]], bool]:
    """([(街区名, 住宅ファイル)], サンプルか)。paths.houses は1ファイル（文字列）か街区のリスト。
    raw に無ければ sample にフォールバック。"""
    h = cfg["paths"]["houses"]
    items = [("街区", h)] if isinstance(h, str) else [(d["name"], d["file"]) for d in h]
    files = [(n, resolve(f)) for n, f in items]
    if all(p.exists() for _, p in files):
        return files, False
    missing = [str(p) for _, p in files if not p.exists()]
    if len(missing) < len(files):
        raise FileNotFoundError("住宅ファイルがありません：" + "、".join(missing))
    return [("サンプル", resolve(cfg["paths"]["sample_dir"]) / "houses.csv")], True


def houses_path(cfg: dict) -> tuple[list[Path], bool]:
    """([住宅ファイル], サンプルか)。"""
    files, sample = houses_files(cfg)
    return [p for _, p in files], sample


def survey_path(cfg: dict) -> tuple[Path, bool]:
    """(アンケートCSV, サンプルか)。raw に無ければ sample にフォールバック。"""
    p = resolve(cfg["paths"]["survey_csv"])
    return (p, False) if p.exists() else (resolve(cfg["paths"]["sample_dir"]) / "survey.csv", True)


def file_hash(*paths: Path) -> str:
    h = hashlib.sha256()
    for p in paths:
        h.update(Path(p).read_bytes())
    return h.hexdigest()[:16]


def _read_csv(path: Path) -> pd.DataFrame:
    for enc in ("utf-8-sig", "cp932"):
        try:
            return pd.read_csv(path, encoding=enc)
        except UnicodeDecodeError:
            continue
    raise ValueError(f"{path} の文字コードを判別できません（UTF-8 / Shift_JIS に対応）")


def _rename(df: pd.DataFrame, aliases: dict[str, list[str]], required: list[str], name: str) -> pd.DataFrame:
    lower = {c.strip().lower(): c for c in df.columns}
    mapping = {}
    for key, cands in aliases.items():
        for cand in cands:
            if cand.lower() in lower:
                mapping[lower[cand.lower()]] = key
                break
    df = df.rename(columns=mapping)
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"{name}: 必須列 {missing} が見つかりません。実際の列: {list(df.columns)}。"
            " config.yaml の columns に列名を追加してください。"
        )
    return df


def load_houses(path: Path, cfg: dict) -> pd.DataFrame:
    """CSV（lat, lon 列）または GeoPackage / GeoJSON（点ジオメトリ）を読む。"""
    if Path(path).suffix.lower() in (".gpkg", ".geojson", ".json", ".shp"):
        import geopandas as gpd

        g = gpd.read_file(path)
        if g.crs is not None and g.crs.to_epsg() != 4326:
            g = g.to_crs(4326)
        g = g[g.geometry.notna() & ~g.geometry.is_empty]
        if not (g.geometry.geom_type == "Point").all():
            g = g.set_geometry(g.geometry.representative_point())
        df = pd.DataFrame({"lat": g.geometry.y.to_numpy(), "lon": g.geometry.x.to_numpy()})
        idc = next((c for c in cfg["columns"]["houses"]["house_id"] if c in g.columns), None)
        ids = g[idc] if idc else None
        if ids is not None and ids.notna().all() and ids.is_unique:
            df["house_id"] = ids.to_numpy()
    else:
        df = _rename(_read_csv(path), cfg["columns"]["houses"], ["lat", "lon"], "住宅座標CSV")
    if "house_id" not in df.columns:
        df["house_id"] = np.arange(1, len(df) + 1)
    df = df[["house_id", "lat", "lon"]].dropna(subset=["lat", "lon"]).reset_index(drop=True)
    return df


def load_houses_all(cfg: dict) -> pd.DataFrame:
    """全街区の住宅。列：house_id（通し番号）, lat, lon, 街区, 区画（元ファイルの番号）。"""
    files, _ = houses_files(cfg)
    parts = []
    for name, p in files:
        h = load_houses(p, cfg).rename(columns={"house_id": "区画"})
        h["街区"] = name
        parts.append(h)
    df = pd.concat(parts, ignore_index=True)
    df.insert(0, "house_id", np.arange(1, len(df) + 1))
    return df[["house_id", "lat", "lon", "街区", "区画"]]


def _parse_hour(v) -> int:
    """'7', '7時', '7:30', '7-8時', '07:00～08:00' などから開始時刻（時）を取り出す。"""
    if pd.isna(v):
        raise ValueError("時間帯が空欄の行があります")
    if isinstance(v, (int, np.integer, float, np.floating)):
        return int(v)
    s = str(v).translate(str.maketrans("０１２３４５６７８９：", "0123456789:"))
    digits = ""
    for ch in s:
        if ch.isdigit():
            digits += ch
        elif digits:
            break
    if not digits:
        raise ValueError(f"時間帯 '{v}' を解釈できません")
    return int(digits) % 24


WEEKDAYS = ["月曜日", "火曜日", "水曜日", "木曜日", "金曜日"]
HOLIDAYS = ["土曜日", "日曜日"]


def load_survey_xlsx(path: Path, cfg: dict) -> pd.DataFrame:
    """2023アンケート集計表（シート「トリップ数」）を縦持ちにする。

    表の形：1行目=時間帯（5時台〜22時台、各4列を結合）、2行目=交通手段コード A〜D、
    3〜9行目=月曜日〜日曜日のトリップ数（回答者合計）。凡例（A=徒歩 等）は表の右側。
    午前（survey_xlsx.return_from_hour 未満）を往路、それ以降を復路とみなす。
    戻り値の列: day(平日/休日), hour, purpose(交通手段名), direction, trips(1日あたり、回答者合計)
    """
    import openpyxl

    sx = cfg["survey_xlsx"]
    ws = openpyxl.load_workbook(path, data_only=True)[sx["sheet"]]
    rows = [list(r) for r in ws.iter_rows(values_only=True)]
    header, codes = rows[0], rows[1]
    # 結合セルの時間帯ラベルを右に伝播
    hours, cur = [], None
    for v in header:
        if v is not None and "時" in str(v) and any(ch.isdigit() for ch in str(v)):
            cur = _parse_hour(v)
        hours.append(cur)
    # 凡例（コード → 交通手段名）。見つからなければ config の値
    legend = dict(sx.get("mode_labels", {}))
    last_hour_col = max(i for i, h in enumerate(hours) if h is not None and codes[i] is not None)
    for r in rows:
        for i in range(last_hour_col + 1, len(r) - 1):
            if r[i] in ("A", "B", "C", "D") and isinstance(r[i + 1], str):
                legend[r[i]] = r[i + 1].strip()
    recs = []
    for r in rows[2:]:
        day = str(r[0]).strip() if r[0] is not None else ""
        if day not in WEEKDAYS + HOLIDAYS:
            continue
        for i, v in enumerate(r):
            code = codes[i] if i < len(codes) else None
            if i > last_hour_col or hours[i] is None or code not in legend or v is None:
                continue
            recs.append((day, hours[i], legend[code], float(v)))
    df = pd.DataFrame(recs, columns=["曜日", "hour", "purpose", "trips"])
    if df.empty:
        raise ValueError(f"{path}: シート「{sx['sheet']}」から曜日別のトリップ数を読み取れませんでした")
    df["day"] = np.where(df["曜日"].isin(WEEKDAYS), "平日", "休日")
    n_days = df.groupby("day")["曜日"].nunique()
    out = df.groupby(["day", "hour", "purpose"], as_index=False)["trips"].sum()
    out["trips"] /= out["day"].map(n_days)
    out["direction"] = np.where(out["hour"] < sx["return_from_hour"], "outbound", "return")
    out["respondent_id"] = 0
    return out[["day", "respondent_id", "hour", "purpose", "direction", "trips"]]


def load_survey(path: Path, cfg: dict) -> pd.DataFrame:
    """縦持ち（1行=回答×時間帯×目的）または横持ち（時間帯が列）に対応し、縦持ちに揃える。"""
    if Path(path).suffix.lower() in (".xlsx", ".xlsm"):
        return load_survey_xlsx(path, cfg)
    raw = _read_csv(path)
    aliases = cfg["columns"]["survey"]
    probe = _rename(raw.copy(), aliases, [], "アンケートCSV")
    if "hour" not in probe.columns:
        # 横持ち：時間帯らしい列（数字を含む列）を縦に展開
        id_cols = [c for c in probe.columns if c in ("respondent_id", "purpose", "destination", "direction")]
        hour_cols = [c for c in probe.columns if c not in id_cols and any(ch.isdigit() for ch in str(c))]
        if not hour_cols:
            raise ValueError(f"アンケートCSV: 時間帯の列が見つかりません。実際の列: {list(raw.columns)}")
        probe = probe.melt(id_vars=id_cols, value_vars=hour_cols, var_name="hour", value_name="trips")
    df = _rename(probe, aliases, ["hour", "trips"], "アンケートCSV")
    if "respondent_id" not in df.columns:
        df["respondent_id"] = np.arange(len(df))
    if "purpose" not in df.columns:
        df["purpose"] = "その他"
    df["hour"] = df["hour"].map(_parse_hour)
    df["trips"] = pd.to_numeric(df["trips"], errors="coerce").fillna(0.0)
    df["purpose"] = df["purpose"].fillna("その他").astype(str).str.strip()
    if "destination" in df.columns:
        rev = {a: k for k, al in cfg["destination_aliases"].items() for a in al}
        df["destination"] = df["destination"].map(lambda x: rev.get(str(x).strip(), "other"))
    if "direction" in df.columns:
        df["direction"] = df["direction"].map(
            lambda x: "return" if str(x).strip() in ("復路", "帰り", "return", "inbound") else "outbound"
        )
    keep = [c for c in ("respondent_id", "hour", "purpose", "destination", "direction", "trips") if c in df.columns]
    return df[keep].reset_index(drop=True)


@dataclass
class Network:
    """事前計算結果。距離はすべて m。"""
    W: np.ndarray            # (n_houses, n_cand) 住宅→候補 徒歩
    R: np.ndarray            # (n_cand, n_dest) 候補→目的地ポート 走行
    E: np.ndarray            # (n_dest,) 目的地ポート→施設入口 徒歩
    dest_keys: list[str]
    houses: pd.DataFrame     # house_id, lat, lon
    candidates: pd.DataFrame # cand_id, lat, lon
    meta: dict
    Wn: np.ndarray | None = None        # (n_houses, n_nodes) 住宅→全徒歩ノード 徒歩（-1=到達不能）
    Rn: np.ndarray | None = None        # (n_nodes, n_dest) 全徒歩ノード→目的地ポート 走行
    nodes: pd.DataFrame | None = None   # lat, lon, usable

    @property
    def districts(self) -> list[str]:
        return list(dict.fromkeys(self.houses["街区"])) if "街区" in self.houses else []

    def subset(self, districts) -> "Network":
        """指定した街区の住宅だけにした Network（ポート候補・道路網はそのまま）。"""
        if not districts or "街区" not in self.houses or set(districts) >= set(self.districts):
            return self
        m = self.houses["街区"].isin(list(districts)).to_numpy()
        if not m.any():
            raise ValueError(f"街区 {districts} の住宅がありません（{self.districts}）")
        return dataclasses.replace(self, W=self.W[m], houses=self.houses[m].reset_index(drop=True),
                                   Wn=None if self.Wn is None else self.Wn[m])


def load_network(cfg: dict) -> Network:
    d = resolve(cfg["paths"]["processed_dir"])
    mpath = d / "matrices.npz"
    if not mpath.exists():
        raise FileNotFoundError(
            f"{mpath} がありません。先に `python scripts/precompute_network.py` を実行してください。"
        )
    m = np.load(mpath, allow_pickle=False)
    meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
    return Network(
        W=m["W"], R=m["R"], E=m["E"], dest_keys=list(meta["dest_keys"]),
        houses=pd.read_csv(d / "houses_snapped.csv"),
        candidates=pd.read_csv(d / "candidates.csv"),
        meta=meta,
        Wn=m["Wn"] if "Wn" in m else None,
        Rn=m["Rn"] if "Rn" in m else None,
        nodes=pd.read_csv(d / "walk_nodes.csv") if (d / "walk_nodes.csv").exists() else None,
    )
