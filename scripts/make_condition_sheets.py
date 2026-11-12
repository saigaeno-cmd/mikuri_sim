"""条件ごとの結果シート（1条件＝1ページ、②ポート配置評価と同じ並び）を作る。

  python scripts/make_condition_sheets.py                       # data/conditions/sheets.yaml の全条件
  python scripts/make_condition_sheets.py --only 名前の一部
出力：data/output/sheets/<名前>.png、まとめた 結果シート.pptx・結果シート.pdf
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import sheet  # noqa: E402
from src.loaders import load_config, load_network, load_survey, resolve, survey_path  # noqa: E402


def cond_text(c, net) -> str:
    p = c["params"]
    dist = "・".join(f"{k}{v}" for k, v in net.houses["街区"].value_counts(sort=False).items()) if "街区" in net.houses else ""
    mix = p["mix"]
    src = c.get("source", "")
    return (f"条件：住宅からポートまで徒歩{c['A']:g}分以内／短くなる時間{c['X']:g}分以上／住宅側ポート{c['k']}か所／"
            f"目標充足率{p['target']:.0%}／{p['year']}年{p['month']}月の1か月（{src}）×{p['passes']}回／"
            f"{c['n_houses']}戸（{dist}）・対象住宅1日{p['trips']:g}往復・行き先 駅{c['share'][0]:.0%}\n"
            f"希望車種 自転車{mix[0]:.0%}・キックボード{mix[1]:.0%}・4輪{mix[2]:.0%}／"
            f"{'代替乗車あり' if p['substitution'] else '代替乗車なし'}／運用{p['pattern']}／駅ポート{p['station_cap']}区画・"
            f"モビリティステーション{p['aeon_cap']}区画／手続き{p['procedure_sec']:.0f}秒／"
            f"走行速度{'経路別（実測）' if p['route_speeds'] else '一律'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--conditions", default=str(ROOT / "data" / "conditions" / "sheets.yaml"))
    ap.add_argument("--only", default=None)
    a = ap.parse_args()
    cfg = load_config()
    net = load_network(cfg)
    cfg["simulation"]["n_households"] = cfg["simulation"].get("n_households") or len(net.houses)
    sv = load_survey(survey_path(cfg)[0], cfg)
    conds = yaml.safe_load(Path(a.conditions).read_text(encoding="utf-8"))
    out = resolve(cfg["paths"].get("output_dir", "data/output")) / "sheets"
    out.mkdir(parents=True, exist_ok=True)
    roads = sheet._roads(resolve(cfg["paths"]["processed_dir"]))
    pngs = []
    for cd in conds:
        if a.only and a.only not in cd["name"]:
            continue
        kw = {k: v for k, v in cd.items() if k not in ("name", "page", "A", "X", "k")}
        if cd.get("page", "time_opt") != "time_opt":
            raise SystemExit(f"未対応のページ：{cd['page']}")
        c = sheet.compute_time_opt(net, cfg, sv, A=float(cd["A"]), X=float(cd["X"]), k=int(cd["k"]), **kw)
        fig = sheet.sheet_time_opt(c, "① ポート最適化（時間短縮）", cond_text(c, net), roads)
        png = sheet.to_png(fig, dpi=150)
        (out / f"{cd['name']}.png").write_bytes(png)
        pngs.append(png)
        r, ms = c.get("sel"), c.get("ms")
        print(f"{cd['name']}: 対象{c['n_hh']}戸・ポート{c['S']}・"
              + (f"{r['合計']}台（自{r['自転車']}・キ{r['キックボード']}・4輪{r.get('4輪', 0)}）・充足率{ms['充足率']:.1%}・"
                 f"合計移動{ms['合計移動時間(人・分/日)']:.0f}人・分/日" if ms else "達成不可"))
    if pngs:
        (out / "結果シート.pptx").write_bytes(sheet.to_pptx(pngs))
        from PIL import Image
        ims = [Image.open(__import__("io").BytesIO(b)).convert("RGB") for b in pngs]
        ims[0].save(out / "結果シート.pdf", save_all=True, append_images=ims[1:])
        print("保存:", out)


if __name__ == "__main__":
    main()
