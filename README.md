# 南栗橋BLP街区 モビリティシェア シミュレーター

住宅（既存街区171戸＋新規街区59区画＝230戸）と、南栗橋駅西口ポート・モビリティステーション（イオン）を、
電動アシスト自転車と電動キックボードの混在フリートで結ぶシェアリングの「ポート配置・台数・運用・費用」を検討するツール。

## 手順
```bash
pip install -r requirements.txt        # Python 3.11〜3.14（テスト・報告資料も行う場合は requirements-dev.txt）
python scripts/precompute_network.py   # 道路網・距離行列を作成（data/processed/）。座標・住宅・道路を変えたら再実行
streamlit run app.py                   # 画面（事前計算結果を読むだけ）
```
Windows は `start_windows.bat`、Mac/Linux は `start_mac_linux.sh` でも起動できます（`起動方法.txt`）。

### 最新版への更新（Windows）
最初の1回だけ Git（https://git-scm.com/download/win）を入れ、`git clone https://github.com/saigaeno-cmd/mikuri_sim.git` で取得する。
以後は `更新.bat` をダブルクリックすると最新版になる（続けて起動もできる）。

### 一括更新（全条件の計算 → 感度分析 → 追加検証 → 報告資料）
```bash
python scripts/run_all.py                     # すべて（4コアで1時間前後）
python scripts/run_all.py --skip-precompute   # 道路・住宅を変えていないとき
python scripts/run_all.py --only report       # 報告資料だけ作り直す
python scripts/run_all.py --quick             # 動作確認（回数を減らす）
```
需要・費用などを画面の「設定」で変えたら、これを1回実行すれば結果と報告資料（`data/output/study/報告_モビリティシェア検討.pptx`）がそろいます。

| 工程 | スクリプト | 出力（data/output/study/） |
|---|---|---|
| 全条件 | `run_study.py` | `results.csv`：需要4種×代替乗車2種×リバランス3種×駅ポート容量4種＝96条件×10回 |
| 感度分析 | `run_sensitivity.py` | `sensitivity.csv`：各点10回（報告資料では中央値） |
| 追加検証 | `run_checks.py` | `rebalance_buffer_test.csv`・`full_return_test.csv`・`discourage_test.csv` |
| 報告資料 | `make_report.py` | `報告_モビリティシェア検討.pptx` |

各回で「平日・晴れの1日」を最大60日分シミュレーションして貸出失敗率 20%（充足率80%）・5%・1% を満たす最小の
自転車＋キックボード台数を探し、その台数で1年（12か月）を計算して年間費用を出します。
171戸（既存街区のみ）の旧結果は `data/output/study_171/` にあります。

## ページ構成
| ページ | 内容 |
|---|---|
| 感度分析 | 転換率・ポート数ごとの必要台数とポート位置の感度。配置ファイルの添付・比較 |
| ① ポート最適化 | 転換率（10〜100%、5%刻み）・車種比率・代替乗車・リバランス・駅ポート容量・対象月を指定 → 移動時間最小の住宅側ポート位置と、その1か月（平日・休日・南栗橋の天気）で目標を満たす最小台数。日別の結果、年間費用、a/b/c の比較 |
| ② ポート配置評価 | 地図クリックでポートを追加し、台数と収納台数を指定 → 1か月の充足率・移動時間・返却失敗・費用 |
| 計算結果 | 全条件×10回・感度分析・追加検証の結果を、条件で絞って表・箱ひげ・月別グラフで表示（CSV保存可） |
| 設定 | 需要シナリオ・費用単価・駅ポート容量・運用ルール（代替乗車・満車時の扱い・利用離れ・電池）・街区の追加・道路の追加を画面で変更（`config.yaml` に保存、元の設定は `data/backup/`） |

サイドバーの対象街区（既存・新規）は全ページ共通で、需要は対象街区の戸数で計算します。

## モデル（`src/engine.py`・`src/study.py`）
- **需要**：アンケート（2023、23世帯）の時間帯別トリップ ÷ 23 × 戸数 × 転換率 × 目的地比率。住宅⇔駅は片道、イオンは往復。
- **車種と代替乗車**：利用者ごとに希望車種（自転車60.3%・キックボード39.7%、調査Q16）。基本は (ii)：希望車種が無ければ他の車種に乗り、何もなければ徒歩。
- **イベントシミュレーション**：利用希望をポアソン発生させ、車両1台ずつの位置・電池を追跡。
- **満車時**：到着ポートが満車なら「返却失敗」。`rebalance.full_return` で扱いを選ぶ：outside＝区画外に置き次の作業で回収（基本）／collect＝運営者が回収に出る（作業時間・軽トラック費を計上）。
- **利用離れ**：前日に借りられなかった割合 f に対し、翌日の利用希望を (1−α·f) 倍（`behavior.discourage_alpha`、仮定0.5）。1か月・1年の計算に効き、台数の探索（平日・晴れの1日）には効かない。
- **キックボード電池**：航続30km、残量20%未満で交換・4時間使えない。交換1回10分の作業費を計上（仮定）。
- **リバランス**：a 夜に初期配置へ戻す／b a＋9:30に駅の余剰を住宅側へ／c b＋15:30に住宅側と駅をそろえる。
- **ポート位置**：徒歩＋走行（車種比率で加重）＋施設側徒歩の合計を最小にする p-median。
- **1年**：12か月を連続で計算（曜日・祝日・月ごとの平年気象を抽選）。年間費用・利用回数・稼働率はこの1年分から。
- **最小台数の探索**：同じ乱数列で台数の組み合わせを比べ、目標との大小が統計的にはっきりした組み合わせは途中の日数で打ち切る（逐次判定）。

前提値と出典は `docs/前提と出典.md`。値はすべて `config.yaml` にあり、画面「設定」から変えられます。

## データの差し替え手順
| 何を | 手順 |
|---|---|
| 需要（転換率）を確定値に | 画面「設定」→需要シナリオで値と名前を変更 → `python scripts/run_all.py --skip-precompute` |
| アンケートを更新 | `data/templates/survey_template.xlsx` に記入 → CSVで保存 → `config.yaml` の `paths.survey_csv` に指定 → run_all |
| 実証の利用記録を反映 | 記録を `data/templates/usage_log_template.csv` の形でCSVに → `python scripts/import_usage_log.py 記録.csv [--refusals 断られた記録.csv]` → `data/output/usage/calibration.csv` の推定転換率・車種比率・速度を「設定」に入れる → run_all |
| 駅ポートの区画案 | 画面「設定」→駅ポートで容量の案を追加・変更（今は 18・25・32・40〈仮〉台） → run_all |
| 南栗橋の実際の天気 | `python scripts/fetch_weather.py`（インターネットに接続できるPCで実行）→ `data/raw/weather_minamikurihashi.csv`。気象庁の久喜アメダスのCSVを同じ名前で置いてもよい。①②は選んだ月の実際の天気で計算（無い月は平年値から抽選） |
| 気象の倍率・出現日数 | 気象庁「過去の気象データ検索」で久喜（アメダス）または熊谷の平年値を確認し、`config.yaml` の `climate`（月別日数）・`weather`（倍率）を置換 → run_all |
| 車両の仕様 | 画面「設定」→運用・行動で電池の航続距離・交換の目安・停止時間を変更 |
| 費用の単価 | 画面「設定」→費用単価 |
| 街区・道路の追加 | 画面「設定」→街区・道路でファイルを添付（前処理は自動）。または `config.yaml` の `paths.houses` に追加して precompute |

## データ（data/raw/）
| ファイル | 内容 |
|---|---|
| `houses.gpkg` | 既存街区の住宅ポイント 171戸 |
| `houses_new_lots_59.geojson` | 新規街区の区画ポイント 59区画（`lot_no`＝区画番号） |
| `new_roads_patch.osm` / `new_roads.geojson` | 新規街区の区画道路（W6m）7本・歩行者専用通路（W2m）25本。計画図をデジタイズ |
| `map_merged.osm` | map.osm＋新規街区の道路（`python scripts/merge_osm.py data/raw/map.osm data/raw/new_roads_patch.osm data/raw/map_merged.osm`）。precompute はこれを使う |
| `map.osm` | OSM抽出（南栗橋駅〜街区〜イオン）。道路網の元 |
| `survey.xlsx` | 2023アンケート集計表（曜日×5〜22時台×交通手段） |
| `ride_logs/` | 走行記録（fit・スマホ位置ログ）と `speed_summary.csv`（自転車12.6・キックボード14.0 km/h） |

ひな形（data/templates/）：`survey_template.xlsx`（調査票）、`usage_log_template.csv`・`refusals_template.csv`（実証の利用記録）。

## 構成
| ファイル | 役割 |
|---|---|
| `app.py` | ページ切替（`views/` の5ページ） |
| `src/engine.py` | イベントシミュレーション（車両・電池・返却失敗・回収・リバランス・利用離れ・1か月／1年） |
| `src/plan.py` | 配置（Layout）・運用条件（Ops）・需要の組み立て |
| `src/study.py` | 最小台数の探索（逐次判定）・1か月／1年の評価・年間費用 |
| `src/demand.py` / `location.py` / `weather.py` / `loaders.py` | 需要・p-median・気象倍率・読込 |
| `src/ui.py` / `src/viz.py` | サイドバー・グラフと地図 |
| `scripts/` | 前処理・一括更新・全条件・感度分析・追加検証・報告資料・利用記録の取り込み・道路の統合 |
| `docs/report_template.pptx` | 報告資料のテンプレート |
| `tests/` | `python -m pytest`（GitHub に記録するたびに自動実行：`.github/workflows/tests.yml`） |
