#!/bin/sh
# 使い方: ターミナルで  sh start_mac_linux.sh
cd "$(dirname "$0")" || exit 1
if [ ! -f app.py ]; then echo "[エラー] app.py が見つかりません。zip を展開したフォルダで実行してください。"; exit 1; fi
PY=python3
for v in 3.14 3.13 3.12 3.11; do command -v python$v >/dev/null && { PY=python$v; break; }; done
command -v $PY >/dev/null || { echo "[エラー] python3 が見つかりません。Python 3.11〜3.14 をインストールしてください。"; exit 1; }
if [ ! -x .venv/bin/python ]; then
  echo "初回セットアップ中（数分かかります）..."
  $PY -m venv .venv || { echo "[エラー] 仮想環境を作れませんでした"; exit 1; }
  .venv/bin/python -m pip install --upgrade pip
  .venv/bin/python -m pip install -r requirements.txt || { echo "[エラー] インストール失敗。上のメッセージを確認してください"; rm -rf .venv; exit 1; }
fi
echo "起動中... 表示された Local URL（通常 http://localhost:8501 、開かなければ http://127.0.0.1:8501 ）を開いてください。終了は Ctrl+C"
.venv/bin/python -m streamlit run app.py --browser.gatherUsageStats false
