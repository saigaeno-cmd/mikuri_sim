@echo off
chcp 65001 > nul
cd /d %~dp0
if not exist app.py (
  echo [エラー] app.py が見つかりません。zip を「すべて展開」してから、展開したフォルダ内のこのファイルを実行してください。
  pause & exit /b 1
)
rem 動作確認済みの Python（3.14 / 3.13 / 3.12 / 3.11）のうち新しいものを使う
set PY=
for %%V in (3.14 3.13 3.12 3.11) do (
  if not defined PY ( py -%%V --version >nul 2>nul && set PY=py -%%V )
)
if not defined PY ( where py >nul 2>nul && set PY=py -3 )
if not defined PY set PY=python
%PY% --version >nul 2>nul || (
  echo [エラー] Python が見つかりません。python.org から Python 3.14（3.11〜3.14 なら可）をインストールし、
  echo         「Add Python to PATH」にチェックを入れてください。
  pause & exit /b 1
)
if not exist .venv\Scripts\python.exe (
  echo 初回セットアップ中（数分かかります）...
  %PY% --version
  %PY% -m venv .venv || (echo [エラー] 仮想環境を作れませんでした。 & pause & exit /b 1)
  .venv\Scripts\python -m pip install --upgrade pip
  .venv\Scripts\python -m pip install -r requirements.txt || (
    echo [エラー] ライブラリのインストールに失敗しました。上のメッセージを確認してください。
    echo         社内ネットワークの場合はプロキシ設定が必要なことがあります。
    rmdir /s /q .venv
    pause & exit /b 1
  )
)
echo.
echo 起動中... 「You can now view your Streamlit app」と表示されたら、ブラウザで
echo   http://localhost:8501   （開かない場合は http://127.0.0.1:8501 ）
echo を開いてください。表示された Local URL のポート番号が違う場合はそちらを使います。
echo 終了するには、この画面で Ctrl+C を押してください。この画面を閉じるとアプリも止まります。
echo.
.venv\Scripts\python -m streamlit run app.py --browser.gatherUsageStats false
echo.
echo アプリが停止しました。上にエラーが出ていれば、その内容を確認してください。
pause
