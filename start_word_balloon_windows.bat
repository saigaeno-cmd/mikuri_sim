@echo off
chcp 65001 > nul
cd /d %~dp0
rem 気になるワード：サーバーを起動し、リンク一覧（管理者画面・投影画面・スマホ回答）をブラウザで開く
where py > nul 2>&1
if %errorlevel%==0 (
  py -3 word_balloon\server.py --open %*
  goto end
)
where python > nul 2>&1
if %errorlevel%==0 (
  python word_balloon\server.py --open %*
  goto end
)
echo Python が見つかりません。https://www.python.org/downloads/ からインストールしてください。
echo （インストール時に「Add python.exe to PATH」にチェックを入れてください）
:end
pause
