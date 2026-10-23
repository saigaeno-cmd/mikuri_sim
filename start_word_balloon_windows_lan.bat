@echo off
chcp 65001 > nul
cd /d %~dp0
rem 気になるワード：インターネットに公開せず、同じWi-Fiのスマホだけで使う場合
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
