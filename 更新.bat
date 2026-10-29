@echo off
chcp 65001 > nul
cd /d "%~dp0"
echo 南栗橋BLP モビリティシェア シミュレーター：最新版に更新します
echo.
where git >nul 2>nul || (
  echo [エラー] Git が見つかりません。https://git-scm.com/download/win から入れてください。
  pause & exit /b 1
)
if not exist ".git" (
  echo [エラー] このフォルダは git で取得されていません。
  echo   次の手順で取得し直してください：
  echo     cd C:\
  echo     git clone https://github.com/saigaeno-cmd/mikuri_sim.git
  pause & exit /b 1
)
git pull --ff-only
if errorlevel 1 (
  echo.
  echo [エラー] 更新できませんでした。
  echo   画面の「設定」などで変えたファイルがあると、更新と食い違うことがあります。
  echo   変えた内容が不要なら、このフォルダで次を実行してから、もう一度このファイルを実行してください：
  echo     git checkout -- .
  pause & exit /b 1
)
echo.
echo 更新しました。
choice /C YN /M "続けてシミュレーターを起動しますか"
if errorlevel 2 exit /b 0
call "%~dp0start_windows.bat"
