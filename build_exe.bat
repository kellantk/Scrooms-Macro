@echo off
rem Update PATCH_NOTES.txt before building a new version!
rem Builds App\ScroomMacro.exe (+ App\_internal) and packs
rem releases\ScroomMacro_v<VERSION>.zip. Old release zips are never overwritten.
rem Folder build (no self-extracting single exe), no UPX, with version info:
rem all of these make antivirus false positives less likely.
cd /d "%~dp0"
python -m pip install --quiet numpy mss pyinstaller
for /f %%v in ('python make_version.py') do set VER=%%v
if "%VER%"=="" goto :fail
if exist releases\ScroomMacro_v%VER%.zip (
  echo releases\ScroomMacro_v%VER%.zip already exists - bump VERSION in ScroomMacro.pyw first.
  goto :end
)
python -m PyInstaller --noconfirm --onedir --windowed --noupx --name ScroomMacro ^
  --icon scroom.ico --add-data "scroom.ico;." --version-file version_info.txt ^
  ScroomMacro.pyw || goto :fail
rem keep the user's settings/log when replacing the app folder
if exist App\settings.json copy /y App\settings.json settings.keep >nul
if exist App rmdir /s /q App
if exist App goto :locked
move dist\ScroomMacro App >nul
if exist settings.keep move /y settings.keep App\settings.json >nul
rmdir /s /q build dist 2>nul & del ScroomMacro.spec 2>nul
if not exist releases mkdir releases
rem leave your own settings/log out of the shareable zip
if exist App\settings.json move /y App\settings.json settings.keep >nul
if exist App\macro_log.txt move /y App\macro_log.txt macro_log.keep >nul
if exist App\debug move App\debug debug.keep >nul
if exist App\bug_report_*.zip move /y App\bug_report_*.zip . >nul
powershell -NoProfile -Command "Compress-Archive -Path App,README.txt,PATCH_NOTES.txt,ScroomMacro.pyw,bot.py,detect.py,hotkeys.py,strings.py,keys.py,percussion.py,countin.py,notify.py,updater.py,startup_check.py,make_version.py,scroom.ico,build_exe.bat -DestinationPath releases\ScroomMacro_v%VER%.zip"
if exist settings.keep move /y settings.keep App\settings.json >nul
if exist macro_log.keep move /y macro_log.keep App\macro_log.txt >nul
if exist debug.keep move debug.keep App\debug >nul
echo.
echo Done: releases\ScroomMacro_v%VER%.zip   (run App\ScroomMacro.exe)
goto :end
:locked
echo Close ScroomMacro first - the App folder is in use.
goto :end
:fail
echo Build failed.
:end
pause
