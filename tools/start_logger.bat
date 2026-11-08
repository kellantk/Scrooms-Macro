@echo off
cd /d "%~dp0"
python -c "import pynput" 2>nul || python -m pip install pynput
python input_logger.py
pause
