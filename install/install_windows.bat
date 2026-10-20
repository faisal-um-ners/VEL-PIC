@echo off
REM VEL-PIC installer for Windows: creates a private virtual environment in .venv
cd /d "%~dp0\.."
where py >nul 2>nul && (set PY=py -3) || (set PY=python)
%PY% --version || (echo Python 3 not found - install from https://www.python.org/downloads/ & pause & exit /b 1)
%PY% -m venv .venv
call .venv\Scripts\activate.bat
python -m pip install --upgrade pip
python -m pip install -e ".[gui]"
echo Optional MPI: install Microsoft MPI (msmpisetup.exe) then:  pip install mpi4py
python -m picstudio info
python -m picstudio run examples\child_langmuir_1d.pic.json --out "%TEMP%\picstudio_warmup" --steps 20 >nul 2>nul
echo.
echo Done. Start the GUI with install\run_gui.bat
pause
