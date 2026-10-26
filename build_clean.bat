@echo off
REM Create a clean venv and build — avoids all conda/PyQt5 conflicts
echo === Creating clean build environment ===

cd /d "%~dp0"

REM Use system Python or conda base to create a vanilla venv
python -m venv .buildenv
call .buildenv\Scripts\activate.bat

echo === Installing minimal dependencies ===
pip install --upgrade pip
pip install PySide6 pyserial numpy scipy nibabel pydicom scikit-image matplotlib pyinstaller

echo === Building MACS ===
pyinstaller --noconfirm --clean ^
    --name MACS ^
    --console ^
    --add-data "resources;resources" ^
    --add-data "core;core" ^
    --hidden-import serial ^
    --hidden-import serial.tools.list_ports ^
    --hidden-import PySide6.QtCore ^
    --hidden-import PySide6.QtGui ^
    --hidden-import PySide6.QtWidgets ^
    --exclude-module PyQt5 ^
    --exclude-module PyQt6 ^
    --exclude-module torch ^
    --exclude-module tkinter ^
    --exclude-module IPython ^
    --noupx ^
    main.py

echo === Done ===
echo Output: dist\MACS\MACS.exe
call deactivate
pause
