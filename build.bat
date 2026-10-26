@echo off
REM Build MACS Qt app into a standalone .exe
REM Requires: pip install pyinstaller PySide6

echo === Building MACS ===

pyinstaller --noconfirm --clean macs.spec

echo === Build complete ===
echo Output: dist\MACS\MACS.exe
pause
