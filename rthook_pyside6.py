"""Runtime hook: ensure PySide6 DLLs are findable before import."""
import os
import sys

# Remove conflicting QT_API env var
os.environ.pop('QT_API', None)
os.environ['QT_API'] = 'PySide6'

# Add PySide6 directory to DLL search path
if sys.platform == 'win32':
    base = sys._MEIPASS if getattr(sys, 'frozen', False) else os.path.dirname(__file__)
    pyside6_dir = os.path.join(base, 'PySide6')
    if os.path.isdir(pyside6_dir):
        os.add_dll_directory(pyside6_dir)
        os.environ['PATH'] = pyside6_dir + os.pathsep + os.environ.get('PATH', '')
    # Also add plugins
    plugins_dir = os.path.join(pyside6_dir, 'plugins')
    if os.path.isdir(plugins_dir):
        os.environ['QT_PLUGIN_PATH'] = plugins_dir
