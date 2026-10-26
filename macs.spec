# -*- mode: python ; coding: utf-8 -*-
"""MACS — Multi-channel Active Coil Shimming.

The C++ BET engine (_cpp_bet + ONNX Runtime) and its model are collected from
cpp_bet/. The model lands in _internal/model/ as a plain file, so a distilled
student can be swapped in later by replacing that file and its *_config.json —
no rebuild, no repackage. cpp_bet_bridge picks up whatever .onnx it finds there.
"""
import atexit
import os
import sys
from pathlib import Path

_spec_dir = Path(os.path.abspath(SPECPATH))

# Bake the build date and git SHA into core/version.py for the duration of the
# build, then put the file back so packaging leaves no working-tree change.
sys.path.insert(0, str(_spec_dir))
from scripts.stamp_version import stamp as _stamp, restore as _restore
_stamp_info = _stamp()
atexit.register(_restore)
print(f"[macs.spec] stamping {_stamp_info}")
# Public repo: prebuilt module and model sit in ./cpp_bet. Dev tree: built
# from source in ../cpp_bet/build/Release.
_cpp_bet = _spec_dir / "cpp_bet"
if not _cpp_bet.is_dir():
    _cpp_bet = _spec_dir.parent / "cpp_bet"
_cpp_build = _cpp_bet / "build" / "Release"
if not _cpp_build.is_dir():
    _cpp_build = _cpp_bet

# The compiled module and the ONNX Runtime DLLs it links against.
from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs

_binaries = []
# CVXPY reaches its solvers through plugin modules that nothing imports by
# name, and both solvers are compiled extensions, so PyInstaller's static
# analysis finds neither the Python side nor the native side on its own.
for _pkg in ('clarabel', 'scs', 'cvxpy'):
    try:
        _binaries += collect_dynamic_libs(_pkg)
    except Exception as _e:
        print(f"[macs.spec] no dynamic libs for {_pkg}: {_e}")
_solver_datas = []
for _pkg in ('cvxpy', 'clarabel', 'scs'):
    try:
        _solver_datas += collect_data_files(_pkg)
    except Exception as _e:
        print(f"[macs.spec] no data files for {_pkg}: {_e}")
for _pat in ("_cpp_bet*.pyd", "onnxruntime*.dll", "zlib*.dll"):
    for _f in _cpp_build.glob(_pat):
        _binaries.append((str(_f), "."))

# Which BET model to ship. cpp_bet/model/ holds several; exactly one goes into
# the bundle, because cpp_bet_bridge picks whatever .onnx it finds there and
# shipping two would make the choice implicit.
#
#   bet_micro    distilled student — 0.06 MB, ~1.2 s/volume, Dice 0.983 vs HD-BET
#   hdbet_light  the HD-BET teacher — 117 MB, ~100 s/volume, the reference
#
# See cpp_bet/training/BET_MODEL_BENCHMARK.md for the measurements.
BET_MODEL = "bet_micro"

_cfg_name = ("model_config.json" if BET_MODEL == "hdbet_light"
             else f"{BET_MODEL}_config.json")
_model_files = []
for _name in (f"{BET_MODEL}.onnx", _cfg_name):
    _p = _cpp_bet / "model" / _name
    if _p.is_file():
        _model_files.append((str(_p), "model"))
    else:
        raise SystemExit(f"BET model file missing: {_p}")

if not _binaries:
    raise SystemExit(
        f"_cpp_bet not built. Expected it in {_cpp_build}.\n"
        "Build it first:  cmake --build cpp_bet/build --config Release")
if not _model_files:
    raise SystemExit(
        f"No BET model in {_cpp_bet / 'model'}. Run: python cpp_bet/export_onnx.py")


# core/ ships file by file rather than as a directory. Copying the whole
# directory also shipped the research modules -- the route 1/2/5 candidates and
# the participant split -- which the app never imports and which are not part
# of a release. They are excluded by name, and `scripts/check_package.py`
# fails the build if one reappears, so adding a new one cannot leak silently.
_RESEARCH_ONLY = {
    'controllability_audit.py',   # route 1
    'pfc_guarded_joint.py',       # route 2
    'cvar_shim.py',               # route 5
    'subject_split.py',           # train/validation/test split
    'sh_canonical.py',            # SH basis comparison against SF_v2
}
_core_files = [
    (str(p), 'core') for p in sorted(Path('core').glob('*.py'))
    if p.name not in _RESEARCH_ONLY
]

# The linear-encoding page is in the main navigation now, so its modules and
# its SOCP solver are shipping code rather than research. `check_package.py`
# runs a real solve against the built package, because "cvxpy imports" is a
# weaker claim than "a cone program actually solves".
_solver_hidden = [
    'cvxpy', 'cvxpy.reductions', 'cvxpy.reductions.solvers',
    'cvxpy.reductions.solvers.defines',
    'cvxpy.reductions.solvers.conic_solvers',
    'cvxpy.reductions.solvers.conic_solvers.clarabel_conif',
    'cvxpy.reductions.solvers.conic_solvers.scs_conif',
    'cvxpy.interface.numpy_interface',
    'clarabel', 'scs', '_scs_direct',
    'scipy.sparse.csgraph._validation',
]

a = Analysis(
    ['main.py'],
    pathex=[],
    binaries=_binaries,
    # config/coils/*.json turns the coil choice into a dropdown, and coils/
    # carries the field maps themselves so the app is self-contained -- asking
    # the user to locate coil data would just trade one path for another.
    # coils/ holds only *_deltaFieldmap.nii.gz (~58 MB); regenerate it with
    # `python scripts/trim_coil_for_package.py --all`.
    datas=([('resources', 'resources'), ('docs', 'docs')] + _core_files
           + _solver_datas
           + [('atlases', 'atlases'), ('config', 'config'), ('coils', 'coils')]
           + _model_files),
    hiddenimports=(['serial', 'serial.tools.list_ports', 'PySide6.QtCore',
                    'PySide6.QtGui', 'PySide6.QtWidgets',
                    'matplotlib.backends.backend_qtagg']
                   + _solver_hidden),
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    # pyarrow (77 MB) is an optional pandas backend and cvxopt (50 MB) an
    # optional cvxpy solver; nothing in the app imports either, and cvxpy
    # reaches its solvers through try/except so a missing one is not an error.
    # The build ships CLARABEL and SCS, which is what the SOCP page solves
    # with. Dropping both takes the package from 724 MB to ~600 MB.
    excludes=['PyQt5', 'PyQt6', 'torch', 'tkinter', 'IPython',
              'pyarrow', 'cvxopt'],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='MACS',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name='MACS',
)
