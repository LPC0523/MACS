"""Resource path resolution — works both from source and from a PyInstaller bundle.

When frozen, PyInstaller extracts `datas` next to the exe (onedir: `_internal/`)
and exposes that root as `sys._MEIPASS`. From source the root is the package dir.
"""
import os
import sys


def app_base() -> str:
    """Root directory that `datas` entries (resources/, atlases/) live under."""
    if getattr(sys, "frozen", False):
        return getattr(sys, "_MEIPASS", os.path.dirname(sys.executable))
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def resource(*parts) -> str:
    """Absolute path to a bundled resource, e.g. resource('atlases', 'x.nii.gz')."""
    return os.path.join(app_base(), *parts)


def user_data_dir() -> str:
    """Writable per-user directory for settings that must survive a restart.

    Never write into app_base(): when frozen it is a temp dir that PyInstaller
    deletes on exit.

    The app was renamed PFC Shim -> MACS; anything saved under the old
    ~/.pfc_shim is migrated once so saved scanner shim values are not lost.
    """
    home = os.path.expanduser("~")
    d = os.path.join(home, ".macs")
    os.makedirs(d, exist_ok=True)

    legacy = os.path.join(home, ".pfc_shim")
    if os.path.isdir(legacy):
        import shutil
        for name in os.listdir(legacy):
            src, dst = os.path.join(legacy, name), os.path.join(d, name)
            if os.path.isfile(src) and not os.path.exists(dst):
                try:
                    shutil.copy2(src, dst)
                except OSError:
                    pass
    return d


def find_atlases():
    """Locate (MNI brain, PFC mask) atlas files.

    Returns (mni_path, pfc_path); either may be "" if not found.
    Searches the bundled atlases/ dir first, then legacy locations so an
    existing source checkout keeps working.
    """
    base = app_base()
    candidates = [
        os.path.join(base, "atlases"),
        os.path.join(base, "resources"),
        # legacy: sibling Streamlit app
        os.path.join(base, "..", "shimming_app", "shimming_app", "atlases"),
        os.path.join(base, "..", "shimming_app", "release", "atlases"),
    ]

    mni_path = ""
    pfc_path = ""
    for d in candidates:
        if not os.path.isdir(d):
            continue
        for f in sorted(os.listdir(d)):
            if not f.endswith(".nii.gz"):
                continue
            low = f.lower()
            if not mni_path and "mni152" in low and "brain" in low:
                mni_path = os.path.join(d, f)
            if not pfc_path and "pfc_mask" in low:
                pfc_path = os.path.join(d, f)
        if mni_path and pfc_path:
            break

    return mni_path, pfc_path
