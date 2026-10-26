"""Utility script to copy core modules from the Streamlit app.

Run this once to populate the core/ directory:
    python core/copy_core.py

The core modules (physics.py, shim_solver.py, fsl_replace.py, etc.)
have NO Streamlit imports and can be used directly.
"""
import shutil
import os

SRC = os.path.join(os.path.dirname(__file__), "..", "..",
                   "shimming_app", "shimming_app", "core")
DST = os.path.dirname(__file__)

FILES = [
    "physics.py",
    "shim_solver.py",
    "fsl_replace.py",
    "B0_cal_algorithm.py",
    "data_context.py",
    "dicom_to_nifti.py",
]


def main():
    src = os.path.abspath(SRC)
    dst = os.path.abspath(DST)
    print(f"Source: {src}")
    print(f"Dest:   {dst}")

    for f in FILES:
        s = os.path.join(src, f)
        d = os.path.join(dst, f)
        if os.path.exists(s):
            shutil.copy2(s, d)
            print(f"  ✓ {f}")
        else:
            print(f"  ✗ {f} not found")

    print("\nDone. Review data_context.py — remove any `import streamlit` lines.")


if __name__ == "__main__":
    main()
