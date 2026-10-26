"""Check a built `dist/MACS` before it is handed to anyone.

    python tools/packaging/check_package.py

Run after `pyinstaller macs.spec`. It answers three questions that a successful
build does not:

  * did any research-only module get shipped? `core/` used to be copied whole,
    which put the route 1/2/5 candidates and the participant split into the
    package. They were never imported, but unfinished research code has no
    business in a release, and the failure was silent.
  * is everything the app needs actually there -- atlases, coil field maps,
    the ONNX model, the shim envelope, the BET extension?
  * does the packaged binary pass its own `--selftest`?

Exit status is 0 only if all three hold.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
DIST = ROOT / "dist" / "MACS"
INTERNAL = DIST / "_internal"

#: Must not appear in `_internal/core/`. Keep in step with `_RESEARCH_ONLY`
#: in macs.spec -- this list is the check, that one is the filter, and having
#: both means a mistake in either is caught here rather than after shipping.
#: Nothing under these may appear in a package. Research scripts, archived
#: helpers, subject-derived data and training weights are all things that
#: either identify a participant or describe an experiment, and a release is
#: neither. The production BET model is an ONNX file under `model/` and is
#: explicitly not a training weight.
FORBIDDEN_TREES = ["tools/research", "tools/archive", "tools/bet", "tests",
                   "results", "runs", "research_data", "test_dic"]

#: Training weights: never, anywhere. The production BET model is ONNX.
FORBIDDEN_SUFFIXES = [".pt", ".pth", ".ckpt"]

#: DICOM is forbidden as *subject* data, but a third-party library shipping
#: its own character-set fixtures is not subject data -- pydicom carries
#: chrArab.dcm and friends, and flagging those would train the reader to
#: ignore this check. Only DICOM outside a library's own data directory is a
#: finding.
SUBJECT_SUFFIXES = [".dcm", ".ima"]
LIBRARY_DATA_MARKERS = ("pydicom/data", "pydicom\data",
                        "nibabel/tests", "nibabel\tests")

#: Subject-derived NIfTI. The atlases and the coil field maps are the only
#: NIfTI a release carries, and they live in known places.
ALLOWED_NIFTI_DIRS = ["atlases", "coils"]

RESEARCH_ONLY = [
    "controllability_audit.py",
    "pfc_guarded_joint.py",
    "cvar_shim.py",
    "subject_split.py",
    "sh_canonical.py",
]

#: Paths relative to `_internal` that the app cannot run without.
REQUIRED = [
    "core/pipeline.py",
    "core/shim_state.py",
    "core/hybrid_shim.py",
    "core/adaptive_joint.py",
    "core/linear_encoding_waveform.py",
    "core/linear_encoding_model.py",
    "core/linear_encoding_metrics.py",
    "core/linear_encoding_pareto.py",
    "core/linear_encoding_run.py",
    "core/linear_encoding_frames.py",
    "core/linear_encoding_export.py",
    "core/spatial_grid.py",
    "atlases/1MNI152_T1_1mm_brain.nii.gz",
    "atlases/pfc_mask_v3.nii.gz",
    "config/shim_delta_bounds.json",
    "model/bet_micro.onnx",
    "onnxruntime.dll",
]

#: Wanted in the package but not yet collected.  explains the linear
#: encoding page, which is now in the main navigation, so the doc should ship
#: with it -- but adding it means editing macs.spec, which belongs to the
#: packaging round. Listing it as REQUIRED before the spec collects it would
#: fail every build for a reason nobody could fix here.
PENDING_PACKAGE_INCLUSION = ["docs/LINEAR_ENCODING_SIMULATION.md"]

#: Directories that must be non-empty, with what they are for.
REQUIRED_DIRS = [
    ("coils", "coil field maps; without them the coil dropdown is empty"),
    ("config/coils", "coil descriptors that populate that dropdown"),
]


def main() -> int:
    fails = []

    if not DIST.is_dir():
        print("[FAIL] %s does not exist -- build first" % DIST)
        return 1
    exe = DIST / "MACS.exe"
    if not exe.is_file():
        fails.append("MACS.exe missing")

    leaked = [n for n in RESEARCH_ONLY if (INTERNAL / "core" / n).is_file()]
    if leaked:
        fails.append("research modules shipped: " + ", ".join(leaked))
        print("[FAIL] research-only modules in the package: %s"
              % ", ".join(leaked))
    else:
        print("[OK]   no research-only modules shipped (%d names checked)"
              % len(RESEARCH_ONLY))

    missing = [r for r in REQUIRED if not (INTERNAL / r).exists()]
    if missing:
        fails.append("missing: " + ", ".join(missing))
        for m in missing:
            print("[FAIL] missing %s" % m)
    else:
        print("[OK]   %d required files present" % len(REQUIRED))

    for rel, why in REQUIRED_DIRS:
        d = INTERNAL / rel
        n = len(list(d.glob("*"))) if d.is_dir() else 0
        if not n:
            fails.append("%s is empty (%s)" % (rel, why))
            print("[FAIL] %s empty -- %s" % (rel, why))
        else:
            print("[OK]   %s: %d entries" % (rel, n))

    # An unstamped build reports "dev" and cannot be traced back to a source
    # state, which defeats the point of stamping at package time.
    if exe.is_file():
        r = subprocess.run([str(exe), "--selftest"], capture_output=True,
                           text=True, timeout=1800, encoding="utf-8",
                           errors="replace")
        out = (r.stdout or "") + (r.stderr or "")
        globals()["_selftest_out"] = out
        banner = next((l for l in out.splitlines() if l.startswith("MACS v")), "")
        if r.returncode != 0 or "SELFTEST PASSED" not in out:
            bad = [l for l in out.splitlines() if "[FAIL]" in l]
            fails.append("packaged selftest failed")
            print("[FAIL] packaged --selftest exit %d: %s"
                  % (r.returncode, " | ".join(bad)[:200]))
        else:
            print("[OK]   packaged --selftest passed")
        if "build dev" in banner:
            fails.append("build not stamped")
            print("[FAIL] version not stamped: %s" % banner)
        elif banner:
            print("[OK]   %s" % banner)

    # Whole trees, file kinds and stray subject NIfTI.
    stray_trees = []
    for rel in FORBIDDEN_TREES:
        if (INTERNAL / rel).exists() or (DIST / rel).exists():
            stray_trees.append(rel)
    if stray_trees:
        fails.append("forbidden trees in package: " + ", ".join(stray_trees))
        print("[FAIL] these must never ship: %s" % ", ".join(stray_trees))
    else:
        print("[OK]   no research, test, results or training trees shipped")

    stray_files = [str(p.relative_to(DIST)) for p in DIST.rglob("*")
                   if p.is_file() and p.suffix.lower() in FORBIDDEN_SUFFIXES]
    if stray_files:
        fails.append("training weights shipped: " + ", ".join(stray_files[:5]))
        print("[FAIL] training weights in package: %s"
              % ", ".join(stray_files[:5]))
    else:
        print("[OK]   no training weights (*.pt/*.pth/*.ckpt) in the package")

    subject_dcm = []
    for p in DIST.rglob("*"):
        if not p.is_file() or p.suffix.lower() not in SUBJECT_SUFFIXES:
            continue
        rel = str(p.relative_to(DIST))
        if any(m in rel for m in LIBRARY_DATA_MARKERS):
            continue
        subject_dcm.append(rel)
    if subject_dcm:
        fails.append("subject DICOM shipped: " + ", ".join(subject_dcm[:5]))
        print("[FAIL] DICOM outside a library data directory: %s"
              % ", ".join(subject_dcm[:5]))
    else:
        print("[OK]   no subject DICOM (library fixtures ignored)")

    stray_nii = []
    for p in DIST.rglob("*.nii*"):
        rel = p.relative_to(INTERNAL) if INTERNAL in p.parents else p.relative_to(DIST)
        if rel.parts and rel.parts[0] in ALLOWED_NIFTI_DIRS:
            continue
        stray_nii.append(str(rel))
    if stray_nii:
        fails.append("NIfTI outside atlases/coils: " + ", ".join(stray_nii[:5]))
        print("[FAIL] subject NIfTI may have leaked: %s" % ", ".join(stray_nii[:5]))
    else:
        print("[OK]   NIfTI only under %s" % "/".join(ALLOWED_NIFTI_DIRS))

    # The solver evidence comes from the packaged binary's own --selftest,
    # which runs a real cone program inside the frozen interpreter. Running a
    # probe with the development python would test this machine, not the
    # bundle, which is the mistake this replaced.
    if exe.is_file():
        so = (globals().get("_selftest_out") or "")
        if "linear encoding SOCP" in so and "[OK]   linear encoding SOCP" in so:
            line = next(l for l in so.splitlines()
                        if l.startswith("[OK]   linear encoding SOCP"))
            print("[OK]   packaged SOCP solves: %s" % line.split(":", 1)[1].strip())
        else:
            fails.append("packaged SOCP did not solve")
            bad = [l for l in so.splitlines() if "linear encoding" in l
                   or "cvxpy" in l]
            print("[FAIL] packaged linear encoding: %s"
                  % (" | ".join(bad)[:160] or "no report in --selftest"))
        if "[OK]   cvxpy" in so:
            print("[OK]   %s" % next(l for l in so.splitlines()
                                     if l.startswith("[OK]   cvxpy"))[7:])
        else:
            fails.append("no SOCP solver in the package")
            print("[FAIL] no CLARABEL or SCS reported by the packaged selftest")

    n_files = sum(1 for p in DIST.rglob("*") if p.is_file())
    size_mb = sum(p.stat().st_size for p in DIST.rglob("*") if p.is_file()) / 1048576
    print("       %d files, %.0f MB" % (n_files, size_mb))

    print()
    if fails:
        print("PACKAGE CHECK FAILED: %s" % "; ".join(fails))
        return 1
    print("PACKAGE CHECK PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
