"""Shared helpers for the research batch scripts. No test file imports here.

A research script used to reach into `tests/run_ridge_sweep` for
`find_subjects`, which made a batch analysis depend on a test module: renaming
or trimming a test would then break a study, and the dependency ran the wrong
way round. The shared parts live here instead. Tests may import this; nothing
here imports a test.

Paths are derived from this file's own location, never from the working
directory and never from a hard-coded interpreter or user directory, so a
script runs the same from anywhere:

    python tools/research/run_controllability_audit.py
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

#: tools/research/common.py -> tools/research -> tools -> shimming_qt
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

#: Where processed subjects are looked for, in order. Kept as a list so a new
#: cohort is one line rather than a rewrite.
SUBJECT_ROOTS = [
    PROJECT_ROOT.parent.parent / "2nd_order_SH",
    PROJECT_ROOT.parent / "test_dic",
]

__all__ = ["PROJECT_ROOT", "RESULTS_DIR", "SUBJECT_ROOTS", "find_subjects",
           "results_path", "sha256_file", "find_fieldmap", "find_roi"]


def results_path(*parts) -> Path:
    """A path under `results/`, created on demand.

    Research output stays where it has always been. Moving the scripts must not
    move the results they have already produced.
    """
    p = RESULTS_DIR.joinpath(*parts)
    p.mkdir(parents=True, exist_ok=True)
    return p


def sha256_file(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def find_subjects(limit: int = 0, need_pfc: bool = True) -> list:
    """(name, deltaB0_hz, brain_mask, pfc_mask) for each processed subject.

    A subject counts only when its brain mask exists alongside the field map;
    `need_pfc` additionally requires a PFC mask, either beside the field map or
    in the subject's `roi_cache`.
    """
    found, seen = [], set()
    for root in SUBJECT_ROOTS:
        if not root.is_dir():
            continue
        for hz in sorted(root.rglob("*_deltaB0_hertz.nii.gz")):
            name = hz.name[: -len("_deltaB0_hertz.nii.gz")]
            if name in seen:
                continue
            brain = hz.parent / ("%s_bet_mask.nii.gz" % name)
            if not brain.is_file():
                continue
            pfc = next((p for p in (hz.parent / ("%s_pfc_mask.nii.gz" % name),
                                    hz.parent / "roi_cache" / "pfc_native.nii.gz")
                        if p.is_file()), None)
            if need_pfc and pfc is None:
                continue
            seen.add(name)
            found.append((name, hz, brain, pfc))
            if limit and len(found) >= limit:
                return found
    return found


def find_fieldmap(channel_dir):
    """The field map inside one channel directory, or None."""
    d = Path(channel_dir)
    for pattern in ("*deltaFieldmap.nii*", "*_resampled_to_brain.nii*", "*.nii*"):
        hits = sorted(d.glob(pattern))
        if hits:
            return hits[0]
    return None


def find_roi(subject_dir, name="pfc"):
    """A named ROI mask for a subject directory, or None."""
    d = Path(subject_dir)
    for pattern in ("*_%s_mask.nii*" % name, "roi_cache/%s_native.nii*" % name):
        hits = sorted(d.glob(pattern))
        if hits:
            return hits[0]
    return None
