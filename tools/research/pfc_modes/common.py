"""Shared pieces for the PFC residual-mode study. Research only.

Nothing here is imported by the Qt app or by `core/` -- this module reads the
production algorithm layer but never writes to it. The study's whole premise is
the measured fact that the 12-channel prefrontal array is nearly collinear with
the 8 scanner harmonics *inside* the PFC (0.20% independent energy on S28) while
being 32.7% independent over the whole brain: its one irreplaceable property is
spatial localisation, not high order.

Frozen conventions, per the study brief. Changing any of these invalidates every
number already reported:

  * scanner SH        A10 A11 B11 A20 A21 B21 A22 B22   (8 terms, production)
  * diagnostic only   + A00                             (9 terms, never solved)
  * secondary         A20 A21 B21 A22 B22               (5 terms)
  * coil              12 channels, Hz/A, |i| <= 3 A
  * SH bounds         intersection of delta and absolute limits from shim_state
  * region weighting  PFC and rest each normalised by their own voxel count,
                      lambda_rest = 1
  * objective         plain least squares, no pre-solve demeaning
  * primary metric    ROI RMSE with no re-lock of the centre frequency
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import nibabel as nib

from tools.research.common import PROJECT_ROOT, find_subjects  # noqa: F401
from core.hybrid_shim import build_sh_basis, GAMMA, SH1, SH2, SH12
from core.data_context import resample_to_target, same_grid
from core.subject_split import split_subjects

__all__ = [
    "SH8", "SH9", "SH5", "COIL_DIR", "IMAX", "LAMBDA_REST", "OUT_DIR",
    "EXCLUDED_RADIAN", "RADIAN_P99_HZ", "cohort",
    "basis_with_a00", "load_subject", "coil_maps_for", "sh_matrix",
    "independent_energy", "sh_bounds_production", "triplane_png",
    "protected_hashes", "write_json", "Progress",
]

#: Scans whose `*_deltaB0_hertz.nii.gz` holds radian-scale data. Measured:
#: p99|b0| over the brain is 9.0-9.2 Hz for these four against a cohort median
#: of 273.5 Hz and a next-smallest of 182.7. `core/subject_split` already
#: anticipates them ("four are excluded ... the quotas scale down with them"),
#: so the fold split must be computed on the surviving 40, not on all 44.
#: Frozen as a list, and re-checked numerically by `cohort()`.
EXCLUDED_RADIAN = ("S22", "S22_ear", "S24", "S24_ear")

#: A scan below this p99|b0| over the brain is not in hertz. Set an order of
#: magnitude clear of both sides of the observed gap (9.2 vs 182.7 Hz).
RADIAN_P99_HZ = 20.0

#: Scans whose field map was acquired *after* a local coil had already shimmed.
#: They are residuals, not baselines: averaging them into a target removes
#: exactly the component the present hardware can already cancel, which biases
#: the target away from what a new coil should be designed for.
#:
#: Classification is by DICOM series description, not directory name, because
#: the two disagree. Convention, confirmed with the user 2026-08-17:
#:   b0map_tra*            baseline (standard acquisition)
#:   b0map_tra_2nd*        baseline, after second-order shim
#:   *earshim*, *ear_shim* baseline, passive earplug condition
#:   any `_ear` directory  baseline (the passive condition is the variable)
#:   b0map_tra_shim*       EXCLUDED -- local coil active
#:   *pfc_shim*, *hybrid*  EXCLUDED -- local coil active
#:
#: Under that rule four scans are excluded, all without an `_ear` suffix:
EXCLUDED_SHIMMED = ("S15", "S21", "S28", "S29")

#: `_ear` directories whose series description nonetheless says a coil was on.
#: Kept, per the rule above, but flagged: if a sensitivity check shows the
#: target moves when they are dropped, this is the first thing to revisit.
DISPUTED_EAR = ("S21_ear", "S28_ear", "S20_ear", "S29_ear", "S27_ear")


def series_description(subject_dir) -> str:
    """First SeriesDescription/ProtocolName found in the scan's JSON sidecars."""
    import json as _json

    d = Path(subject_dir)
    for j in sorted(d.glob("*.json")):
        try:
            meta = _json.loads(j.read_text(encoding="utf-8", errors="replace"))
        except Exception:
            continue
        for k in ("SeriesDescription", "ProtocolName"):
            if meta.get(k):
                return str(meta[k])
    hits = sorted(d.glob("*b0map*"))
    return hits[0].name if hits else ""


def is_baseline(name: str, series: str) -> bool:
    """Whether this scan predates any local-coil shim. See EXCLUDED_SHIMMED."""
    if str(name).lower().endswith("_ear"):
        return True
    s = (series or "").lower()
    if "earshim" in s or "ear_shim" in s:
        return True
    if "tra_2nd" in s:
        return True
    if "pfc_shim" in s or "hybrid" in s:
        return False
    return "tra_shim" not in s

#: The production variable set. A00 is not among them: the console does not take
#: it from this page.
SH8 = list(SH12)
#: Diagnostic only -- what the field would look like if a uniform offset were
#: also free. Never solved for.
SH9 = ["A00"] + SH8
#: Second order alone, the set that survives if prescan owns first order.
SH5 = list(SH2)

COIL_DIR = PROJECT_ROOT / "coils" / "pfc_12ch"
IMAX = 3.0
LAMBDA_REST = 1.0

OUT_DIR = PROJECT_ROOT / "results" / "hybrid_pfc_residual_modes"

#: Production files this study must not modify. Hashed before and after.
PROTECTED = [
    "core/hybrid_shim.py", "core/adaptive_joint.py", "core/shim_state.py",
    "core/version.py", "widgets/system_shim_widget.py",
    "workers/system_shim_worker.py", "macs.spec",
    "config/coils/pfc_12ch.json", "config/coils/sf_12ch.json",
]


class Progress:
    """Timestamped progress to stdout, unbuffered."""

    def __init__(self, tag):
        import time
        self._t0 = time.time()
        self._tag = tag
        self.log(f"=== {tag} ===")

    def log(self, msg):
        import sys, time
        print(f"[{time.time() - self._t0:7.1f}s] {msg}", flush=True)
        sys.stdout.flush()


def cohort(verify_scale: bool = True, baseline_only: bool = True) -> tuple:
    """The usable cohort and its fold assignment. Returns (entries, folds).

    Excludes `EXCLUDED_RADIAN` and (with `baseline_only`) `EXCLUDED_SHIMMED`
    *before* splitting, because the split's quotas are proportional to how many
    scans are handed in -- splitting everything and dropping some afterwards
    would leave the folds unbalanced and would silently differ from what every
    other study on this cohort used.

    With `verify_scale` the exclusion lists are checked against the data rather
    than trusted: a scan on the radian list that looks like hertz (or vice
    versa), or a shimmed-list entry whose series description no longer says so,
    raises instead of quietly proceeding.
    """
    entries = [e for e in find_subjects(need_pfc=True)]
    kept = [e for e in entries if e[0] not in EXCLUDED_RADIAN]
    if baseline_only:
        bad = []
        for name, hz, _, _ in kept:
            series = series_description(Path(hz).parent)
            listed = name in EXCLUDED_SHIMMED
            if is_baseline(name, series) == listed:
                bad.append(f"{name}: series {series!r}, "
                           f"{'on' if listed else 'not on'} EXCLUDED_SHIMMED")
        if verify_scale and bad:
            raise RuntimeError(
                "cohort baseline check failed -- the frozen shimmed-scan list "
                "no longer matches the series descriptions:\n  "
                + "\n  ".join(bad))
        kept = [e for e in kept if e[0] not in EXCLUDED_SHIMMED]

    if verify_scale:
        import nibabel as nib
        wrong = []
        for name, hz, brain, _ in entries:
            b = np.asarray(nib.load(str(brain)).get_fdata()) > 0.5
            v = np.asarray(nib.load(str(hz)).get_fdata())[b]
            p99 = float(np.percentile(np.abs(v), 99)) if v.size else 0.0
            listed = name in EXCLUDED_RADIAN
            looks_radian = p99 < RADIAN_P99_HZ
            if listed != looks_radian:
                wrong.append(f"{name}: p99|b0|={p99:.1f} Hz, "
                             f"{'on' if listed else 'not on'} the exclusion list")
        if wrong:
            raise RuntimeError(
                "cohort scale check failed -- the frozen exclusion list no "
                "longer matches the data:\n  " + "\n  ".join(wrong))

    folds = split_subjects([e[0] for e in kept])
    return kept, folds


def basis_with_a00(affine, shape) -> dict:
    """The 8 production harmonics plus a constant column named A00.

    A00's scaling follows the same convention as the rest -- df = gamma * C *
    f(x,y,z) with f == 1 -- so its coefficient would be in uT. Only the span
    matters for the projections it is used in, but keeping the units consistent
    means the reported coefficient is comparable with the others.
    """
    b = build_sh_basis(affine, shape)
    b["A00"] = np.full(tuple(shape), GAMMA, dtype=np.float64)
    return b


def load_subject(entry) -> dict:
    """(name, hz, brain, pfc) -> arrays, with every grid checked against the B0."""
    name, hz_path, brain_path, pfc_path = entry
    hz = nib.load(str(hz_path))
    b0 = np.asarray(hz.get_fdata(), dtype=np.float64)
    affine, shape = hz.affine, b0.shape

    br = nib.load(str(brain_path))
    brain = np.asarray(br.get_fdata()) > 0.5
    if brain.shape != shape:
        raise ValueError(f"{name}: brain mask {brain.shape} != B0 {shape}")
    brain_same_grid = same_grid(br.shape, br.affine, shape, affine)

    pfc = None
    pfc_same_grid = None
    if pfc_path is not None:
        pm = nib.load(str(pfc_path))
        pfc_same_grid = same_grid(pm.shape, pm.affine, shape, affine)
        raw = np.asarray(pm.get_fdata())
        if pfc_same_grid:
            pfc = raw > 0.5
        else:
            # World-coordinate resample; nearest neighbour, because a mask
            # interpolated linearly is no longer a mask.
            pfc = resample_to_target(raw.astype(np.float32), pm.affine,
                                     affine, shape, order=0) > 0.5
        pfc = pfc & brain

    return {"name": name, "b0": b0, "brain": brain, "pfc": pfc,
            "affine": affine, "shape": shape,
            "hz_path": Path(hz_path), "brain_path": Path(brain_path),
            "pfc_path": Path(pfc_path) if pfc_path else None,
            "brain_same_grid": bool(brain_same_grid),
            "pfc_same_grid": None if pfc_same_grid is None else bool(pfc_same_grid)}


_COIL_CACHE: dict = {}


def coil_maps_for(affine, shape, log=None) -> tuple:
    """The 12 channel maps on this subject's grid, in Hz/A.

    Cached by (shape, affine bytes): a batch over 44 subjects would otherwise
    resample the same 12 volumes once per subject.
    """
    key = (tuple(shape), np.asarray(affine, dtype=np.float64).tobytes())
    if key in _COIL_CACHE:
        return _COIL_CACHE[key]

    import re
    dirs = sorted([p for p in COIL_DIR.iterdir()
                   if p.is_dir() and re.match(r"ch\d+", p.name, re.I)],
                  key=lambda p: int(re.findall(r"\d+", p.name)[0]))
    if not dirs:
        raise FileNotFoundError(f"no ch*_nii under {COIL_DIR}")

    vols, names = [], []
    for d in dirs:
        hits = (sorted(d.glob("*deltaFieldmap.nii*"))
                or sorted(d.glob("*_resampled_to_brain.nii*"))
                or sorted(d.glob("*.nii*")))
        if not hits:
            raise FileNotFoundError(f"no field map in {d}")
        img = nib.load(str(hits[0]))
        data = np.asarray(img.get_fdata(), dtype=np.float64)
        if not same_grid(img.shape, img.affine, shape, affine):
            data = resample_to_target(data.astype(np.float32), img.affine,
                                      affine, shape, order=1)
        vols.append(data)
        names.append(d.name.replace("_nii", ""))
        if log:
            log(f"    {names[-1]}")
    out = (np.stack(vols, axis=-1), names)
    _COIL_CACHE[key] = out
    return out


def sh_matrix(basis, terms, region) -> np.ndarray:
    """(n_voxels, n_terms) design matrix over a boolean region."""
    return np.stack([basis[n][region] for n in terms], axis=-1).astype(np.float64)


def independent_energy(C_region, A_region) -> float:
    """Percent of the array's field energy that the given harmonics cannot reach.

    ||C - Q Qt C||_F^2 / ||C||_F^2, with Q an orthonormal basis of the harmonic
    span. Frobenius over all channels at once, so it measures the *subspace*,
    not a per-channel average -- channels that duplicate each other cannot
    inflate it.
    """
    if C_region.size == 0:
        return float("nan")
    Q, _ = np.linalg.qr(A_region)
    R = C_region - Q @ (Q.T @ C_region)
    denom = float(np.sum(C_region ** 2))
    return float("nan") if denom <= 0 else 100.0 * float(np.sum(R ** 2)) / denom


def sh_bounds_production(terms) -> list:
    """Per-term (low, high) from the shipping shim state.

    The intersection of the delta envelope and the console's absolute limit,
    evaluated against the default baseline (all zero) -- which is what a site
    with no imported scanner state actually gets.
    """
    from core.shim_state import SystemShimState

    st = SystemShimState()
    return [tuple(map(float, b)) for b in st.delta_bounds(list(terms))]


def triplane_png(path, background, overlays, affine=None, title="",
                 clim=None, cmap="jet"):
    """Axial / coronal / sagittal through the centroid of the first overlay.

    Written for QA of a registration, so the slice is chosen by where the
    overlay actually is rather than by the middle of the volume -- a mask that
    landed in the wrong place is invisible in a mid-volume slice.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    bg = np.asarray(background, dtype=np.float64)
    ov = [(np.asarray(m, dtype=bool), c) for m, c in overlays]
    ref = ov[0][0] if ov else (bg != 0)
    if ref.any():
        cx, cy, cz = [int(round(v)) for v in np.argwhere(ref).mean(axis=0)]
    else:
        cx, cy, cz = [s // 2 for s in bg.shape]

    if clim is None:
        finite = bg[np.isfinite(bg)]
        v = np.percentile(np.abs(finite), 98) if finite.size else 1.0
        clim = (-v, v)

    views = [("Axial", lambda a: a[:, :, cz].T),
             ("Coronal", lambda a: a[:, cy, :].T),
             ("Sagittal", lambda a: a[cx, :, :].T)]

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.6))
    for ax, (nm, cut) in zip(axes, views):
        ax.imshow(cut(bg), cmap=cmap, origin="lower",
                  vmin=clim[0], vmax=clim[1])
        for m, colour in ov:
            sl = cut(m.astype(float))
            ax.contour(sl, levels=[0.5], colors=[colour], linewidths=1.2)
        ax.set_title(f"{nm}  (i={cx}, j={cy}, k={cz})", fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)


def protected_hashes() -> dict:
    out = {}
    for rel in PROTECTED:
        p = PROJECT_ROOT / rel
        out[rel] = (hashlib.sha256(p.read_bytes()).hexdigest()
                    if p.is_file() else None)
    return out


def write_json(path, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)

    def _default(o):
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, Path):
            return str(o)
        if isinstance(o, (np.bool_, bool)):
            return bool(o)
        raise TypeError(type(o))

    Path(path).write_text(
        json.dumps(obj, indent=2, ensure_ascii=False, default=_default),
        encoding="utf-8")
