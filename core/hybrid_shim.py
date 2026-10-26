"""System-shim algorithms: spherical-harmonic basis and the joint solve.

Live module. Three functions in it are not -- `load_channels`,
`resample_mask_to_native` and `load_b0_and_mask` date from when the page loaded
its own data, and are individually marked. Everything else is on the system-shim
page's hot path.

Originally: earplug (passive) + 2nd order SH (system) + PFC coils (active).

Pure algorithm layer, ported 1:1 from the Streamlit page
`shimming_app/pages/_page_hybrid_shim.py`. No Qt and no Streamlit imports, so it
can be unit-tested and called from a worker thread.
"""
import os
import re
import glob as _glob

import numpy as np
import nibabel as nib
from scipy.linalg import lstsq
from scipy.ndimage import affine_transform
from scipy.optimize import minimize

from core.data_context import natural_key, resample_to_target, same_grid

# ── Constants (must match the reference implementation) ──────────────
GAMMA = 42.577
SH1 = ["A10", "A11", "B11"]
SH2 = ["A20", "A21", "B21", "A22", "B22"]
#: The eight variables the optimiser solves for. Zeroth order (A00, the centre
#: frequency) is NOT among them -- it is not one of the shim coefficients the
#: console accepts. It is not free either: the solve is plain least squares
#: (`solve_adaptive_joint(centre=False)`), so whatever uniform offset the eight
#: harmonics cannot remove stays in the residual and is reported as it is,
#: rather than being written off against a prescan that does not happen.
SH12 = SH1 + SH2
ORDER = {n: 1 for n in SH1}
ORDER.update({n: 2 for n in SH2})
UNIT = {1: "uT/m", 2: "uT/m2"}
CALIBRATION = {
    "A10": -0.9188, "A11": -0.9252, "B11": +0.9169,
    "A20": +0.8806, "A21": +0.8810, "B21": -0.8602,
    "A22": +0.8837, "B22": -0.8648,
}
SH_BOUNDS = {n: (-2000.0, 2000.0) for n in SH12}


def build_sh_basis(affine, shape):
    """Calibrated spherical-harmonic basis maps, in Hz per unit shim value."""
    s = shape
    ii, jj, kk = np.meshgrid(np.arange(s[0]), np.arange(s[1]), np.arange(s[2]),
                             indexing="ij")
    ijk = np.stack([ii, jj, kk, np.ones_like(ii)], axis=-1)
    w = np.einsum("ij,...j->...i", affine, ijk)[..., :3]
    x, y, z = w[..., 0] / 1000, w[..., 1] / 1000, w[..., 2] / 1000
    polys = {
        "A10": z, "A11": x, "B11": y,
        "A20": 2 * z ** 2 - x ** 2 - y ** 2, "A21": x * z, "B21": y * z,
        "A22": x ** 2 - y ** 2, "B22": 2 * x * y,
    }
    return {n: CALIBRATION[n] * GAMMA * polys[n] for n in SH12}


# LEGACY / NOT USED BY CURRENT QT APP —— 系统匀场页改为复用一键匀场已重采样好的线圈场图（app_state["coil_maps"]），不再自行加载。
# 替代：core/pipeline.py::_load_coil_maps
def load_channels(folder, ref_affine, ref_shape, log_cb=None):
    """Load PFC channel field maps, resampling to the reference grid."""
    def _log(m):
        if log_cb:
            log_cb(m)

    entries = sorted(
        [e for e in os.listdir(folder)
         if os.path.isdir(os.path.join(folder, e)) and re.match(r"ch\d+", e, re.I)],
        key=natural_key)
    if not entries:
        raise FileNotFoundError(f"No ch*_nii folders in {folder}")

    B0_list, names = [], []
    for ch_dir_name in entries:
        ch_dir = os.path.join(folder, ch_dir_name)
        ch_label = ch_dir_name.replace("_nii", "")
        resampled = _glob.glob(os.path.join(ch_dir, "*resampled_to_brain.nii*"))
        delta = _glob.glob(os.path.join(ch_dir, "*deltaFieldmap.nii*"))
        load_path = resampled or delta or _glob.glob(os.path.join(ch_dir, "*.nii*"))
        if not load_path:
            _log(f"  x {ch_dir_name}: no NIfTI")
            continue

        try:
            img = nib.load(load_path[0])
            if not same_grid(img.shape, img.affine, ref_shape, ref_affine):
                _log(f"  {ch_dir_name}: resample {img.shape[:3]} -> {tuple(ref_shape)}")
                data = resample_to_target(img.get_fdata(dtype=np.float32),
                                          img.affine, ref_affine, ref_shape, order=1)
            else:
                data = img.get_fdata(dtype=np.float32)
            B0_list.append(data)
            names.append(ch_label)
            _log(f"  + {ch_dir_name}")
        except Exception as e:
            _log(f"  x {ch_dir_name}: {e}")

    if not B0_list:
        raise FileNotFoundError(f"No channels loaded from {folder}")
    return np.stack(B0_list, axis=-1), names


# LEGACY / NOT USED BY CURRENT QT APP —— 颞叶保护移除后没有调用方；ROI 重采样统一走 roi_registry。
# 替代：core/roi_registry.py::_resample_to
def resample_mask_to_native(mask_path, ref_img):
    """Load a mask (MNI or native) and bring it onto the reference grid."""
    m_img = nib.load(mask_path)
    if m_img.shape == ref_img.shape:
        return m_img.get_fdata() > 0.5
    M = np.linalg.inv(m_img.affine) @ ref_img.affine
    resampled = affine_transform(m_img.get_fdata(), M[:3, :3], offset=M[:3, 3],
                                 output_shape=ref_img.shape, order=0, cval=0.0)
    return resampled > 0.5


def local_sh_equivalent(coil_correction, basis, mask, terms=None):
    """How much of the coil field the scanner would see as a shim change.

    A surface array's field is not orthogonal to the low-order harmonics: part
    of what it produces is, to the scanner, indistinguishable from a gradient
    offset. So the change the scanner must actually make is the model's total
    minus that part -- `scanner_delta = theta - local_sh_equiv` in
    2nd_order_SH/pfc_protected_ge_shim.py (L954).

    Telling the scanner the full total instead double-counts: the coils and the
    scanner would both supply the same component, and the field would be
    over-corrected by exactly the overlap.

    Least-squares projection of the coil correction onto the basis, over
    `mask`, since we have field maps rather than the mode-coefficient matrix
    `k_coeff` that the simulation carries.
    """
    terms = list(terms or SH12)
    m = mask > 0
    if not m.any():
        return np.zeros(len(terms))
    A = np.stack([basis[n][m] for n in terms], axis=-1)
    b = coil_correction[m]
    coef, *_ = np.linalg.lstsq(A, b, rcond=None)
    return np.asarray(coef, dtype=float)


def suggest_centre_frequency(residual, mask) -> float:
    """Mean residual field in Hz -- the centre-frequency offset still present.

    Reported, never applied: A00 is not among the coefficients the console
    takes from this page. Since the solve is plain least squares this offset is
    error the eight harmonics were unable to remove, not a quantity something
    downstream will absorb, so a large value means the harmonics ran out of
    authority rather than that a prescan is owed.
    """
    import numpy as _np

    m = mask > 0 if mask is not None else _np.ones(residual.shape, bool)
    if not m.any():
        return 0.0
    return float(_np.mean(residual[m]))


def joint_optimize(b0, basis, brain_mask, temporal_mask, pfc_array, n_pfc,
                   pfc_imax, lam_temp, lam_sh, lam_pfc, log_cb=None,
                   sh_terms=None, sh_bounds=None,
                   lambda_pfc_low=0.0, adaptive_weights=False,
                   scanner_baseline=None, scanner_limits=None,
                   roi_mask=None, lambda_rest=1.0):
    """Joint system-SH + local-coil solve, with the coils projected off the harmonics first.

    Each coil map is split by weighted least squares against the harmonic basis
    into the part the scanner could reproduce (low) and the part only the coil
    can (high). The solve uses the high part, so the two sets of variables stop
    competing for the same field, and the low part is subtracted from what the
    scanner is told to change:

        scanner_delta = model_delta - pfc_low_coeff @ currents

    Previously the design matrix was a bare concatenation and the coil's
    harmonic equivalent was computed *after* solving, by regressing the finished
    coil correction onto the basis. That worked, but the three reported numbers
    (model delta, coil equivalent, scanner delta) came from two different fits;
    now they come from one, so the identity above holds by construction rather
    than approximately.

    `lambda_pfc_low` additionally penalises the coils for producing low-order
    field, pushing the harmonics to take it. It defaults to **off**: measured
    across five subjects and two arrays, the harmonics and the coils divide the
    work on their own, and the penalty only removed coil authority (net coil
    contribution fell from 14.7 Hz at 0 to 3.7 Hz at 1.0). The knob is here for
    a calibration where they do fight.

    `adaptive_weights` likewise defaults to off. Weighting voxels by distance
    from the median field improves the weighted objective but costs whole-brain
    RMSE badly on this data (39.1 -> 67.5 Hz).

    `roi_mask` weights a target region against the rest of the brain, with
    `lambda_rest` the weight the rest carries. Without it the objective counts
    every voxel equally, and a surface array sized for one region then loses the
    vote: the PFC is 5.7% of the brain, so the solver spends the currents on
    global low-order field and the PFC gets *worse* than harmonics alone
    (measured on S28, whole brain: PFC std 40.5 -> 43.8 Hz on adding the array,
    while the brain improves 47.9 -> 45.9). The coils end up standing in for a
    harmonic -- their low-order equivalent reached B21 +299 against a model
    total of +283, leaving the scanner nothing to do.

    `lambda_rest=1.0` is not a preference for the target: `make_region_weights`
    normalises each region by its own voxel count first, so 1.0 is the point
    where the two regions carry *equal* weight rather than equal votes per
    voxel. On S28 that recovers the array's purpose at a slight cost to the
    rest -- PFC 43.8 -> 37.2 Hz, whole brain 45.9 -> 47.2. Pushing further
    (0.2, 0.05) buys little more and degrades the brain steeply.

    Returns (delta_sh, i_pfc, residual, stats, res, scanner_delta, local_sh_equiv).
    """
    from core.adaptive_joint import solve_adaptive_joint

    def _log(m):
        if log_cb:
            log_cb(m)

    sh_terms = list(sh_terms) if sh_terms else list(SH2)
    unknown = [n for n in sh_terms if n not in basis]
    if unknown:
        raise ValueError(f"unknown SH terms: {unknown}")

    brain = brain_mask > 0.5
    if sh_bounds is None:
        sh_bounds = [SH_BOUNDS[n] for n in sh_terms]
    else:
        sh_bounds = [tuple(map(float, b)) for b in sh_bounds]
        if len(sh_bounds) != len(sh_terms):
            raise ValueError(
                f"sh_bounds has {len(sh_bounds)} entries for {len(sh_terms)} terms")

    coil_maps = pfc_array if n_pfc else None
    _log(f"优化 {len(sh_terms)}+{n_pfc}={len(sh_terms) + n_pfc} 个变量 "
         f"(SH: {', '.join(sh_terms)})")

    # Target-region weighting, when a region was given and it actually overlaps
    # the brain. A mask that misses the brain means the registration failed, and
    # falling back to whole-brain is better than refusing to shim -- but it is
    # said out loud, because the currents it produces are the ones this page
    # was just shown to spend on the wrong region.
    region = None
    if roi_mask is not None:
        region = np.asarray(roi_mask, dtype=bool) & brain
        if not region.any():
            _log("⚠ 目标区与脑掩膜无交集（配准可能失败）——退回全脑等权")
            region = None

    weighting = ({"pfc_mask": region, "lambda_rest": float(lambda_rest)}
                 if region is not None else
                 {"weight_kwargs": None if adaptive_weights else {
                     "base": 1.0, "alpha": 0.0, "top_fraction": 0.0}})

    sol = solve_adaptive_joint(
        b0, brain, [basis[n] for n in sh_terms], coil_maps,
        sh_bounds=sh_bounds, imax=pfc_imax,
        # lam_sh / lam_pfc keep their existing meaning: harmonic and current
        # regularisation. The UI spin boxes feed straight through.
        lambda_sh=lam_sh, lambda_current=lam_pfc,
        lambda_pfc_low=lambda_pfc_low,
        **weighting,
        # Deliverability is checked on theta - K i, which is what the console
        # receives; the solver's box only ever constrained theta.
        scanner_baseline=scanner_baseline, scanner_limits=scanner_limits,
        sh_names=sh_terms, on_violation="report",
        # Plain overdetermined least squares: min ||A w + b0||^2, no free
        # constant column. The coefficients this page produces are typed into
        # the console and scanned with directly -- no prescan re-locks the
        # centre frequency in between -- so a uniform offset is real error and
        # has to be paid for by the solve, not handed to a step that never
        # happens. See solve_adaptive_joint's `centre`.
        centre=False,
        log_cb=_log)

    delta_sh = sol["model_sh_delta"]
    i_pfc = sol["coil_currents"]
    residual = sol["residual"]

    def _rmse(v, m):
        return float(np.sqrt(np.mean(v[m] ** 2)))

    def _std(v, m):
        return float(np.std(v[m]))

    # `temporal_mask` is accepted for signature compatibility and is expected to
    # be empty: the objective is whole-brain RMSE with no region held back.
    temporal_brain = (temporal_mask > 0.5) & brain if temporal_mask is not None         else np.zeros_like(brain)
    hybrid_mask = brain & (~temporal_brain)

    stats = {}
    regions = [("brain", brain), ("temporal", temporal_brain),
               ("hybrid", hybrid_mask)]
    if region is not None:
        # Reported whether or not it was weighted, so "did the target actually
        # improve" is answerable from the result alone.
        regions.append(("roi", region))
        stats["_roi_weighted"] = True
        stats["_roi_lambda_rest"] = float(lambda_rest)
        stats["_roi_voxels"] = int(region.sum())
    for rn, rm in regions:
        if rm.sum() == 0:
            continue
        stats[f"{rn}_rmse_pre"] = _rmse(b0, rm)
        stats[f"{rn}_rmse_post"] = _rmse(residual, rm)
        stats[f"{rn}_std_pre"] = _std(b0, rm)
        stats[f"{rn}_std_post"] = _std(residual, rm)
    stats["_sh_terms"] = list(sh_terms)

    class _Res:
        pass

    res = _Res()
    res.x = np.concatenate([delta_sh, i_pfc])
    res.success = sol["success"]
    res.nit = sol["n_iter"]
    res.message = sol["message"]

    stats["centre_frequency_hz"] = sol["centre_frequency_hz"]
    stats["_scanner_violations"] = sol.get("scanner_violations", [])
    return (delta_sh, i_pfc, residual, stats, res,
            sol["scanner_sh_delta"], sol["pfc_low_equiv"])


# LEGACY / NOT USED BY CURRENT QT APP —— 系统匀场页不再自行读取 NIfTI，B0 与脑掩膜都来自 app_state。
# 替代：core/pipeline.py::run_pipeline
def load_b0_and_mask(b0_path, mask_path=""):
    """Load the B0 map and derive the brain mask the same way the page does."""
    img = nib.load(b0_path)
    b0 = img.get_fdata().astype(np.float64)
    if mask_path and os.path.isfile(mask_path):
        mask = nib.load(mask_path).get_fdata() > 0.5
    else:
        mask = b0 != 0
    mask &= np.isfinite(b0) & (np.abs(b0) > 1.0)
    return img, b0, mask


def format_currents_64ch(i_pfc, total_ch=64):
    """Dev-board text format: 64 values, 8 per line."""
    padded = np.zeros(total_ch)
    padded[:len(i_pfc)] = i_pfc
    return "\n".join(" ".join(f"{v:12.6f}" for v in padded[j:j + 8])
                     for j in range(0, total_ch, 8))
