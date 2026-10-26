"""
core/shim_solver.py  —  RobustL2ShimSolver + NIfTI channel loader + ROI registration
"""

# ─────────────────────────────────────────────────────────────────────────
# LEGACY / NOT USED BY CURRENT QT APP
#
# 全项目零 import。其中的 ANTs 配准与通道加载逻辑已分别由下列模块承担。
#
# 替代模块：core/roi_registry.py（ROI 配准）、core/pipeline.py::_load_coil_maps（线圈场图）
#
# 保留原因：路径与 import 兼容，且其中的实现细节仍有参考价值。修改这里不会影响
# 运行中的程序——先确认调用方是谁再动手。
# ─────────────────────────────────────────────────────────────────────────

import os, re
import numpy as np
import nibabel as nib
from scipy.optimize import minimize


# ══════════════════════════════════════════════════════════════════════
#  Solver
# ══════════════════════════════════════════════════════════════════════

class RobustL2ShimSolver:
    """
    目标函数:
        J(w) = ROI_MSE + roi_mean_weight * ROI_mean²
               + lambda_nonroi * nonROI_MSE
               + eta_reg * ||w||²

    残差模型: r = A_roi @ w + b  (b = 原始 B0，正方向=需要补偿)
    """

    def __init__(self,
                 lambda_nonroi=0.1,
                 eta_reg=1e-3,
                 roi_mean_weight=0.5,
                 use_roi_weights=True,
                 current_max_amp=4.0):
        self.lambda_nonroi  = float(lambda_nonroi)
        self.eta_reg        = float(eta_reg)
        self.roi_mean_weight= float(roi_mean_weight)
        self.use_roi_weights= bool(use_roi_weights)
        self.current_max_amp= float(current_max_amp)

    def _roi_weights(self, b):
        if not self.use_roi_weights:
            return np.ones(b.shape[0])
        absb = np.abs(b.astype(np.float64))
        return 0.5 + 0.5 * absb / (absb.max() + 1e-6)

    def _loss_grad(self, w, A_roi, b, A_nonroi, wv,
                   wAtA=None, wAtb=None, AtA_nr=None):
        """
        Compute loss and analytical gradient.

        If precomputed matrices (wAtA, wAtb, AtA_nr) are provided,
        uses them for O(C^2) gradient instead of O(M*C) matrix-vector products.
        """
        r    = A_roi @ w + b
        N    = r.size
        loss = np.mean(wv * r**2)
        if wAtA is not None:
            grad = (2.0 / N) * (wAtA @ w + wAtb)
        else:
            grad = (2.0 / N) * (A_roi.T @ (wv * r))

        if self.roi_mean_weight > 0:
            m     = r.mean()
            loss += self.roi_mean_weight * m**2
            grad += 2.0 * self.roi_mean_weight * m * A_roi.mean(axis=0)

        if A_nonroi is not None and A_nonroi.size > 0 and self.lambda_nonroi > 0:
            Nn    = A_nonroi.shape[0]
            if AtA_nr is not None:
                loss += self.lambda_nonroi * np.dot(w, AtA_nr @ w) / Nn
                grad += (2.0 * self.lambda_nonroi / Nn) * (AtA_nr @ w)
            else:
                rn    = A_nonroi @ w
                loss += self.lambda_nonroi * np.mean(rn**2)
                grad += (2.0 * self.lambda_nonroi / Nn) * (A_nonroi.T @ rn)

        loss += self.eta_reg * np.dot(w, w)
        grad += 2.0 * self.eta_reg * w
        return loss, grad

    def fit(self, A_roi, b, A_nonroi=None, w0=None, current_max_vec=None):
        A_roi = np.asarray(A_roi, dtype=np.float64)
        b     = np.asarray(b,     dtype=np.float64)
        if A_nonroi is not None:
            A_nonroi = np.asarray(A_nonroi, dtype=np.float64)
        n_ch = A_roi.shape[1]
        wv   = self._roi_weights(b)
        w0   = np.zeros(n_ch) if w0 is None else np.asarray(w0, np.float64)
        if current_max_vec is None:
            bounds = [(-self.current_max_amp, self.current_max_amp)] * n_ch
        else:
            bounds = [(-float(m), float(m)) for m in current_max_vec]

        # Precompute normal equation matrices for O(C^2) gradient
        wA = wv[:, np.newaxis] * A_roi
        wAtA = wA.T @ A_roi
        wAtb = wA.T @ b
        AtA_nr = A_nonroi.T @ A_nonroi if A_nonroi is not None else None

        res = minimize(lambda w: self._loss_grad(w, A_roi, b, A_nonroi, wv,
                                                  wAtA, wAtb, AtA_nr),
                       w0, method="L-BFGS-B", jac=True, bounds=bounds,
                       options={"maxiter": 800, "ftol": 1e-12, "gtol": 1e-8})
        return res.x, {"success": res.success, "message": res.message,
                        "n_iter": res.nit, "final_loss": float(res.fun)}

    @staticmethod
    def evaluate(B0_target, B0_array, w, roi_mask, brain_mask=None):
        """
        Returns (stats_dict, B0_shimmed_3d).
        B0_array: (X,Y,Z,C), w: (C,)
        """
        B0_target = np.asarray(B0_target, np.float64)
        B0_array  = np.asarray(B0_array,  np.float64)
        roi_mask  = roi_mask > 0.5
        brain_mask= (B0_target != 0) if brain_mask is None else (brain_mask > 0.5)

        B0_comp = np.tensordot(B0_array, w, axes=([3],[0]))
        B0_shim = B0_target + B0_comp

        def rmse(v,m): return float(np.sqrt(np.mean(v[m]**2)))
        def std (v,m): return float(np.std(v[m]))

        return {
            "roi_rmse_before":   rmse(B0_target, roi_mask),
            "roi_rmse_after":    rmse(B0_shim,   roi_mask),
            "roi_std_before":    std (B0_target, roi_mask),
            "roi_std_after":     std (B0_shim,   roi_mask),
            "brain_rmse_before": rmse(B0_target, brain_mask),
            "brain_rmse_after":  rmse(B0_shim,   brain_mask),
            "brain_std_before":  std (B0_target, brain_mask),
            "brain_std_after":   std (B0_shim,   brain_mask),
        }, B0_shim


# ══════════════════════════════════════════════════════════════════════
#  NIfTI channel loader
# ══════════════════════════════════════════════════════════════════════

from core.data_context import natural_key, resample_to_target


def batch_resample_channel_maps(root_folder, reference_nii,
                                n_channels=36, log_cb=None):
    """
    Step 1 of the channel-map pipeline.

    For each ch in 1..n_channels:
        root_folder/ch{N}_nii/ch{N}_nii_deltaFieldmap.nii[.gz]
            → resample to reference_nii space
            → save as root_folder/ch{N}_nii/ch{N}_nii_resampled_to_brain.nii.gz

    Parameters
    ----------
    root_folder   : str   parent directory containing ch1_nii/, ch2_nii/, ...
    reference_nii : str or nibabel image   reference space (target B0 map)
    n_channels    : int   how many channels to look for (default 36)
    log_cb        : callable(str) optional progress callback

    Returns
    -------
    resampled_paths : list of str   successfully written files
    skipped         : list of str   channels that were skipped/failed
    """
    import glob as _glob

    def _log(msg):
        if log_cb: log_cb(msg)

    if isinstance(reference_nii, str):
        ref_img = nib.load(reference_nii)
    else:
        ref_img = reference_nii
    ref_affine = ref_img.affine
    ref_shape  = ref_img.shape[:3]

    _log(f"Reference space: {ref_shape}  affine det={np.linalg.det(ref_affine[:3,:3]):.3f}")

    resampled_paths, skipped = [], []

    for ch in range(1, n_channels + 1):
        ch_dir = os.path.join(root_folder, f"ch{ch}_nii")
        if not os.path.isdir(ch_dir):
            _log(f"[WARN] ch{ch}: directory {ch_dir} not found -- skipped")
            skipped.append(f"ch{ch} (no dir)")
            continue

        # Find deltaFieldmap
        candidates = (
            _glob.glob(os.path.join(ch_dir, f"ch{ch}_nii_deltaFieldmap.nii.gz")) +
            _glob.glob(os.path.join(ch_dir, f"ch{ch}_nii_deltaFieldmap.nii"))   +
            _glob.glob(os.path.join(ch_dir, "*deltaFieldmap.nii*"))
        )
        if not candidates:
            _log(f"✗ ch{ch}: deltaFieldmap not found in {ch_dir} — skipped")
            skipped.append(f"ch{ch} (no deltaFieldmap)")
            continue

        delta_path = candidates[0]
        out_path   = os.path.join(ch_dir, f"ch{ch}_nii_resampled_to_brain.nii.gz")

        try:
            src_img    = nib.load(delta_path)
            resampled  = resample_to_target(
                src_img.get_fdata(dtype=np.float32), src_img.affine,
                ref_affine, ref_shape, order=1)
            nib.save(nib.Nifti1Image(resampled, ref_affine), out_path)
            resampled_paths.append(out_path)
            _log(f"✓ ch{ch}: {src_img.shape} → {ref_shape}  saved → {out_path}")

        except Exception as e:
            _log(f"✗ ch{ch}: resampling failed — {e}")
            skipped.append(f"ch{ch} ({e})")

    _log(f"Done. {len(resampled_paths)} channels resampled, {len(skipped)} skipped.")
    return resampled_paths, skipped


def _find_nifti_in_dir(dirpath, prefer_keyword=None, log_cb=None):
    """
    Find the best NIfTI file in a directory.
    If prefer_keyword given, prefer files containing that substring.
    Returns filepath or None.  Also logs all files found for diagnosis.
    """
    all_nii = sorted(
        [f for f in os.listdir(dirpath)
         if f.endswith('.nii.gz') or f.endswith('.nii')],
        key=natural_key)
    if log_cb:
        log_cb(f"  dir contents: {all_nii if all_nii else '(no .nii files)'}") 
    if not all_nii:
        return None
    if prefer_keyword:
        preferred = [f for f in all_nii if prefer_keyword in f]
        if preferred:
            return os.path.join(dirpath, preferred[0])
    return os.path.join(dirpath, all_nii[0])


def load_channel_niftis(folder, pattern="{ch}_nii_resampled_to_brain.nii.gz",
                        ref_affine=None, ref_shape=None, log_cb=None):
    """
    Load per-channel B0 response maps.

    Supported folder structures
    ---------------------------
    A) Sub-dir per channel  (primary, matches batch_resample output):
         folder/
           ch1_nii/   ch1_nii_resampled_to_brain.nii.gz
           ch2_nii/   ch2_nii_resampled_to_brain.nii.gz
           ...

    B) Flat NIfTI files:
         folder/
           ch1_resampled_to_brain.nii.gz
           ...

    Pattern tokens
    --------------
    {ch}     → sub-dir name with "_nii" stripped  (e.g. "ch3")
    {ch_dir} → raw sub-dir name                   (e.g. "ch3_nii")

    If ref_affine/ref_shape are given, each volume is resampled on load
    (safety net — ideally already done by batch_resample_channel_maps).
    """
    def _log(m):
        if log_cb: log_cb(m)

    try:
        all_entries = os.listdir(folder)
    except FileNotFoundError:
        raise FileNotFoundError(f"Folder not found: {folder}")

    _log(f"Scanning: {folder}  ({len(all_entries)} entries)")

    entries_dir = sorted(
        [e for e in all_entries if os.path.isdir(os.path.join(folder, e))],
        key=natural_key)
    entries_nii = sorted(
        [e for e in all_entries if e.endswith('.nii.gz') or e.endswith('.nii')],
        key=natural_key)

    _log(f"  sub-dirs: {len(entries_dir)}   flat nii: {len(entries_nii)}")

    B0_list, names = [], []

    if entries_dir:
        for ch_dir in entries_dir:
            ch_label = ch_dir.replace("_nii", "")   # ch3_nii → ch3

            # Try pattern with both token variants
            candidates = []
            for tok in [ch_label, ch_dir]:
                try:
                    fname = pattern.format(ch=tok)
                    fpath = os.path.join(folder, ch_dir, fname)
                    candidates.append(fpath)
                except KeyError:
                    pass

            fpath = None
            for c in candidates:
                if os.path.exists(c):
                    fpath = c
                    break

            if fpath is None:
                # Fallback: scan the sub-dir, prefer "resampled" files
                _log(f"  {ch_dir}: pattern not matched, scanning sub-dir…")
                fpath = _find_nifti_in_dir(
                    os.path.join(folder, ch_dir),
                    prefer_keyword="resampled",
                    log_cb=log_cb)

            if fpath is None or not os.path.exists(fpath):
                _log(f"  ✗ {ch_dir}: no NIfTI found — skipping")
                continue

            try:
                img  = nib.load(fpath)
                if ref_affine is not None and ref_shape is not None and img.shape[:3] != tuple(ref_shape):
                    _log(f"  {ch_dir}: shape {img.shape[:3]} ≠ {ref_shape}, resampling…")
                    data = resample_to_target(img.get_fdata(dtype=np.float32), img.affine, ref_affine, ref_shape)
                else:
                    data = img.get_fdata().astype(np.float32)
                B0_list.append(data)
                names.append(ch_label)
                _log(f"  ✓ {ch_dir}  [{fpath.split(os.sep)[-1]}]  shape={data.shape}")
            except Exception as e:
                _log(f"  ✗ {ch_dir}: load failed — {e}")
    else:
        _log("No sub-dirs found, loading flat NIfTI files…")
        for fname in entries_nii:
            fpath = os.path.join(folder, fname)
            try:
                img  = nib.load(fpath)
                if ref_affine is not None and ref_shape is not None and img.shape[:3] != tuple(ref_shape):
                    data = resample_to_target(img.get_fdata(dtype=np.float32), img.affine, ref_affine, ref_shape)
                else:
                    data = img.get_fdata().astype(np.float32)
                B0_list.append(data)
                names.append(os.path.splitext(os.path.splitext(fname)[0])[0])
                _log(f"  ✓ {fname}  shape={data.shape}")
            except Exception as e:
                _log(f"  ✗ {fname}: {e}")

    if not B0_list:
        raise FileNotFoundError(
            f"No channel NIfTIs could be loaded from {folder}\n"
            f"Sub-dirs found: {entries_dir[:8]}\n"
            f"Flat files found: {entries_nii[:8]}")
    _log(f"Loaded {len(B0_list)} channels successfully.")
    B0_array = np.stack(B0_list, axis=-1)
    return B0_array, names


def load_channel_niftis_flat(folder, pattern="{ch}_resampled_to_brain.nii.gz",
                              brain_mask=None, ref_affine=None, ref_shape=None,
                              log_cb=None):
    """
    Load channels, optionally resample to ref space, return flattened A matrix.
    ref_affine / ref_shape: pass the target B0 affine & shape to auto-resample.
    """
    B0_array, names = load_channel_niftis(folder, pattern,
                                          ref_affine=ref_affine,
                                          ref_shape=ref_shape,
                                          log_cb=log_cb)
    shape3 = B0_array.shape[:3]
    if brain_mask is None:
        brain_mask = (np.abs(B0_array[..., 0]) > 1e-6)
    else:
        brain_mask = np.asarray(brain_mask, bool)
    idx = np.where(brain_mask.ravel())[0]
    A   = B0_array.reshape(-1, B0_array.shape[-1])[idx]
    return A, idx, shape3, names, B0_array


# ══════════════════════════════════════════════════════════════════════
#  ROI registration helpers (wraps fsl_replace.py fallbacks)
# ══════════════════════════════════════════════════════════════════════

def _ants_register_subprocess(fixed_path, moving_path, mask_path,
                               out_reg, out_mask, log_cb=None):
    """
    Run ANTsPy registration in a subprocess using D:\\python.exe
    (which has a compatible scipy + antspyx).
    Returns True on success, False on failure.
    """
    import subprocess, json, tempfile

    def _log(msg):
        if log_cb:
            log_cb(msg)

    # Find a Python with working ANTsPy
    candidates = [
        r"D:\python.exe",
        r"D:\Python\python.exe",
    ]
    py_exe = None
    for c in candidates:
        if os.path.isfile(c):
            py_exe = c
            break
    if py_exe is None:
        _log("  subprocess: no D:\\python.exe found")
        return False

    # Test if this Python can import ants
    test = subprocess.run([py_exe, "-c", "import ants; print('ok')"],
                          capture_output=True, text=True, timeout=30)
    if test.returncode != 0 or "ok" not in test.stdout:
        _log(f"  subprocess: {py_exe} cannot import ants: {test.stderr[:200]}")
        return False

    _log(f"  subprocess: using {py_exe} for ANTsPy registration")

    # Write a small registration script
    script = f'''
import ants, json, sys
fixed  = ants.image_read(r"{fixed_path}")
moving = ants.image_read(r"{moving_path}")
mask   = ants.image_read(r"{mask_path}")
reg = ants.registration(
    fixed=fixed, moving=moving,
    type_of_transform="Affine",
    aff_metric="mattes", aff_sampling=32,
    aff_iterations=(2100, 1200, 1200, 100),
    aff_shrink_factors=(6, 4, 2, 1),
    aff_smoothing_sigmas=(3, 2, 1, 0),
    verbose=False,
)
ants.image_write(reg["warpedmovout"], r"{out_reg}")
mask_nat = ants.apply_transforms(
    fixed=fixed, moving=mask,
    transformlist=reg["fwdtransforms"],
    interpolator="nearestNeighbor",
)
ants.image_write(mask_nat, r"{out_mask}")
n = int((mask_nat.numpy() > 0.5).sum())
print(json.dumps({{"status": "ok", "n_roi": n}}))
'''
    script_path = os.path.join(tempfile.gettempdir(), "_shimlab_ants_reg.py")
    with open(script_path, "w") as f:
        f.write(script)

    _log("  subprocess: running ANTsPy registration…")
    proc = subprocess.run(
        [py_exe, script_path],
        capture_output=True, text=True, timeout=300,
    )

    if proc.returncode != 0:
        _log(f"  subprocess: FAILED — {proc.stderr[:300]}")
        return False

    # Parse result
    for line in proc.stdout.strip().split("\n"):
        line = line.strip()
        if line.startswith("{"):
            try:
                result = json.loads(line)
                if result.get("status") == "ok":
                    _log(f"  subprocess: OK — {result.get('n_roi', '?')} ROI voxels")
                    return True
            except json.JSONDecodeError:
                pass

    _log(f"  subprocess: unexpected output — {proc.stdout[:200]}")
    return False


def register_mni_to_native(mag_brain_path, mni_t1_path, mni_mask_path,
                            out_dir, log_cb=None):
    """
    MNI → native space registration.

    Priority:
      1. ANTsPy in-process
      2. ANTsPy via subprocess (D:\\python.exe)
      3. Pure-Python FLIRT fallback

    Returns path to native-space ROI mask.
    """
    import os

    def _log(msg):
        if log_cb:
            log_cb(msg)

    os.makedirs(out_dir, exist_ok=True)
    mni_in_native = os.path.join(out_dir, "registered_MNI152.nii.gz")
    mask_native   = os.path.join(out_dir, "roi_mask_native.nii.gz")

    # ── Try ANTsPy in-process ────────────────────────────────────
    try:
        import ants
        _log("Step 1/2  ANTsPy: MNI T1 → native mag (Affine, dof=12)…")
        fixed  = ants.image_read(str(mag_brain_path))
        moving = ants.image_read(str(mni_t1_path))
        mask_mni = ants.image_read(str(mni_mask_path))

        reg = ants.registration(
            fixed=fixed, moving=moving,
            type_of_transform="Affine",
            aff_metric="mattes", aff_sampling=32,
            aff_iterations=(2100, 1200, 1200, 100),
            aff_shrink_factors=(6, 4, 2, 1),
            aff_smoothing_sigmas=(3, 2, 1, 0),
            verbose=False,
        )
        ants.image_write(reg["warpedmovout"], mni_in_native)
        _log("Step 2/2  ANTsPy: applying transform to ROI mask…")
        mask_native_ants = ants.apply_transforms(
            fixed=fixed, moving=mask_mni,
            transformlist=reg["fwdtransforms"],
            interpolator="nearestNeighbor",
        )
        ants.image_write(mask_native_ants, mask_native)
        n_roi = int((mask_native_ants.numpy() > 0.5).sum())
        _log(f"  ROI mask → {mask_native}  ({n_roi:,} voxels)")
        return mask_native
    except Exception as e:
        _log(f"  ANTsPy in-process failed: {e}")

    # ── Try ANTsPy via subprocess (D:\python.exe) ────────────────
    _log("Trying ANTsPy via subprocess…")
    if _ants_register_subprocess(
        str(mag_brain_path), str(mni_t1_path), str(mni_mask_path),
        mni_in_native, mask_native, log_cb=log_cb
    ):
        return mask_native

    # ── Fallback: pure-Python FLIRT ──────────────────────────────
    _log("Falling back to scipy FLIRT (basic quality)…")
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from core.fsl_replace import flirt as _flirt_py

    xfm_matrix = os.path.join(out_dir, "mni2native.mat")

    _log("Step 1/2  FLIRT: MNI T1 → native mag (dof=12)…")
    _flirt_py(mni_t1_path, mag_brain_path,
              out=mni_in_native, omat=xfm_matrix,
              dof=12, cost="mutualinfo", log_cb=log_cb)

    _log("Step 2/2  FLIRT: applying transform to ROI mask…")
    _flirt_py(mni_mask_path, mag_brain_path,
              out=mask_native, init=xfm_matrix,
              applyxfm=True, interp="nearestneighbour", log_cb=log_cb)

    _log(f"ROI mask saved → {mask_native}")
    return mask_native
