"""
core/fsl_replace.py
═══════════════════
Drop-in replacements for FSL tools, prioritising high-quality backends:

  bet()      — Brain Extraction
               Priority: HD-BET → SynthStrip → Otsu fallback
  fslmaths() — Chainable NIfTI arithmetic (pure Python, unchanged)
  flirt()    — Co-registration
               Priority: ANTsPy (in-process) → ANTsPy (subprocess) → scipy fallback

Install recommended backends:
    pip install hd-bet          # best BET
    pip install antspyx         # best registration
"""

from __future__ import annotations
import os
import numpy as np
import nibabel as nib
from pathlib import Path
from scipy import ndimage
from scipy.optimize import minimize
import warnings


# ══════════════════════════════════════════════════════════════════
#  Helpers
# ══════════════════════════════════════════════════════════════════

def _load(path: str) -> tuple[np.ndarray, np.ndarray, nib.Nifti1Header]:
    img = nib.load(str(path))
    data = np.asanyarray(img.dataobj, dtype=np.float32)
    return data, img.affine.copy(), img.header.copy()


def _save(data: np.ndarray, affine: np.ndarray,
          header: nib.Nifti1Header, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(data.astype(np.float32), affine, header), str(path))


# ══════════════════════════════════════════════════════════════════
#  BET — Brain Extraction Tool
#
#  Priority: HD-BET → SynthStrip → Otsu morphological fallback
# ══════════════════════════════════════════════════════════════════

def _bet_hdbet(input_file, output_file, _log):
    """HD-BET: deep-learning brain extraction (GPU or CPU)."""
    from HD_BET.run import run_hd_bet

    _log("  bet: running HD-BET...")

    # HD-BET writes to output directory
    out_dir = str(Path(output_file).parent)
    out_stem = str(Path(output_file).stem).replace(".nii", "")
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    # run_hd_bet returns (brain_path, mask_path)
    # mode='fast' uses only one fold (faster, slightly less accurate)
    # device='cpu' for compatibility; set to 0 for GPU
    try:
        import torch
        device = 0 if torch.cuda.is_available() else "cpu"
    except ImportError:
        device = "cpu"

    brain_path, mask_path = run_hd_bet(
        input_file, output_file,
        mode="fast",
        device=device,
        postprocess=True,
        do_tta=False,
    )
    _log(f"  bet: HD-BET complete → {brain_path}")

    # Load result for mask output
    img = nib.load(str(mask_path))
    mask_data = np.asanyarray(img.dataobj).astype(np.float32)
    return mask_data, img.affine.copy(), img.header.copy()


def _bet_synthstrip(input_file, output_file, _log):
    """SynthStrip: FreeSurfer's robust DL skull-stripping."""
    import subprocess
    import shutil

    # Try to find synthstrip command
    synthstrip_cmd = shutil.which("mri_synthstrip")
    if synthstrip_cmd is None:
        # Try via python module
        try:
            from synthstrip import predict as ss_predict
            _log("  bet: using SynthStrip Python module...")

            img = nib.load(str(input_file))
            data = np.asanyarray(img.dataobj, dtype=np.float32)

            # SynthStrip predict returns (brain, mask)
            brain, mask = ss_predict(data, img.affine)

            Path(output_file).parent.mkdir(parents=True, exist_ok=True)
            nib.save(nib.Nifti1Image(brain, img.affine, img.header), str(output_file))
            _log(f"  bet: SynthStrip complete → {output_file}")
            return mask.astype(np.float32), img.affine.copy(), img.header.copy()
        except ImportError:
            raise ImportError("SynthStrip not available")

    # Use command-line SynthStrip
    _log("  bet: running mri_synthstrip...")
    mask_file = str(output_file).replace(".nii", "_mask.nii")
    result = subprocess.run(
        [synthstrip_cmd, "-i", str(input_file), "-o", str(output_file), "-m", mask_file],
        capture_output=True, text=True, timeout=300,
    )
    if result.returncode != 0:
        raise RuntimeError(f"SynthStrip failed: {result.stderr[:200]}")

    _log(f"  bet: SynthStrip complete → {output_file}")
    img = nib.load(mask_file)
    mask_data = np.asanyarray(img.dataobj).astype(np.float32)
    return mask_data, img.affine.copy(), img.header.copy()


def _bet_otsu(input_file, output_file, f, m, robust):
    """Otsu thresholding + morphological cleanup (always available)."""
    from skimage.filters import threshold_otsu
    from skimage.morphology import binary_closing, ball

    data, affine, header = _load(input_file)

    flat = data[data > 0]
    if flat.size == 0:
        raise ValueError("Input volume appears to be all-zero.")

    otsu_thresh = threshold_otsu(flat)
    thresh = otsu_thresh * (0.5 + f)
    mask = data > thresh

    # Largest connected component
    labelled, n_labels = ndimage.label(mask)
    if n_labels == 0:
        raise RuntimeError("BET: no connected component found.")
    sizes = ndimage.sum(mask, labelled, range(1, n_labels + 1))
    largest = int(np.argmax(sizes)) + 1
    mask = labelled == largest

    # Morphological closing
    if robust:
        radius = max(1, int(round(3.0 / np.abs(np.diag(affine[:3, :3])).mean())))
        mask = binary_closing(mask, ball(radius))

    mask_f32 = mask.astype(np.float32)
    brain = data * mask_f32
    _save(brain, affine, header, output_file)

    if m:
        _save_mask(mask_f32, affine, header, output_file)


def _save_mask(mask_data, affine, header, output_file):
    """Save binary mask alongside brain-extracted output."""
    stem = str(output_file)
    for ext in (".nii.gz", ".nii"):
        if stem.endswith(ext):
            stem = stem[:-len(ext)]
            break
    mask_path = stem + "_mask.nii.gz"
    _save(mask_data, affine, header, mask_path)


# ══════════════════════════════════════════════════════════════════
#  fslmaths — chainable arithmetic on NIfTI volumes
# ══════════════════════════════════════════════════════════════════

class fslmaths:
    """
    Chainable NIfTI arithmetic, drop-in for fsl.wrappers.fslmaths.

    Usage:
        fslmaths(input).mul(mask).run(output)
        fslmaths(input).div(2).div(np.pi).div(delta_TE).run(output)
    """

    def __init__(self, input_file: str):
        self._data, self._affine, self._header = _load(input_file)
        self._ops: list = []

    def _resolve(self, x) -> np.ndarray | float:
        if isinstance(x, (int, float, np.floating, np.integer)):
            return float(x)
        d, _, _ = _load(str(x))
        unique = np.unique(d[np.isfinite(d)])
        if len(unique) <= 3 and d.max() > 1.5:
            d = (d > 0).astype(np.float32)
        return d

    def mul(self, x):   self._ops.append(("mul", x));  return self
    def div(self, x):   self._ops.append(("div", x));  return self
    def add(self, x):   self._ops.append(("add", x));  return self
    def sub(self, x):   self._ops.append(("sub", x));  return self
    def abs(self):      self._ops.append(("abs", None)); return self
    def bin(self):      self._ops.append(("bin", None)); return self
    def thr(self, x):   self._ops.append(("thr", x));  return self
    def uthr(self, x):  self._ops.append(("uthr", x)); return self

    def run(self, output_file: str) -> None:
        result = self._data.copy()
        for op, x in self._ops:
            v = self._resolve(x) if x is not None else None
            if   op == "mul":  result = result * v
            elif op == "div":
                if np.isscalar(v):
                    if v == 0:
                        raise ZeroDivisionError("fslmaths.div: divisor is zero")
                    result = result / v
                else:
                    with np.errstate(divide="ignore", invalid="ignore"):
                        result = np.where(v != 0, result / v, 0.0)
            elif op == "add":  result = result + v
            elif op == "sub":  result = result - v
            elif op == "abs":  result = np.abs(result)
            elif op == "bin":  result = (result > 0).astype(np.float32)
            elif op == "thr":  result = np.where(result >= v, result, 0.0)
            elif op == "uthr": result = np.where(result <= v, result, 0.0)
        _save(result, self._affine, self._header, output_file)


# ══════════════════════════════════════════════════════════════════
#  FLIRT — Co-registration
#
#  Priority: ANTsPy (in-process) → ANTsPy (subprocess) → scipy
# ══════════════════════════════════════════════════════════════════

def flirt(src: str,
          ref: str,
          out: str | None = None,
          omat: str | None = None,
          applyxfm: bool = False,
          init: str | None = None,
          dof: int = 6,
          cost: str = "normmi",
          interp: str = "trilinear",
          log_cb=None) -> None:
    """
    Co-registration using ANTsPy as primary backend.

    Parameters
    ----------
    src    : moving image path
    ref    : fixed/reference image path
    out    : output warped image path
    omat   : output 4x4 matrix path
    applyxfm : apply existing transform (init required)
    init   : initial transform matrix path
    dof    : degrees of freedom (6=rigid, 7=similarity, 12=affine)
    interp : 'trilinear' or 'nearestneighbour'
    log_cb : logging callback
    """
    def _log(msg):
        if log_cb:
            log_cb(msg)

    # ── Apply-only mode ──────────────────────────────────────────
    if applyxfm:
        _apply_existing_transform(src, ref, out, init, interp, _log)
        return

    # ── Try ANTsPy in-process ────────────────────────────────────
    try:
        _flirt_antspy(src, ref, out, omat, dof, interp, _log)
        return
    except ImportError as e:
        _log(f"  flirt: ANTsPy not available: {e}")
    except Exception as e:
        _log(f"  flirt: ANTsPy failed: {e}")

    # ── Try ANTsPy subprocess ────────────────────────────────────
    try:
        if _flirt_antspy_subprocess(src, ref, out, omat, dof, interp, _log):
            return
    except Exception as e:
        _log(f"  flirt: ANTsPy subprocess failed: {e}")

    # ── Scipy fallback (basic quality) ───────────────────────────
    _log("  flirt: WARNING — falling back to scipy (install antspyx for better results)")
    _flirt_scipy_fallback(src, ref, out, omat, dof, interp, _log)


# ─────────────────────────────────────────────────────────────────
#  ANTsPy Backend (recommended)
# ─────────────────────────────────────────────────────────────────

def _flirt_antspy(src, ref, out, omat, dof, interp, _log):
    """ANTsPy in-process registration."""
    # Patch scipy compatibility if needed
    _patch_scipy_for_ants()

    import ants

    # Reads and writes go through ants_io: ITK opens the filename with the
    # narrow Win32 API, so a Chinese subject folder fails outright.
    from core.ants_io import image_read as _ants_read, image_write as _ants_write

    _log(f"  flirt: using ANTsPy (dof={dof})")

    fixed = _ants_read(ref)
    moving = _ants_read(src)

    # Transform type from DOF
    if dof <= 6:
        tx_type = "Rigid"
    elif dof <= 7:
        tx_type = "Similarity"
    else:
        tx_type = "Affine"

    _log(f"  flirt: ANTs registration ({tx_type})...")
    result = ants.registration(
        fixed=fixed,
        moving=moving,
        type_of_transform=tx_type,
        aff_metric="mattes",
        aff_sampling=32,
        aff_iterations=(2100, 1200, 1200, 100),
        aff_shrink_factors=(6, 4, 2, 1),
        aff_smoothing_sigmas=(3, 2, 1, 0),
        verbose=False,
    )
    _log("  flirt: ANTs registration complete")

    # Save warped output
    if out:
        nn = "nearest" in interp.lower()
        if nn:
            warped = ants.apply_transforms(
                fixed=fixed, moving=moving,
                transformlist=result["fwdtransforms"],
                interpolator="nearestNeighbor",
            )
        else:
            warped = result["warpedmovout"]
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        _ants_write(warped, out)
        _log(f"  flirt: output → {out}")

    # Save matrix
    if omat:
        tx_path = result["fwdtransforms"][0]
        M = _ants_mat_to_numpy(tx_path)
        Path(omat).parent.mkdir(parents=True, exist_ok=True)
        np.savetxt(str(omat), M, fmt="%.10f")
        # Also save ANTs native transform path for applyxfm
        with open(str(omat) + ".ants_xfm", "w") as f:
            f.write("\n".join(result["fwdtransforms"]))
        _log(f"  flirt: matrix → {omat}")


def _flirt_antspy_subprocess(src, ref, out, omat, dof, interp, _log):
    """Run ANTsPy via subprocess (for environments where ants can't import in-process)."""
    py_exe = _find_ants_python()
    if not py_exe:
        _log("  flirt: no Python with ANTsPy found for subprocess")
        return False

    _log(f"  flirt: subprocess using {py_exe}")

    tx_type = "Rigid" if dof <= 6 else ("Similarity" if dof <= 7 else "Affine")
    nn = "nearest" in interp.lower() if isinstance(interp, str) else False

    # The child interpreter has its own sys.path and may not be able to import
    # this package, so staging has to happen here: by the time the paths are
    # baked into the generated source they must already be ASCII.
    from core.ants_io import ascii_inputs, ascii_output

    with ascii_inputs(ref, src) as (ref_p, src_p), ascii_output(out) as out_p:
        return _run_ants_flirt_script(py_exe, src_p, ref_p, out_p,
                                      str(omat) if omat else "",
                                      tx_type, nn, _log)


def _run_ants_flirt_script(py_exe, src, ref, out_path, omat_path,
                           tx_type, nn, _log) -> bool:
    """Write and run the ANTs registration script. Every path must be ASCII."""
    import subprocess, tempfile

    script = f'''
import ants, json, shutil, os
fixed  = ants.image_read(r"{ref}")
moving = ants.image_read(r"{src}")
reg = ants.registration(
    fixed=fixed, moving=moving,
    type_of_transform="{tx_type}",
    aff_metric="mattes",
    aff_sampling=32,
    aff_iterations=(2100, 1200, 1200, 100),
    aff_shrink_factors=(6, 4, 2, 1),
    aff_smoothing_sigmas=(3, 2, 1, 0),
    verbose=False,
)
if r"{out_path}":
    if {nn}:
        warped = ants.apply_transforms(
            fixed=fixed, moving=moving,
            transformlist=reg["fwdtransforms"],
            interpolator="nearestNeighbor")
    else:
        warped = reg["warpedmovout"]
    ants.image_write(warped, r"{out_path}")
if r"{omat_path}":
    import numpy as np
    tx_path = reg["fwdtransforms"][0]
    tx = ants.read_transform(tx_path)
    with open(r"{omat_path}" + ".ants_xfm", "w") as f:
        f.write("\\n".join(reg["fwdtransforms"]))
print(json.dumps({{"status": "ok", "type": "{tx_type}"}}))
'''
    script_file = os.path.join(tempfile.gettempdir(), "_shimlab_flirt_sub.py")
    with open(script_file, "w", encoding="utf-8") as f:
        f.write(script)

    proc = subprocess.run([py_exe, script_file],
                          capture_output=True, text=True, timeout=300)
    if proc.returncode == 0 and '"ok"' in proc.stdout:
        _log("  flirt: subprocess ANTsPy OK")
        return True

    _log(f"  flirt: subprocess failed: {proc.stderr[:200]}")
    return False


def _apply_existing_transform(src, ref, out, init, interp, _log):
    """Apply an existing transform (ANTs sidecar or 4x4 matrix)."""
    if init is None:
        raise ValueError("flirt applyxfm requires init= matrix path")

    # Check for ANTs sidecar
    ants_xfm_path = str(init) + ".ants_xfm"
    if os.path.exists(ants_xfm_path):
        try:
            _patch_scipy_for_ants()
            import ants
            from core.ants_io import (image_read as _ants_read,
                                      image_write as _ants_write)
            with open(ants_xfm_path, "r") as f:
                xfm_paths = [l.strip() for l in f if l.strip()]
            # Verify all transform files exist
            if all(os.path.exists(p) for p in xfm_paths):
                fixed = _ants_read(ref)
                moving = _ants_read(src)
                nn = "nearest" in interp.lower()
                warped = ants.apply_transforms(
                    fixed=fixed, moving=moving,
                    transformlist=xfm_paths,
                    interpolator="nearestNeighbor" if nn else "linear",
                )
                if out:
                    Path(out).parent.mkdir(parents=True, exist_ok=True)
                    _ants_write(warped, out)
                    _log(f"  flirt: applied ANTs transform → {out}")
                return
        except ImportError:
            pass
        except Exception as e:
            _log(f"  flirt: ANTs apply failed: {e}, trying matrix fallback")

    # Fallback: load 4x4 matrix and resample
    _log(f"  flirt: applying 4x4 matrix from {init}")
    M = np.loadtxt(str(init))
    src_data, src_aff, _ = _load(src)
    ref_data, ref_aff, ref_hdr = _load(ref)
    warped = _apply_affine(src_data, src_aff, ref_data.shape, ref_aff, M, interp)
    if out:
        _save(warped, ref_aff, ref_hdr, out)
        _log(f"  flirt: saved → {out}")


# ─────────────────────────────────────────────────────────────────
#  Scipy Fallback (basic quality, always available)
# ─────────────────────────────────────────────────────────────────

def _flirt_scipy_fallback(src, ref, out, omat, dof, interp, _log):
    """Minimal rigid registration using NMI + Powell optimizer."""
    src_data, src_aff, src_hdr = _load(src)
    ref_data, ref_aff, ref_hdr = _load(ref)

    _log(f"  flirt: scipy — src={src_data.shape}, ref={ref_data.shape}")

    # Centre-of-mass alignment as starting point
    M_init = _centre_align(src_data, src_aff, ref_data, ref_aff)
    p0 = _mat_to_params(M_init, min(dof, 6))

    # Downsample for speed
    factor = 2 if min(src_data.shape) > 64 else 1
    src_ds, src_ds_aff = _downsample(src_data, src_aff, factor)
    ref_ds, ref_ds_aff = _downsample(ref_data, ref_aff, factor)

    def _cost(params):
        M = _params_to_mat(params, min(dof, 6), src_ds_aff, ref_ds_aff)
        img = _apply_affine(src_ds, src_ds_aff, ref_ds.shape, ref_ds_aff, M, "nearest")
        return _nmi_cost(img, ref_ds)

    result = minimize(_cost, p0, method="Powell",
                      options={"maxiter": 800, "ftol": 1e-7})
    M_opt = _params_to_mat(result.x, min(dof, 6), src_aff, ref_aff)

    _log(f"  flirt: scipy done, {result.nfev} evals")

    if omat:
        Path(omat).parent.mkdir(parents=True, exist_ok=True)
        np.savetxt(str(omat), M_opt, fmt="%.10f")
    if out:
        warped = _apply_affine(src_data, src_aff, ref_data.shape, ref_aff, M_opt, interp)
        _save(warped, ref_aff, ref_hdr, out)


# ─────────────────────────────────────────────────────────────────
#  Shared utilities
# ─────────────────────────────────────────────────────────────────

def _patch_scipy_for_ants():
    """Patch scipy compatibility for older ANTsPy builds."""
    try:
        from scipy._lib._util import _lazywhere  # noqa: F401
    except ImportError:
        import scipy._lib._util as _scu
        def _lazywhere(cond, arrays, f, fillvalue=0, f2=None):
            cond = np.asarray(cond)
            out = np.full(cond.shape, fill_value=fillvalue,
                          dtype=np.result_type(*[np.asarray(a).dtype for a in arrays]))
            if cond.any():
                args = [np.extract(cond, a) for a in arrays]
                out[cond] = f(*args)
            if f2 is not None and (~cond).any():
                args = [np.extract(~cond, a) for a in arrays]
                out[~cond] = f2(*args)
            return out
        _scu._lazywhere = _lazywhere


def _find_ants_python():
    """Find a Python executable that can import ants."""
    import subprocess
    candidates = [
        r"D:\python.exe", r"D:\Python\python.exe",
        r"C:\Python311\python.exe", r"C:\Python312\python.exe",
    ]
    for c in candidates:
        if os.path.isfile(c):
            try:
                test = subprocess.run(
                    [c, "-c", "import ants; print('ok')"],
                    capture_output=True, text=True, timeout=30)
                if test.returncode == 0 and "ok" in test.stdout:
                    return c
            except Exception:
                pass
    return None


def _ants_mat_to_numpy(ants_mat_path: str) -> np.ndarray:
    """Convert ANTs .mat transform to 4x4 world-space matrix (RAS)."""
    try:
        import ants
        tx = ants.read_transform(ants_mat_path)
        params = tx.parameters
        center = np.array(tx.fixed_parameters)

        if len(params) == 12:
            A = np.array(params[:9]).reshape(3, 3)
            t = np.array(params[9:12])
        elif len(params) == 6:
            A = np.eye(3)
            t = np.array(params[3:6])
        else:
            return np.eye(4)

        offset = t + center - A @ center
        # LPS→RAS
        lps2ras = np.diag([-1.0, -1.0, 1.0])
        A_ras = lps2ras @ A @ lps2ras
        offset_ras = lps2ras @ offset

        M = np.eye(4)
        M[:3, :3] = A_ras
        M[:3, 3] = offset_ras
        return M
    except Exception:
        return np.eye(4)


def _centre_align(src_data, src_aff, ref_data, ref_aff) -> np.ndarray:
    """4x4 matrix aligning volume centres (translation only)."""
    sc = np.array(src_data.shape[:3]) / 2.0
    rc = np.array(ref_data.shape[:3]) / 2.0
    sw = (src_aff @ np.append(sc, 1))[:3]
    rw = (ref_aff @ np.append(rc, 1))[:3]
    M = np.eye(4)
    M[:3, 3] = sw - rw
    return M


def _downsample(data, affine, factor):
    """Downsample volume by integer factor."""
    if factor <= 1:
        return data, affine
    from scipy.ndimage import zoom as _zoom
    down = _zoom(data, 1.0 / factor, order=1, mode='nearest')
    new_aff = affine.copy()
    new_aff[:3, :3] *= factor
    shift = (factor - 1) / 2.0
    new_aff[:3, 3] += affine[:3, :3] @ np.array([shift, shift, shift])
    return down.astype(np.float32), new_aff


def _mat_to_params(M: np.ndarray, dof: int) -> np.ndarray:
    """Decompose 4x4 into [tx,ty,tz,rx,ry,rz]."""
    tx, ty, tz = M[:3, 3]
    R = M[:3, :3]
    sy = np.sqrt(R[0, 0]**2 + R[1, 0]**2)
    if sy > 1e-6:
        rx = np.arctan2(R[2, 1], R[2, 2])
        ry = np.arctan2(-R[2, 0], sy)
        rz = np.arctan2(R[1, 0], R[0, 0])
    else:
        rx = np.arctan2(-R[1, 2], R[1, 1])
        ry = np.arctan2(-R[2, 0], sy)
        rz = 0.0
    return np.array([tx, ty, tz, rx, ry, rz])


def _params_to_mat(p: np.ndarray, dof: int,
                   src_aff: np.ndarray, ref_aff: np.ndarray) -> np.ndarray:
    """Build 4x4 from [tx,ty,tz,rx,ry,rz]."""
    tx, ty, tz, rx, ry, rz = p[:6]
    Rx = np.array([[1, 0, 0], [0, np.cos(rx), -np.sin(rx)], [0, np.sin(rx), np.cos(rx)]])
    Ry = np.array([[np.cos(ry), 0, np.sin(ry)], [0, 1, 0], [-np.sin(ry), 0, np.cos(ry)]])
    Rz = np.array([[np.cos(rz), -np.sin(rz), 0], [np.sin(rz), np.cos(rz), 0], [0, 0, 1]])
    R = Rz @ Ry @ Rx
    M = np.eye(4)
    M[:3, :3] = R
    M[:3, 3] = [tx, ty, tz]
    return M


def _apply_affine(src_data, src_aff, out_shape, ref_aff, world_M, interp="trilinear"):
    """Resample src into ref space via world-space transform."""
    vox_M = np.linalg.inv(src_aff) @ world_M @ ref_aff
    order = 1 if "trilinear" in interp else 0
    warped = ndimage.affine_transform(
        src_data,
        matrix=vox_M[:3, :3],
        offset=vox_M[:3, 3],
        output_shape=out_shape[:3],
        order=order,
        mode="constant",
        cval=0.0
    )
    return warped.astype(np.float32)


def _nmi_cost(img, ref, bins=128):
    """Negative NMI (lower = better alignment)."""
    mask = (img != 0) & (ref != 0)
    n = mask.sum()
    if n < 50:
        ref_nz = max((ref != 0).sum(), 1)
        return 2.0 - (n / ref_nz)

    a = img[mask].ravel()
    b = ref[mask].ravel()
    a = np.clip(a, *np.percentile(a, [0.5, 99.5]))
    b = np.clip(b, *np.percentile(b, [0.5, 99.5]))

    hist2d, _, _ = np.histogram2d(a, b, bins=bins)
    from scipy.ndimage import gaussian_filter
    hist2d = gaussian_filter(hist2d, sigma=0.5)
    hist2d = hist2d / hist2d.sum()

    pA = hist2d.sum(axis=1); pA = pA[pA > 0]
    pB = hist2d.sum(axis=0); pB = pB[pB > 0]
    pAB = hist2d[hist2d > 0]

    HA = -np.sum(pA * np.log(pA))
    HB = -np.sum(pB * np.log(pB))
    HAB = -np.sum(pAB * np.log(pAB))

    if HAB == 0:
        return 1.0
    return -(HA + HB) / HAB
