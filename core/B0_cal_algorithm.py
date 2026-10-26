"""
core/B0_cal_algorithm.py
========================
B0 map computation pipeline — FSL-free edition.
Replaces:  fsl.wrappers.{bet, flirt, fslmaths}  with pure-Python equivalents.
DICOM → NIfTI via core.dicom_to_nifti (pure pydicom, no external binary).
"""

import os, sys, glob, gzip, shutil, argparse
import numpy as np
import nibabel as nib
from skimage.restoration import unwrap_phase

# ── FSL-free replacements ─────────────────────────────────────────────────
from core.fsl_replace import flirt, fslmaths
from core.dicom_to_nifti import dicom_folder_to_nifti


def _safe_print(msg) -> None:
    """print() that can never abort the caller.

    A GBK console cannot encode the check marks these logs use, and a bare
    print inside an `except` block turns a real error into a
    UnicodeEncodeError -- losing the actual reason the step failed. Logging
    must never be able to fail the computation it describes.
    """
    try:
        print(msg)
    except UnicodeEncodeError:
        try:
            enc = getattr(sys.stdout, "encoding", None) or "ascii"
            print(str(msg).encode(enc, "replace").decode(enc, "replace"))
        except Exception:
            pass
    except Exception:
        pass


def bet(input_path: str, output_path: str, f: float = 0.5, m: bool = False, **kw):
    """
    Brain extraction.
    hd-bet correct CLI: hd-bet -i <input> -o <output> [--save_bet_mask]
    """
    import subprocess, sys, os, shutil

    def _try_cmd(cmd):
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode == 0:
            return True
        print(f"[bet] failed (code {r.returncode}): {(r.stderr or r.stdout)[:400]}")
        return False

    def _find_hdbet():
        w = shutil.which("hd-bet")
        if w:
            return w
        _localappdata = os.environ.get("LOCALAPPDATA", "")
        for _pydir in [
            os.path.dirname(sys.executable),
            os.path.join(_localappdata, "Programs", "Python", "Python311"),
            os.path.join(_localappdata, "Programs", "Python", "Python312"),
            r"D:\Python311", r"D:\Python312", "D:\\",
            r"C:\Python311", r"C:\Python312",
        ]:
            for name in ("hd-bet.exe", "hd-bet.EXE", "hd-bet"):
                p = os.path.join(_pydir, "Scripts", name)
                if os.path.isfile(p):
                    return p
        return None

    hdbet = _find_hdbet()
    if hdbet:
        cmd = [hdbet, "-i", input_path, "-o", output_path,
               "-device", "cpu", "--disable_tta"]
        if m:
            cmd.append("--save_bet_mask")
        print(f"[bet] running (CPU): {hdbet}")
        if _try_cmd(cmd):
            print("[bet] HD-BET succeeded.")
            return

    bet_bin = shutil.which("bet") or shutil.which("bet2")
    if bet_bin:
        cmd = [bet_bin, input_path, output_path, "-f", str(f)]
        if m:
            cmd.append("-m")
        if _try_cmd(cmd):
            print("[bet] FSL BET succeeded.")
            return

    raise RuntimeError(
        "脑部提取失败：未找到 hd-bet 或 FSL bet。\n"
        f"sys.executable = {sys.executable}\n"
        "请确认：python -m pip install hd-bet"
    )


def _glob1(folder: str, *patterns) -> str | None:
    """
    Try multiple glob patterns in order; return first match or None.
    Case-insensitive on the filename part.
    """
    for pat in patterns:
        hits = glob.glob(os.path.join(folder, pat))
        if hits:
            return hits[0]
    return None


#: Trailing tokens that say what a converted series *is* rather than which
#: acquisition it came from. Stripping them leaves the acquisition's own name.
_ROLE_SUFFIXES = ("_b0map_ph", "_echo1", "_echo2", "_e1", "_e2", "_ph",
                  "_phase", "_mag", "_magnitude")


def _acquisition_key(basename: str) -> str:
    """`B0fieldmap_ref_Echo2` and `B0fieldmap_ref_B0Map_ph` -> the same key.

    Files are grouped by this so a phase map is only ever paired with the
    magnitude from the *same* acquisition.
    """
    n = basename.lower()
    for ext in (".nii.gz", ".nii"):
        if n.endswith(ext):
            n = n[: -len(ext)]
            break
    changed = True
    while changed:
        changed = False
        for suf in _ROLE_SUFFIXES:
            if n.endswith(suf) and len(n) > len(suf):
                n = n[: -len(suf)]
                changed = True
    return n.rstrip("_")


def _find_phase_magnitude(folder: str, log_cb=None):
    """Locate one phase + magnitude pair, from a single acquisition.

    Naming conventions (see `_series_filename` in core.dicom_to_nifti):
      phase     → *_ph.nii.gz, *_e2_ph.nii.gz, *B0Map_ph.nii.gz, *phase*
      magnitude → *_e2.nii.gz, *Echo2*, *_e1.nii.gz, *Echo1*, *mag*

    Files are grouped by acquisition first. A folder can hold more than one
    complete B0 protocol -- an infant CBCP export carries `B0fieldmap_ref_*`
    and `B0fieldmap_shim_*`, one before and one after the scanner's own shim --
    and the previous version of this function assigned `phase_file` and
    `mag_file` by overwriting them as it walked the folder, so it kept whichever
    happened to come last and could pair a phase map with a magnitude from the
    other acquisition entirely. Grouping makes that impossible; when more than
    one complete acquisition is present the first in sorted order is used and
    every candidate is named in the log, because which of a before/after pair
    is the right one is not something this function can decide.

    Returns (phase_path, magnitude_path) or (None, None).
    """
    def _log(msg):
        _safe_print(msg)
        if log_cb:
            log_cb(msg)

    all_niis = sorted(glob.glob(os.path.join(folder, "*.nii.gz")))

    groups: dict = {}
    for f in all_niis:
        groups.setdefault(_acquisition_key(os.path.basename(f)), []).append(f)

    def _pick(files):
        """(phase, magnitude) within one acquisition, or (None, None)."""
        phase = mag = None
        # Magnitude preference: the second echo, then the first, then anything
        # calling itself a magnitude. Phase is checked first so a file named
        # `..._e2_ph` is never mistaken for the second-echo magnitude.
        for f in files:
            fn = os.path.basename(f).lower()
            if "_ph" in fn or "phase" in fn:
                phase = phase or f
        for want in ("echo2", "_e2", "echo1", "_e1", "mag"):
            for f in files:
                fn = os.path.basename(f).lower()
                if f is phase or "_ph" in fn or "phase" in fn:
                    continue
                if want in fn:
                    mag = f
                    break
            if mag:
                break
        return phase, mag

    complete = []
    for key in sorted(groups):
        p, m = _pick(groups[key])
        if p and m:
            complete.append((key, p, m))

    if complete:
        if len(complete) > 1:
            _log("  [!] 文件夹里有 %d 组完整的 B0 采集：%s"
                 % (len(complete), "、".join(k for k, _, _ in complete)))
            _log("      本次使用 %s。若应当使用另一组，请把对应 DICOM 单独放一个"
                 "文件夹再跑——两组通常是匀场前/后，选错结果没有意义。"
                 % complete[0][0])
        return complete[0][1], complete[0][2]

    # Nothing paired cleanly. Two files and no idea which is which is the one
    # case worth guessing at: the converter writes magnitude before phase.
    if len(all_niis) == 2:
        return all_niis[1], all_niis[0]

    return None, None


# ─────────────────────────────────────────────────────────────────────────
#  Public API
# ─────────────────────────────────────────────────────────────────────────

def dcm2nii(dicom_directory: str, nifti_folder: str) -> None:
    """Convert all sub-folders in dicom_directory to NIfTI."""
    for directory in os.listdir(dicom_directory):
        src = os.path.join(dicom_directory, directory)
        dst = os.path.join(nifti_folder, f"{directory}_nii")
        try:
            dicom_folder_to_nifti(src, dst)
            _safe_print(f"[dicom2nii] OK {directory}")
        except Exception as e:
            _safe_print(f"[dicom2nii] FAILED {directory}: {e}")


#: Set True (or export MACS_ALLOW_OTSU_BET=1) to permit the threshold fallback.
ALLOW_DEGRADED_BET = False


def bet_mask(input_path, output_path, f=0.5, log_cb=None, allow_degraded=None,
             provenance=None):
    """Produce a brain mask, returning its path.

    Backends, in order:

      1. cpp_bet     C++ 3D U-Net (HD-BET weights, ONNX Runtime).
                     Dice 0.991 vs HD-BET on the validation set.
      2. HD-BET CLI  the same network, run by HD-BET itself. Slower, but it is
                     the reference, so falling back here costs no accuracy.

    If neither is available this raises rather than quietly dropping to
    `_bet_robust` (percentile threshold + morphology). That fallback scores
    Dice 0.82 against HD-BET and yields masks 10-43% oversized; a bad brain
    mask then propagates into the unwrapped phase, the B0 map and the shim
    currents, where it is much harder to notice.

    Pass allow_degraded=True (or set MACS_ALLOW_OTSU_BET=1) to accept it.
    """
    import shutil as _shutil
    _log = log_cb or print
    stem = (output_path[:-len(".nii.gz")]
            if output_path.endswith(".nii.gz") else output_path)
    mask_path = stem + "_mask.nii.gz"
    reasons = []

    if allow_degraded is None:
        allow_degraded = (ALLOW_DEGRADED_BET or
                          os.environ.get("MACS_ALLOW_OTSU_BET") == "1")

    provenance = {} if provenance is None else provenance
    provenance.clear()
    provenance["attempted"] = reasons

    # ── 1. C++ U-Net ──
    try:
        from core.cpp_bet_bridge import bet as _cpp_bet_run
        produced = _cpp_bet_run(input_path, output_path, f=f, m=True, log_cb=_log)
        if produced and os.path.exists(produced):
            if os.path.abspath(produced) != os.path.abspath(mask_path):
                _shutil.copyfile(produced, mask_path)
            _log("  [bet] backend: cpp_bet (C++ 3D U-Net)")
            try:
                from core import cpp_bet_bridge as _cb
                br = _cb._get()
                st = br.stats
                provenance.update(
                    backend="cpp_bet",
                    model=os.path.basename(br.model_path),
                    mode=br.mode,
                    brain_voxels=int(st.brain_voxels),
                    patches=int(st.patches),
                    duration_s=round(float(st.seconds_total), 2),
                    inference_s=round(float(st.seconds_inference), 2))
            except Exception:
                provenance.update(backend="cpp_bet")
            return mask_path
        reasons.append("cpp_bet returned no mask file")
    except Exception as e:
        reasons.append(f"cpp_bet: {type(e).__name__}: {e}")
        _log(f"  [bet] cpp_bet unavailable ({type(e).__name__}: {e})")

    # ── 2. HD-BET CLI ──
    try:
        bet(input_path, output_path, f=f, m=True)
        for cand in (mask_path, stem + "_bet.nii.gz"):
            if os.path.exists(cand):
                if os.path.abspath(cand) != os.path.abspath(mask_path):
                    _shutil.copyfile(cand, mask_path)
                _log("  [bet] backend: HD-BET CLI")
                provenance.update(backend="hd_bet_cli")
                return mask_path
        reasons.append("HD-BET CLI produced no mask file")
    except Exception as e:
        reasons.append(f"HD-BET CLI: {type(e).__name__}: {e}")
        _log(f"  [bet] HD-BET CLI unavailable ({type(e).__name__}: {e})")

    # ── 3. Degraded, only on explicit request ──
    if allow_degraded:
        _log("  [bet] backend: threshold+morphology (DEGRADED, explicitly "
             "enabled) — mask will be ~10-43% oversized vs HD-BET")
        provenance.update(backend="threshold_morphology", degraded=True)
        return _bet_robust(input_path, output_path, f=f, m=True, log_cb=_log)

    raise RuntimeError(
        "脑提取失败：没有可用的高精度后端。\n已尝试：\n  - "
        + "\n  - ".join(reasons)
        + "\n\n修复方式（任选其一）：\n"
          "  1. 编译 C++ 模块：cmake --build cpp_bet/build --config Release\n"
          "  2. 导出模型：python cpp_bet/export_onnx.py\n"
          "  3. 安装 HD-BET：pip install hd-bet\n\n"
          "若确实接受精度较低的阈值掩膜（Dice≈0.82，体积偏大 10-43%），"
          "设置环境变量 MACS_ALLOW_OTSU_BET=1。"
    )


def _bet_robust(input_path, output_path, f=0.5, m=False, log_cb=None):
    """
    Fast brain mask from magnitude image.
    ~1 second. Uses percentile threshold + largest CC + fill holes.
    f: fraction of intensity range to threshold (0.1=aggressive, 0.5=conservative)

    Last-resort fallback only — see `bet_mask` for the preferred chain.
    """
    from scipy import ndimage
    from scipy.ndimage import gaussian_filter
    _log = log_cb or print

    img_nib  = nib.load(input_path)
    img_data = np.asanyarray(img_nib.dataobj, dtype=np.float32)
    affine   = img_nib.affine

    # Light smoothing to reduce noise
    img_s = gaussian_filter(img_data, sigma=2.0)

    # FSL BET threshold: g + f*(h-g)
    nz = img_s[img_s > 0]
    g = float(np.percentile(nz, 2))
    h = float(np.percentile(nz, 98))
    thresh = g + f * (h - g)
    _log(f"  [bet] thresh={thresh:.1f} (g={g:.1f} h={h:.1f} f={f})")

    mask = (img_s >= thresh).astype(np.uint8)

    # Largest 3D connected component
    struct = ndimage.generate_binary_structure(3, 2)
    lbl, n = ndimage.label(mask)
    if n > 1:
        sizes = [np.sum(lbl == i) for i in range(1, n+1)]
        mask = (lbl == (np.argmax(sizes) + 1)).astype(np.uint8)

    # Fill holes axially
    for iz in range(mask.shape[2]):
        mask[:, :, iz] = ndimage.binary_fill_holes(mask[:, :, iz])

    # Erode to strip skull (number of iterations controlled by f)
    n_erode = max(1, round(8 * (1 - f)))   # f=0.1->7, f=0.3->5, f=0.5->4
    n_dilate = n_erode + 2                  # always dilate slightly more
    _log(f"  [bet] erode={n_erode} dilate={n_dilate}")

    mask = ndimage.binary_erosion(mask,  structure=struct, iterations=n_erode).astype(np.uint8)
    # Re-take largest CC after erosion (removes stray skull fragments)
    lbl, n = ndimage.label(mask)
    if n > 1:
        sizes = [np.sum(lbl == i) for i in range(1, n+1)]
        mask = (lbl == (np.argmax(sizes) + 1)).astype(np.uint8)
    # Fill holes after erosion
    for iz in range(mask.shape[2]):
        mask[:, :, iz] = ndimage.binary_fill_holes(mask[:, :, iz])
    # Dilate back
    mask = ndimage.binary_dilation(mask, structure=struct, iterations=n_dilate).astype(np.uint8)
    # Final fill
    for iz in range(mask.shape[2]):
        mask[:, :, iz] = ndimage.binary_fill_holes(mask[:, :, iz])

    pct = 100.0 * mask.sum() / mask.size
    _log(f"  [bet] {int(mask.sum())} voxels ({pct:.1f}%)")

    _save_f32(img_data * mask, affine, output_path)
    _stem = output_path[:-len(".nii.gz")] if output_path.endswith(".nii.gz") else output_path
    mask_path = _stem + "_mask.nii.gz"
    _save_f32(mask.astype(np.float32), affine, mask_path)
    _log(f"  [bet] mask: {os.path.basename(mask_path)}")
    return mask_path

def _save_f32(data: np.ndarray, affine, path: str) -> None:
    """Save numpy array as float32 NIfTI, ignoring any existing int16 header."""
    img = nib.Nifti1Image(data.astype(np.float32), affine)
    img.header.set_data_dtype(np.float32)
    nib.save(img, path)

def process_one_subject(
        dicom_dir: str,
        nifti_dir: str,
        delta_TE: float,
        Frequency: float,
        bet_f: float = 0.1,
        log_cb=None) -> dict:
    """DICOM folder for ONE scan -> deltaB0, returning that subject's record.

    Split out of `process_dicom_to_deltaB0` so the one-click flow can run
    exactly the scan the user dropped. The batch entry point still exists and
    now loops over this, so both paths execute identical code.

    Returns {"status": "ok", "files": {...}} or {"status": "error", ...}.
    """
    def _log(msg):
        _safe_print(msg)
        if log_cb:
            log_cb(msg)

    directory = os.path.basename(os.path.normpath(dicom_dir))
    dicom_file_path = dicom_dir
    nifti_file_path = nifti_dir
    os.makedirs(nifti_file_path, exist_ok=True)

    try:
        # 1. DICOM → NIfTI
        _log("  [1/5] DICOM → NIfTI …")
        dicom_folder_to_nifti(dicom_file_path, nifti_file_path, log_cb=log_cb)

        # 2. Locate phase + magnitude
        phase_path, mag_path = _find_phase_magnitude(nifti_file_path, log_cb)
        if not phase_path or not mag_path:
            all_niis = glob.glob(os.path.join(nifti_file_path, "*.nii.gz"))
            raise FileNotFoundError(
                f"Could not identify phase/magnitude images.\n"
                f"Files found: {[os.path.basename(x) for x in all_niis]}"
            )
        _log(f"  [2/5] Phase:  {os.path.basename(phase_path)}")
        _log(f"         Magnitude: {os.path.basename(mag_path)}")

        # 3. Normalise phase to [-π, π]
        _log("  [3/5] Normalising phase …")
        ph_img  = nib.load(phase_path)
        affine  = ph_img.affine.copy()
        header  = ph_img.header.copy()

        # CRITICAL: use raw int16 values, bypass nibabel scl_slope scaling
        # get_fdata() applies scl_slope which turns [-4096,4096] into tiny floats
        raw_data = np.asanyarray(ph_img.dataobj, dtype=np.float32)
        raw_max  = float(np.abs(raw_data).max())
        if raw_max <= np.pi * 1.1:
            # already in radians
            ph_norm = raw_data
            _log(f"  phase in radians (max={raw_max:.3f})")
        else:
            # Siemens int16 encoding: [-4096,4096] -> [-pi,pi]
            ph_norm = (raw_data / 4096.0 * np.pi).astype(np.float32)
            _log(f"  phase int16 (max={raw_max:.1f}) -> rad")
        ph_norm_path = os.path.join(nifti_file_path, f"{directory}_PhaseMap.nii.gz")
        _save_f32(ph_norm, affine, ph_norm_path)

        # 4. BET — generate brain mask from magnitude
        _log("  [4/5] BET brain extraction …")
        bet_path  = os.path.join(nifti_file_path, f"{directory}_bet.nii.gz")
        bet_prov: dict = {}
        mask_path = bet_mask(mag_path, bet_path, f=bet_f, log_cb=_log,
                             provenance=bet_prov)
        if not mask_path or not os.path.exists(mask_path):
            raise FileNotFoundError(
                f"BET mask not found.\nFiles: {os.listdir(nifti_file_path)}"
            )
        _log(f"  mask: {os.path.basename(mask_path)}")

        # 5. mask → unwrap → deltaB0  (all in-memory float32, no int16 roundtrip)
        _log("  [5/5] Mask → Unwrap → deltaB0 …")

        # Use ph_norm directly from memory (already float32, never saved as int16)
        _mask_raw = np.asanyarray(nib.load(mask_path).dataobj, dtype=np.float32)
        _mask_bin = (_mask_raw > 0).astype(np.float32)

        # Unwrap inside the brain only.
        #
        # `unwrap_phase` grows a solution outward from a seed, deciding each
        # voxel's 2*pi branch from its neighbours. Run on the whole volume it
        # walks through the air outside the head, where the phase is noise, and
        # a wrong branch chosen out there can be carried back in across a thin
        # neck of voxels -- producing a brain region offset by a whole 2*pi.
        # Masking afterwards cannot undo that: the damage is already inside.
        # A masked array keeps the algorithm from ever crossing the boundary.
        _brain_bool = _mask_bin > 0.5
        _masked = np.ma.masked_array(ph_norm, mask=~_brain_bool)
        _unwrapped = unwrap_phase(_masked)
        unwrap_data = np.ma.filled(_unwrapped, 0.0).astype(np.float32)
        unwrap_path = os.path.join(nifti_file_path, f"{directory}_unwrapped_Brain.nii.gz")
        _save_f32(unwrap_data, affine, unwrap_path)

        # Brain-masked phase (float32, exact zeros outside brain)
        _brain = unwrap_data * _mask_bin  # ph_norm is float32 from memory
        brain_map_path = os.path.join(nifti_file_path, f"{directory}_BrainMap.nii.gz")
        _save_f32(_brain, affine, brain_map_path)

        # deltaB0 Hz = unwrapped_phase / (2π × ΔTE)
        hz_data = (_brain / (2.0 * np.pi * delta_TE)).astype(np.float32)
        hz_path = os.path.join(nifti_file_path, f"{directory}_deltaB0_hertz.nii.gz")
        _save_f32(hz_data, affine, hz_path)

        # deltaB0 ppm = Hz / Frequency_MHz
        ppm_data = (hz_data / Frequency).astype(np.float32)
        ppm_path = os.path.join(nifti_file_path, f"{directory}_deltaB0_ppm.nii.gz")
        _save_f32(ppm_data, affine, ppm_path)

        _log(f"  ✓ Done → {nifti_file_path}")
        return {
            "status": "ok",
            "bet": bet_prov,
            "files":  {
                "magnitude":    mag_path,
                # The converted phase, before normalisation. Kept because it is
                # the only handle on the phase series' own sidecar, and the two
                # halves of the pair carrying different transmit frequencies is
                # how a mis-pairing across an f0 recalibration would show up.
                "phase":        phase_path,
                "phase_norm":   ph_norm_path,
                "bet":          bet_path,
                "bet_mask":     mask_path,
                "brain_map":    brain_map_path,
                "unwrapped":    unwrap_path,
                "deltaB0_hz":   hz_path,
                "deltaB0_ppm":  ppm_path,
            }
        }
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        _log(f"  Error: {e}\n{tb}")
        return {"status": "error", "error": str(e), "traceback": tb}


def process_dicom_to_deltaB0(
        dicom_directory: str,
        nifti_folder: str,
        delta_TE: float,
        Frequency: float,
        bet_f: float = 0.1,
        log_cb=None) -> dict:
    """Batch wrapper: run `process_one_subject` over every sub-folder.

    Kept for the existing multi-subject workflow. The one-click page calls
    `process_one_subject` directly instead of relying on this picking the right
    scan out of a folder that only ever holds one.
    """
    def _log(msg):
        _safe_print(msg)
        if log_cb:
            log_cb(msg)

    _log("=== B0 pipeline start ===")
    results = {}

    entries = [d for d in os.listdir(dicom_directory)
               if os.path.isdir(os.path.join(dicom_directory, d))]
    if not entries:
        _log("[WARN] No sub-folders found in DICOM directory.")
        return results

    for directory in entries:
        _log(f"\n-- {directory} --")
        results[directory] = process_one_subject(
            os.path.join(dicom_directory, directory),
            os.path.join(nifti_folder, f"{directory}_nii"),
            delta_TE, Frequency, bet_f=bet_f, log_cb=log_cb)

    _log("\n=== Pipeline complete ===")
    return results


def get_reference_file(nifti_folder: str) -> list:
    """Return path to echo-2 image in the *reference* sub-folder."""
    ref_folders = glob.glob(os.path.join(nifti_folder, "*reference*"))
    if not ref_folders:
        print("[get_reference_file] No *reference* folder found.")
        return []
    hits = glob.glob(os.path.join(ref_folders[0], "*e2.nii.gz"))
    return hits if hits else []


def corigistrator(nifti_folder: str,
                  reference_file_path: list,
                  log_cb=None) -> None:
    """Register each deltaB0 map to the reference image using FLIRT."""
    def _log(msg):
        _safe_print(msg)
        if log_cb:
            log_cb(msg)

    if not reference_file_path:
        _log("[corigistrator] No reference file — skipping registration.")
        return

    for directory in os.listdir(nifti_folder):
        nii_path = os.path.join(nifti_folder, directory)
        if not os.path.isdir(nii_path):
            continue

        mag_file  = _glob1(nii_path, "*e2.nii.gz", "*mag*.nii.gz")
        b0_map    = _glob1(nii_path, "*ppm*.nii.gz", "*deltaB0_ppm*")

        if not mag_file or not b0_map:
            _log(f"[corigistrator] Skipping {directory} (mag or ppm map not found)")
            continue

        regis_mat  = os.path.join(nii_path, f"{directory}_regis.mat")
        regis_mag  = os.path.join(nii_path, f"{directory}_regis_mag.nii.gz")
        regis_ppm  = os.path.join(nii_path, f"{directory}_regis_deltB0_ppm.nii.gz")

        try:
            _log(f"[corigistrator] Registering {directory} …")
            flirt(src=mag_file,  ref=reference_file_path[0], omat=regis_mat, out=regis_mag)
            flirt(src=b0_map,    ref=reference_file_path[0],
                  applyxfm=True, init=regis_mat, out=regis_ppm)

            # also save unzipped copy (legacy behaviour)
            regis_ppm_nii = regis_ppm.replace(".gz", "")
            with gzip.open(regis_ppm, "rb") as fi, open(regis_ppm_nii, "wb") as fo:
                shutil.copyfileobj(fi, fo)

            _log(f"  ✓ {directory}")
        except Exception as e:
            _log(f"  ✗ {directory}: {e}")


# ─────────────────────────────────────────────────────────────────────────
#  CLI entry point
# ─────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="B0 map pipeline (FSL-free)")
    parser.add_argument("dicom_directory")
    parser.add_argument("nifti_folder")
    parser.add_argument("--field_strength", type=float, required=True)
    parser.add_argument("--delta_TE",       type=float, default=0.00246)
    args = parser.parse_args()

    Frequency = 42.58 * args.field_strength
    process_dicom_to_deltaB0(args.dicom_directory, args.nifti_folder,
                             args.delta_TE, Frequency)
    ref = get_reference_file(args.nifti_folder)
    corigistrator(args.nifti_folder, ref)
    print("Done!")

if __name__ == "__main__":
    main()
