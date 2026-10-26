"""One-shot shimming: a DICOM folder in, optimised currents out.

This module only orchestrates. Every numerical step is the existing, already
validated code -- `process_dicom_to_deltaB0` for the field map,
`optimize_currents` for the solve -- so if a result changes after this refactor
the cause is the flow, never the maths.

    from core.pipeline import run_pipeline
    result = run_pipeline("D:/data/S28", progress_cb=print)
    result.currents        # np.ndarray, amps per channel
    result.stats           # before/after RMSE, std, ptp
"""
from __future__ import annotations

import os
import re
import glob as _glob
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import nibabel as nib

from core.coil_registry import Coil, get_default as default_coil
from core.data_context import natural_key, resample_to_target, same_grid
from core.shim_optimize import optimize_currents

__all__ = ["run_pipeline", "PipelineResult", "PipelineError", "looks_like_dicom_dir"]


class PipelineError(RuntimeError):
    """A failure the user can act on.

    Carries a plain-language message; `technical` keeps the underlying
    exception text for the log.
    """

    def __init__(self, message: str, technical: str = "",
                 code: str = "", doc: str = ""):
        super().__init__(message)
        self.message = message
        self.technical = technical
        self.code = code or "pipeline"
        self.doc = doc

    @classmethod
    def from_exception(cls, exc: BaseException, *, context: str = ""):
        """Wrap any exception as the actionable message it corresponds to."""
        from core.errors import classify

        a = classify(exc, context=context)
        return cls(a.headline, technical=a.detail, code=a.code, doc=a.doc)


@dataclass
class PipelineResult:
    subject: str = ""
    nifti_dir: Path = field(default_factory=Path)
    files: dict = field(default_factory=dict)

    b0: np.ndarray | None = None            # deltaB0 in Hz
    affine: np.ndarray | None = None

    #: Transmit frequency the field map was acquired at, in Hz, from DICOM
    #: (0018,0084). The zero of `b0`, and so the only frequency the system-shim
    #: page's centre-frequency offset may be added to. None when the DICOM did
    #: not give one consistent value.
    imaging_frequency_hz: float | None = None
    brain_mask: np.ndarray | None = None    # BET output, bool
    roi_mask: np.ndarray | None = None      # what was optimised over
    roi_name: str = "whole brain"

    coil_name: str = ""
    channel_names: list = field(default_factory=list)
    coil_maps: np.ndarray | None = None     # (nx, ny, nz, n_ch)

    currents: np.ndarray | None = None
    residual: np.ndarray | None = None
    stats: dict = field(default_factory=dict)
    channels_at_limit: int = 0

    #: Non-fatal findings worth showing next to the result, each
    #: {code, headline, detail} -- see core.errors.
    warnings: list = field(default_factory=list)
    b0_std_hz: float = 0.0

    #: Where the provenance JSON was written; the Teensy page appends the
    #: upload outcome to this same file.
    report_path: Path | None = None

    #: "static" -- one current vector for the whole head, `currents` is (n_ch,).
    #: "dynamic" -- one vector per slice, `slice_currents` is (n_slices, n_ch)
    #: and `currents` holds the mean, so anything expecting a vector still works.
    mode: str = "static"
    slice_currents: np.ndarray | None = None
    slice_rmse_before: np.ndarray | None = None
    slice_rmse_after: np.ndarray | None = None
    slice_voxels: np.ndarray | None = None
    slice_solved: np.ndarray | None = None
    slice_axis: int = 2
    csv_path: Path | None = None

    seconds: dict = field(default_factory=dict)

    @property
    def total_seconds(self) -> float:
        return sum(self.seconds.values())


# ──────────────────────────────────────────────────────────────────────
def looks_like_dicom_dir(path: str | os.PathLike, max_depth: int = 3) -> tuple[bool, str]:
    """Cheap check before starting a long run. Returns (ok, why-not).

    Scanner exports nest differently -- files may sit directly in the dropped
    folder, one level down (subject/series), or two (subject/series/files) --
    so look down a few levels rather than assuming one layout.
    """
    p = Path(path)
    if not p.is_dir():
        return False, f"不是文件夹：{p}"

    def _dicom_count(d: Path) -> int:
        n = 0
        try:
            for f in list(d.iterdir())[:400]:
                if f.is_file() and (f.suffix.lower() in (".dcm", ".ima")
                                    or not f.suffix):
                    n += 1
        except OSError:
            return 0
        return n

    level = [p]
    for _ in range(max_depth):
        if any(_dicom_count(d) >= 5 for d in level):
            return True, ""
        nxt = []
        for d in level[:30]:                 # keep the scan bounded
            try:
                nxt += [c for c in d.iterdir() if c.is_dir()]
            except OSError:
                pass
        if not nxt:
            break
        level = nxt

    if list(p.rglob("*.nii*"))[:1]:
        return False, ("这个文件夹里是 NIfTI 而不是 DICOM。"
                       "请拖入包含 DICOM 序列的原始文件夹。")
    return False, ("没有在这个文件夹里找到 DICOM 文件。"
                   "请拖入扫描仪导出的原始序列文件夹（.dcm/.IMA 或无后缀文件）。")


def _grid_key(coil: Coil, ref_affine, ref_shape) -> str:
    """Identifies a (coil, target grid) pair for caching."""
    import hashlib
    h = hashlib.sha1()
    h.update(coil.name.encode("utf-8"))
    h.update(np.asarray(ref_shape, dtype=np.int64).tobytes())
    h.update(np.round(np.asarray(ref_affine, dtype=np.float64), 6).tobytes())
    return h.hexdigest()[:16]


def _load_coil_maps(coil: Coil, ref_affine, ref_shape, log):
    """Channel field maps on the subject's B0 grid.

    The coil's own `*_deltaFieldmap.nii.gz` lives on the calibration grid with
    its own affine, and is resampled straight onto the subject grid in one
    trilinear pass through world coordinates. There is deliberately no
    intermediate `_resampled_to_brain` file: that is the same map already
    resampled onto some other subject's matrix, so consuming it would
    interpolate twice for no benefit.

    The result is cached per (coil, grid), so switching ROI on the same subject
    does not repeat the work.
    """
    dirs = coil.channel_dirs()
    if not dirs:
        raise PipelineError(
            f"线圈「{coil.name}」没有找到通道场图。\n"
            f"检查目录：{coil.fieldmap_dir}",
            technical=f"no ch*_nii under {coil.fieldmap_dir}")

    import tempfile
    cache_dir = Path(tempfile.gettempdir()) / "macs_coil_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache = cache_dir / f"{_grid_key(coil, ref_affine, ref_shape)}.npz"

    if cache.is_file():
        try:
            blob = np.load(cache, allow_pickle=False)
            maps = blob["maps"]
            names = [str(n) for n in blob["names"]]
            if maps.shape[:3] == tuple(ref_shape) and len(names) == maps.shape[-1]:
                log(f"  载入 {len(names)} 个通道场图（缓存）")
                return maps, names
        except Exception:
            pass        # a damaged cache is not worth reporting; just redo it

    maps, names = [], []
    for d in dirs:
        cand = sorted(d.glob("*deltaFieldmap.nii*")) or sorted(d.glob("*.nii*"))
        if not cand:
            log(f"  跳过 {d.name}：没有场图文件")
            continue
        img = nib.load(str(cand[0]))
        data = img.get_fdata(dtype=np.float32)
        # Matching matrix size does not mean matching space -- see `same_grid`.
        if not same_grid(data.shape, img.affine, ref_shape, ref_affine):
            data = resample_to_target(data, img.affine, ref_affine,
                                      ref_shape, order=1)
        maps.append(data)
        names.append(d.name.replace("_nii", ""))

    if not maps:
        raise PipelineError(
            f"线圈「{coil.name}」的通道场图全部无法读取。",
            technical=f"no readable NIfTI in {coil.fieldmap_dir}")

    stacked = np.stack(maps, axis=-1)
    try:
        np.savez_compressed(cache, maps=stacked, names=np.array(names))
    except OSError:
        pass            # caching is an optimisation, never a requirement
    log(f"  载入 {len(maps)} 个通道场图（重采样到被试网格）")
    return stacked, names


def resolve_subject_dir(dropped: Path) -> tuple[Path, list[Path]]:
    """Work out which folder is the scan to process.

    Scanner exports nest inconsistently: the dropped folder may hold the DICOM
    series directly, or one sub-folder per series, or one sub-folder per
    subject each holding series. Returns (chosen, all_candidates) so the caller
    can say when it picked one of several.
    """
    def has_dicom(d: Path, depth: int = 2) -> bool:
        level = [d]
        for _ in range(depth + 1):
            for x in level:
                try:
                    n = sum(1 for f in list(x.iterdir())[:200]
                            if f.is_file() and (f.suffix.lower() in (".dcm", ".ima")
                                                or not f.suffix))
                except OSError:
                    n = 0
                if n >= 5:
                    return True
            nxt = []
            for x in level[:20]:
                try:
                    nxt += [c for c in x.iterdir() if c.is_dir()]
                except OSError:
                    pass
            if not nxt:
                break
            level = nxt
        return False

    subs = sorted(d for d in dropped.iterdir() if d.is_dir()) \
        if dropped.is_dir() else []

    # Series folders sitting directly in the drop: that IS one scan.
    direct = any(f.is_file() and (f.suffix.lower() in (".dcm", ".ima") or not f.suffix)
                 for f in list(dropped.iterdir())[:200]) if dropped.is_dir() else False
    if direct:
        return dropped, [dropped]

    with_dicom = [d for d in subs if has_dicom(d)]
    if not with_dicom:
        return dropped, [dropped]
    if len(with_dicom) == 1:
        return with_dicom[0], [with_dicom[0]]

    # Several sub-folders hold DICOMs. They are either the series of one scan
    # (B0Map + Echo1 + Echo2, which must be processed together) or different
    # subjects. StudyInstanceUID settles it -- folder names cannot.
    def study_of(d: Path) -> str | None:
        try:
            import pydicom
        except ImportError:
            return None
        for f in sorted(d.rglob("*"))[:400]:
            if not f.is_file():
                continue
            try:
                ds = pydicom.dcmread(str(f), stop_before_pixels=True,
                                     specific_tags=["StudyInstanceUID"])
                uid = getattr(ds, "StudyInstanceUID", None)
                if uid:
                    return str(uid)
            except Exception:
                continue
        return None

    studies = {}
    for d in with_dicom:
        studies.setdefault(study_of(d), []).append(d)

    if len(studies) <= 1:
        # One study spread over series folders: the drop itself is the scan.
        return dropped, [dropped]
    # Genuinely different studies -> one folder per subject.
    return with_dicom[0], with_dicom


def _pick_subject(results: dict) -> tuple[str, dict]:
    ok = {k: v for k, v in results.items() if v.get("status") == "ok"}
    if not ok:
        errs = []
        for k, v in results.items():
            e = str(v.get("error", "")).strip()
            errs.append(f"{k}: {e}" if e else k)
        raise PipelineError(
            "B0 场图计算失败，没有任何序列成功。\n"
            "常见原因：DICOM 里不含双回波 GRE（需要幅值 + 相位），"
            "或 ΔTE 填写有误。",
            technical="; ".join(errs)[:800])
    name = sorted(ok)[0]
    return name, ok[name]


# ──────────────────────────────────────────────────────────────────────
def run_pipeline(dicom_dir: str | os.PathLike,
                 output_dir: str | os.PathLike | None = None,
                 coil: Coil | None = None,
                 roi: str = "whole_brain",     # key from roi_registry, not a label
                 roi_mask: np.ndarray | None = None,
                 custom_mask_path=None,
                 mode: str = "static",
                 delta_te: float = 0.00246,
                 field_t: float = 5.0,
                 auto_params: bool = True,
                 lambda_penalty: float = 0.5,
                 progress_cb=None) -> PipelineResult:
    """DICOM folder -> optimised shim currents, with no further interaction.

    `roi_mask` overrides `roi`; pass a native-space boolean array to shim a
    specific region. Otherwise the BET brain mask is used.
    """
    log = progress_cb or (lambda *_: None)
    t = {}

    dicom_dir = Path(dicom_dir)
    ok, why = looks_like_dicom_dir(dicom_dir)
    if not ok:
        raise PipelineError(why, technical=str(dicom_dir))

    out = Path(output_dir) if output_dir else dicom_dir.parent / f"{dicom_dir.name}_nii"
    out.mkdir(parents=True, exist_ok=True)

    coil = coil or default_coil()
    if coil is None:
        raise PipelineError(
            "没有可用的线圈配置。\n"
            "请在 config/coils/ 下放置线圈 JSON，并确认其 fieldmap_dir 存在。",
            technical="coil_registry returned no usable coil")

    if mode not in ("static", "dynamic"):
        raise PipelineError(f"未知的匀场模式：{mode}", code="bad_mode")

    # Dynamic shimming is per-slice over the whole head by construction: the
    # sequence steps through slices, not through anatomical regions, so an
    # anatomical ROI has nothing to select here.
    if mode == "dynamic":
        roi, roi_mask, custom_mask_path = "whole_brain", None, None

    res = PipelineResult(coil_name=coil.name, roi_name=roi, mode=mode)

    # ── 1. DICOM -> deltaB0 (conversion, BET, unwrap, Hz) ──
    # ΔTE and field strength are recorded in every DICOM slice. Reading them
    # beats defaulting: a wrong ΔTE rescales the whole field map, and because
    # the map keeps its shape the error is invisible -- the currents come out
    # proportionally wrong with nothing on screen looking unusual.
    # The probe runs whether or not `auto_params` is set: the flag governs
    # whether the DICOM's ΔTE and field strength *override* what the page was
    # given, and the transmit frequency overrides nothing -- it is the zero of
    # the field map, recorded so the system-shim page can report an absolute
    # target f0 instead of an offset the operator has to attach to the right
    # baseline by hand. Reading it costs one header pass with no pixel data.
    from core.dicom_to_nifti import probe_acquisition

    probe = probe_acquisition(dicom_dir, log_cb=log)
    res.imaging_frequency_hz = probe.get("imaging_frequency_hz")
    if auto_params:
        if probe["delta_te_s"]:
            if abs(probe["delta_te_s"] - delta_te) > 1e-6:
                log(f"  ΔTE 采用 DICOM 的 {probe['delta_te_s']*1000:.3f} ms"
                    f"（页面填的是 {delta_te*1000:.3f} ms）")
            delta_te = float(probe["delta_te_s"])
        else:
            log(f"  DICOM 未给出 ΔTE，沿用页面填的 {delta_te*1000:.3f} ms")
        if probe["field_t"]:
            field_t = float(probe["field_t"])

    log(f"ΔTE = {delta_te:.5f} s ({delta_te*1000:.2f} ms)   "
        f"field = {field_t:.2f} T   coil = {coil.name}")
    log("[1/3] DICOM → ΔB0 …")
    t0 = time.time()
    from core.B0_cal_algorithm import process_one_subject

    # Process exactly the scan that was dropped. The batch entry point would
    # walk every sub-folder, which for one-click means doing 20 subjects'
    # worth of work to get one answer.
    subject_dir, candidates = resolve_subject_dir(dicom_dir)
    if len(candidates) > 1:
        log(f"  检测到 {len(candidates)} 个被试，已选择第一个：{subject_dir.name}")
        log(f"  （其余：{', '.join(d.name for d in candidates[1:6])}"
            f"{' …' if len(candidates) > 6 else ''}）")
    subject = subject_dir.name

    # `out` is already named "<dropped>_nii". Appending "<subject>_nii" again
    # when the subject IS the dropped folder produced "X_nii/X_nii". The
    # sub-folder still earns its place when the subject is a child, because
    # `_find_phase_magnitude` globs the directory -- files left from a
    # previously processed subject would otherwise be candidates for pairing.
    nifti_dir = out if subject_dir == dicom_dir else out / f"{subject}_nii"

    try:
        info = process_one_subject(
            str(subject_dir), str(nifti_dir),
            delta_te, 42.58 * field_t, bet_f=0.1, log_cb=log)
    except Exception as e:
        raise PipelineError.from_exception(
            e, context=f"阶段：DICOM → ΔB0；被试目录：{subject_dir}") from e
    t["b0"] = time.time() - t0

    subject, info = _pick_subject({subject: info})
    res.subject, res.nifti_dir, res.files = subject, out, info["files"]

    # The transmit frequency of the pair that was actually processed, which is
    # not a property of the folder: a calibration session holds several
    # acquisitions, and moving the table or swapping the phantom changes f0
    # legitimately. Measured on one 5T phantom folder -- two phantoms, three
    # table positions -- the accepted B0 series span 823 Hz, so a folder-wide
    # value would be a different scan's zero. `probe_acquisition` only answers
    # when the folder is unambiguous; the sidecar always answers for this pair.
    from core.dicom_to_nifti import sidecar_frequency_hz

    f_mag = sidecar_frequency_hz(info["files"].get("magnitude"))
    f_ph = sidecar_frequency_hz(info["files"].get("phase"))
    if f_mag and f_ph and f_mag != f_ph:
        # One acquisition has one f0. Two different ones mean phase and
        # magnitude were paired across a recalibration, which makes the field
        # map itself wrong, not just its zero -- so neither value is used.
        log(f"  ⚠ 相位与幅值的采集频率不一致（相差 {abs(f_mag - f_ph):.0f} Hz）"
            "——这两组可能不是同一次采集，中心频率不再给绝对值")
        res.imaging_frequency_hz = None
    elif f_mag or f_ph:
        res.imaging_frequency_hz = f_mag or f_ph
        log(f"  场图采集频率 {res.imaging_frequency_hz / 1e6:.6f} MHz"
            f"（{os.path.basename(info['files'].get('magnitude') or '')}）")
    elif res.imaging_frequency_hz:
        log(f"  场图采集频率 {res.imaging_frequency_hz / 1e6:.6f} MHz（文件夹唯一值）")

    hz_path = info["files"].get("deltaB0_hz", "")
    if not hz_path or not os.path.isfile(hz_path):
        raise PipelineError("B0 场图文件没有生成。",
                            technical=f"deltaB0_hz missing for {subject}")
    img = nib.load(hz_path)
    res.b0 = np.asanyarray(img.dataobj, dtype=np.float32)
    res.affine = img.affine.copy()

    mask_path = info["files"].get("bet_mask", "")
    res.brain_mask = (np.asanyarray(nib.load(mask_path).dataobj) > 0.5
                      if mask_path and os.path.isfile(mask_path)
                      else res.b0 != 0)

    # A field map that is essentially flat, or wildly large, almost always
    # means dTE was entered in the wrong unit -- say so now rather than let it
    # propagate into the currents.
    inside = res.b0[res.brain_mask]
    if inside.size:
        from core.errors import check_b0_scale

        res.b0_std_hz = float(np.std(inside))
        warn = check_b0_scale(res.b0_std_hz, delta_te)
        if warn is not None:
            res.warnings.append({"code": warn.code, "headline": warn.headline,
                                 "detail": warn.detail})
            log(f"  ⚠ {warn.headline}")

    # ── 2. ROI ──
    # A user-supplied mask is validated here, not when it was picked in the
    # dialog: only now does the B0 grid it has to match actually exist.
    if custom_mask_path:
        from core.roi_registry import load_custom_mask, ROIError

        try:
            cm = load_custom_mask(custom_mask_path,
                                  ref_affine=res.affine, ref_shape=res.b0.shape[:3])
        except ROIError as e:
            raise PipelineError(str(e), technical=f"custom mask={custom_mask_path}",
                                code="custom_mask_invalid") from e
        for w in cm.warnings:
            log(f"  ⚠ {w}")
            res.warnings.append({"code": "custom_mask", "headline": w, "detail": ""})
        log(f"  ROI: 自定义 mask「{cm.name}」（{int(cm.mask.sum())} 体素）")
        res.roi_mask = cm.mask & res.brain_mask.astype(bool)
        res.roi_name = f"custom:{cm.name}"
        if not res.roi_mask.any():
            raise PipelineError(
                "自定义 mask 与脑掩膜没有交集。\n"
                "请确认这个 mask 来自同一次扫描，且覆盖脑内区域。",
                technical=f"custom mask={custom_mask_path}",
                code="custom_mask_disjoint")
    elif roi_mask is not None:
        if roi_mask.shape != res.b0.shape:
            raise PipelineError(
                f"所选脑区的尺寸 {roi_mask.shape} 与 B0 图 {res.b0.shape} 不一致。",
                technical="roi_mask shape mismatch")
        res.roi_mask = roi_mask.astype(bool)
    elif roi and roi != "whole_brain":
        from core.roi_registry import get_roi_mask, ROIError
        log(f"  ROI: {roi}")
        try:
            res.roi_mask = get_roi_mask(
                roi, b0_shape=res.b0.shape, b0_affine=res.affine,
                brain_mask=res.brain_mask,
                subject_dir=Path(hz_path).parent,
                magnitude_path=info["files"].get("magnitude"),
                log_cb=log)
        except ROIError as e:
            # ROIError is already user-facing; keep its wording, add the code.
            raise PipelineError(str(e), technical=f"roi={roi}",
                                code="roi_failed") from e
        except Exception as e:
            raise PipelineError.from_exception(e, context=f"ROI：{roi}") from e
        res.roi_name = roi
    else:
        res.roi_mask = res.brain_mask
        res.roi_name = "whole_brain"
    if not res.roi_mask.any():
        raise PipelineError("所选区域内没有体素，无法优化。",
                            technical="empty ROI mask")

    # ── 3. Coil maps + solve ──
    log("[2/3] 载入线圈场图 …")
    t0 = time.time()
    res.coil_maps, res.channel_names = _load_coil_maps(
        coil, res.affine, res.b0.shape[:3], log)
    t["coils"] = time.time() - t0

    t0 = time.time()
    if mode == "dynamic":
        from core.dynamic_shim import optimize_per_slice

        log("[3/3] 逐层优化电流 …")
        dyn = optimize_per_slice(res.b0, res.brain_mask, res.coil_maps,
                                 lambda_penalty, coil.current_limit_a,
                                 log_cb=log)
        res.slice_currents = dyn["currents"]
        res.slice_rmse_before = dyn["rmse_before"]
        res.slice_rmse_after = dyn["rmse_after"]
        res.slice_voxels = dyn["voxels"]
        res.slice_solved = dyn["solved"]
        res.slice_axis = dyn["axis"]
        res.residual = dyn["residual"]
        res.stats = dyn["stats"]
        res.channels_at_limit = int(np.max(dyn["channels_at_limit"]))
        # A single representative vector, for the parts of the UI and the
        # amplifier bridge that predate dynamic mode.
        solved = dyn["solved"]
        res.currents = (dyn["currents"][solved].mean(axis=0) if solved.any()
                        else np.zeros(dyn["n_channels"]))
    else:
        log("[3/3] 优化电流 …")
        # `outside_mask` is passed explicitly, never defaulted: without it the
        # penalty region is every voxel outside the ROI, which is almost all
        # air, and the solution would depend on the field of view.
        sol = optimize_currents(res.b0, res.roi_mask.astype(np.float32),
                                res.coil_maps, lambda_penalty,
                                coil.current_limit_a, log_cb=log,
                                outside_mask=res.brain_mask)
        res.currents = sol["currents"]
        res.residual = sol["residual"]
        res.stats = sol["stats"]
        res.channels_at_limit = sol["channels_at_limit"]
    t["optimise"] = time.time() - t0
    res.seconds = t

    log(f"完成，总耗时 {res.total_seconds:.1f}s "
        f"(ΔB0 {t['b0']:.1f}s · 线圈 {t['coils']:.1f}s · 优化 {t['optimise']:.1f}s)")

    # Provenance is written last, so the file only ever describes a run that
    # actually produced currents. A failure to write it must not lose the
    # result the user is waiting for -- report it and carry on.
    try:
        from core import run_report

        res.report_path = run_report.save(
            res, dicom_dir=dicom_dir, delta_te=delta_te, field_t=field_t,
            lambda_penalty=lambda_penalty, coil=coil,
            bet_info=info.get("bet") or {})
        log(f"  运行记录 → {res.report_path}")

        if mode == "dynamic" and res.slice_currents is not None:
            from core.dynamic_shim import to_csv

            res.csv_path = Path(to_csv(res.slice_currents,
                                       Path(res.report_path).parent / "dynamic_currents.csv",
                                       res.channel_names))
            log(f"  电流表 → {res.csv_path}")
    except Exception as e:
        log(f"  ⚠ 运行记录未能写入（结果不受影响）：{type(e).__name__}: {e}")

    return res
