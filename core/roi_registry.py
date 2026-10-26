"""ROI choices as a list, not a file path.

The user picks "whole brain" or "PFC" from a dropdown; this module works out
where the mask comes from and returns it on the subject's own grid.

Sources, in the order they are tried per ROI:

  whole brain  the BET mask the pipeline just produced -- always available
  PFC          atlases/pfc_mask_v3.nii.gz, registered MNI -> native with ANTs
  temporal     a `*temporal_mask*.nii.gz` sitting next to the subject's data,
               else nothing (there is no temporal atlas shipped)

Registration output is cached under `<subject_dir>/roi_cache/` so switching
ROIs back and forth does not re-register every time.
"""
from __future__ import annotations

import glob as _glob
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import nibabel as nib

from core.data_context import resample_to_target, same_grid
from core.paths import find_atlases

__all__ = ["ROI", "list_rois", "get_roi_mask", "ROIError",
           "CUSTOM_KEY", "load_custom_mask", "CustomMask"]


class ROIError(RuntimeError):
    """Actionable failure while producing a mask."""


@dataclass
class ROI:
    key: str
    label: str
    description: str
    needs_registration: bool = False


#: Selected in the dropdown to open a file dialog rather than name a built-in.
CUSTOM_KEY = "custom"


ROIS = [
    ROI("whole_brain", "全脑", "BET 输出的脑掩膜，无需额外配准"),
    ROI("pfc", "前额叶 (PFC)", "MNI 图谱配准到被试空间", needs_registration=True),
    ROI("temporal", "颞叶", "使用被试目录下已有的颞叶 mask"),
    ROI(CUSTOM_KEY, "自定义…", "导入自己的 mask 文件 (.nii / .nii.gz)"),
]


def list_rois() -> list[ROI]:
    return list(ROIS)


# ─────────────────────────────────────────────────────────────────────────
#  Custom mask files
# ─────────────────────────────────────────────────────────────────────────

@dataclass
class CustomMask:
    """A mask the user supplied, already checked and on the caller's grid."""
    name: str
    path: Path
    mask: np.ndarray
    resampled: bool = False
    warnings: list = None

    def __post_init__(self):
        if self.warnings is None:
            self.warnings = []


def load_custom_mask(path, *, ref_affine, ref_shape) -> CustomMask:
    """Read a user mask, validate it, and put it on the reference grid.

    Resampling is done when the grids differ rather than refusing, because a
    mask drawn on the magnitude image is the common case and its grid legitimately
    differs from the B0 map. It is reported as a warning, not done silently --
    a mask that needed resampling has been through nearest-neighbour
    interpolation and its edges are no longer exactly what was drawn.
    """
    p = Path(path)
    if not p.is_file():
        raise ROIError(f"找不到 mask 文件：{p}")

    try:
        img = nib.load(str(p))
    except Exception as e:
        raise ROIError(
            f"无法读取 mask 文件：{p.name}\n"
            f"请确认它是 NIfTI 格式（.nii / .nii.gz）。\n"
            f"（{type(e).__name__}: {e}）") from e

    arr = np.asanyarray(img.dataobj)
    if arr.ndim == 4 and arr.shape[3] == 1:
        arr = arr[..., 0]                      # a 4D mask with one volume is 3D
    if arr.ndim != 3:
        raise ROIError(
            f"mask 必须是 3D 图像，当前是 {arr.ndim}D，尺寸 {tuple(arr.shape)}。")

    warnings: list = []

    finite = arr[np.isfinite(arr)]
    uniq = np.unique(finite)
    if uniq.size > 2:
        # A label atlas or a probability map, not a binary mask. Thresholding at
        # >0 keeps it usable but the user should know it was not binary.
        warnings.append(
            f"mask 不是二值图（有 {uniq.size} 种取值），已按 >0 处理为二值。")

    m = np.zeros(arr.shape, dtype=bool)
    np.greater(np.nan_to_num(arr, nan=0.0), 0, out=m)
    if not m.any():
        raise ROIError(f"mask 全为 0，没有任何体素被选中：{p.name}")

    resampled = False
    if not same_grid(m.shape, img.affine, ref_shape, ref_affine):
        same_size = tuple(m.shape[:3]) == tuple(ref_shape)
        warnings.append(
            ("mask 尺寸与 B0 图相同，但空间信息（原点/方向/体素间距）不一致，"
             "已按空间坐标重采样（最近邻）。"
             if same_size else
             f"mask 尺寸 {m.shape} 与 B0 图 {tuple(ref_shape)} 不一致，"
             "已按空间坐标重采样（最近邻）。"))
        m = _resample_to(m.astype(np.float32), img.affine, ref_affine, ref_shape)
        resampled = True
        if not m.any():
            raise ROIError(
                f"重采样后 mask 为空：{p.name}\n"
                "两幅图像大概率不在同一个空间（affine 不重叠），"
                "请确认 mask 来自这一次扫描。")

    name = p.name
    for suf in (".nii.gz", ".nii"):
        if name.lower().endswith(suf):
            name = name[: -len(suf)]
            break

    return CustomMask(name=name, path=p, mask=m,
                      resampled=resampled, warnings=warnings)


def _resample_to(mask: np.ndarray, src_affine, ref_affine, ref_shape) -> np.ndarray:
    # Same matrix size is not the same space: origin, orientation and spacing
    # can all differ. Skipping the resample on shape alone would place the mask
    # over the wrong anatomy while every count and statistic still looked fine.
    if same_grid(mask.shape, src_affine, ref_shape, ref_affine):
        return mask > 0.5
    out = resample_to_target(mask.astype(np.float32), src_affine,
                             ref_affine, ref_shape, order=0)
    return out > 0.5


def _find_subject_mask(subject_dir: Path, *patterns) -> Path | None:
    for pat in patterns:
        hits = sorted(subject_dir.glob(pat))
        if hits:
            return hits[0]
    return None


def _register_pfc(magnitude: Path, brain_mask: np.ndarray,
                  ref_affine, ref_shape, cache_dir: Path, log) -> np.ndarray:
    """MNI PFC mask -> native space, via brain-to-brain affine registration."""
    cached = cache_dir / "pfc_native.nii.gz"
    if cached.is_file():
        img = nib.load(str(cached))
        log("  使用缓存的 PFC 配准结果")
        return _resample_to(np.asanyarray(img.dataobj), img.affine,
                            ref_affine, ref_shape)

    mni_path, pfc_path = find_atlases()
    if not mni_path or not pfc_path:
        raise ROIError(
            "找不到 MNI 图谱文件，无法生成前额叶 mask。\n"
            "请确认 atlases/ 下有 1MNI152_T1_1mm_brain.nii.gz 和 pfc_mask_v3.nii.gz。")

    try:
        import ants
    except ImportError as e:
        raise ROIError(
            "前额叶 ROI 需要 ANTsPy 做配准，当前环境没有安装。\n"
            "安装：pip install antspyx\n"
            "或改用「全脑」模式。") from e

    if not magnitude or not Path(magnitude).is_file():
        raise ROIError("找不到幅值图像，无法把 MNI 图谱配准到被试空间。")

    cache_dir.mkdir(parents=True, exist_ok=True)

    # Register brain to brain: the atlas is skull-stripped, so the moving
    # image must be too, otherwise the affine locks onto the skull.
    mag_img = nib.load(str(magnitude))
    mag = np.asanyarray(mag_img.dataobj, dtype=np.float32)
    bm = brain_mask
    if not same_grid(bm.shape, ref_affine, mag.shape, mag_img.affine):
        bm = _resample_to(brain_mask.astype(np.float32), ref_affine,
                          mag_img.affine, mag.shape[:3])
    brain_path = cache_dir / "mag_brain.nii.gz"
    nib.save(nib.Nifti1Image(mag * bm.astype(np.float32), mag_img.affine),
             str(brain_path))

    log("  配准 MNI → 被试空间 (ANTs affine, mattes MI) …")
    # Through ants_io, not ants directly: the cache dir sits under the subject
    # folder, whose name may be Chinese, and ITK cannot open such a path.
    from core import ants_io

    fixed = ants_io.image_read(brain_path)
    moving = ants_io.image_read(mni_path)
    atlas = ants_io.image_read(pfc_path)
    reg = ants.registration(fixed=fixed, moving=moving,
                            type_of_transform="Affine",
                            aff_metric="mattes", aff_sampling=32,
                            aff_iterations=(2100, 1200, 1200, 100),
                            aff_shrink_factors=(6, 4, 2, 1),
                            aff_smoothing_sigmas=(3, 2, 1, 0),
                            verbose=False)
    native = ants.apply_transforms(fixed=fixed, moving=atlas,
                                   transformlist=reg["fwdtransforms"],
                                   interpolator="nearestNeighbor")
    ants_io.image_write(native, cached)

    img = nib.load(str(cached))
    return _resample_to(np.asanyarray(img.dataobj), img.affine,
                        ref_affine, ref_shape)


def get_roi_mask(key: str, *, b0_shape, b0_affine,
                 brain_mask: np.ndarray,
                 subject_dir: str | os.PathLike | None = None,
                 magnitude_path: str | os.PathLike | None = None,
                 log_cb=None) -> np.ndarray:
    """Boolean mask on the subject's B0 grid for the chosen ROI."""
    log = log_cb or (lambda *_: None)
    shape = tuple(b0_shape)

    if key == "whole_brain":
        return brain_mask.astype(bool)

    sdir = Path(subject_dir) if subject_dir else None

    if key == "temporal":
        if sdir is None:
            raise ROIError("颞叶 ROI 需要知道被试目录才能找到已有的 mask。")
        found = _find_subject_mask(sdir, "*temporal_mask*.nii.gz",
                                   "*temporal*.nii.gz")
        if not found:
            raise ROIError(
                f"在 {sdir} 下没有找到颞叶 mask 文件。\n"
                "本软件未内置颞叶图谱，请先准备 *_temporal_mask.nii.gz，"
                "或改用「全脑」/「前额叶」。")
        log(f"  使用 {found.name}")
        img = nib.load(str(found))
        m = _resample_to(np.asanyarray(img.dataobj), img.affine,
                         b0_affine, shape)
        return m & brain_mask.astype(bool)

    if key == "pfc":
        if sdir is None:
            raise ROIError("前额叶 ROI 需要被试目录来缓存配准结果。")
        m = _register_pfc(Path(magnitude_path) if magnitude_path else None,
                          brain_mask, b0_affine, shape,
                          sdir / "roi_cache", log)
        m = m & brain_mask.astype(bool)
        if not m.any():
            raise ROIError(
                "配准得到的前额叶 mask 与脑掩膜没有交集，配准可能失败。\n"
                "建议改用「全脑」，或检查幅值图像质量。")
        return m

    raise ROIError(f"未知的 ROI：{key}")
