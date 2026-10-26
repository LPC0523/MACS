"""
core/data_context.py — 统一数据上下文管理 (Qt版，无Streamlit依赖)

提供 resample_to_target 和辅助函数，供其他模块使用。
B0 上下文管理由 Qt app_state dict 替代 session_state。
"""
import os, re, tempfile
import numpy as np
from scipy.ndimage import map_coordinates
import nibabel as nib


def natural_key(s):
    """自然排序 key: 'ch2' < 'ch10'"""
    return [int(t) if t.isdigit() else t for t in re.split(r'(\d+)', s)]


#: Two grids count as the same when the matrix matches *and* the affines agree.
#: Matching shape alone is not enough: a volume can have the same matrix size and
#: still sit at a different origin, in a different orientation, or on different
#: voxel spacing. Treating those as aligned silently overlays anatomy that is not
#: in the same place, and every number downstream still looks perfectly ordinary.
_AFFINE_TOL = 1e-4


def same_grid(shape_a, affine_a, shape_b, affine_b) -> bool:
    """True when two volumes occupy the same voxels in world space."""
    import numpy as _np

    if tuple(shape_a[:3]) != tuple(shape_b[:3]):
        return False
    if affine_a is None or affine_b is None:
        return False
    return bool(_np.allclose(_np.asarray(affine_a, dtype=float),
                             _np.asarray(affine_b, dtype=float),
                             atol=_AFFINE_TOL, rtol=0.0))


def resample_to_target(src_data, src_affine, ref_affine, ref_shape, order=1):
    """
    将 src_data 重采样到 ref_affine/ref_shape 定义的目标空间。

    Parameters
    ----------
    src_data   : ndarray (X, Y, Z)
    src_affine : 4x4 affine
    ref_affine : 4x4 target affine
    ref_shape  : (X, Y, Z) target shape
    order      : 0=nearest (mask), 1=trilinear (field map)

    Returns
    -------
    resampled : ndarray (ref_shape), float32
    """
    ijk = np.indices(ref_shape).reshape(3, -1)
    world = nib.affines.apply_affine(ref_affine, ijk.T)
    src_ijk = np.linalg.inv(src_affine) @ np.vstack(
        [world.T, np.ones((1, world.shape[0]))])
    src_ijk = src_ijk[:3]
    mode = 'constant' if order == 0 else 'nearest'
    cval = 0.0
    resampled = map_coordinates(
        src_data.astype(np.float32), src_ijk,
        order=order, mode=mode, cval=cval)
    return resampled.reshape(ref_shape).astype(np.float32)


def _is_temp_dir(path):
    """判断路径是否在系统临时目录下"""
    tmp = tempfile.gettempdir().lower().rstrip(os.sep)
    return path.lower().rstrip(os.sep) == tmp


def set_b0_context(app_state: dict, data, coords, label,
                   vol3d=None, affine=None, source_path=None):
    """
    所有 B0 加载路径的统一入口。

    Parameters
    ----------
    app_state   : dict, 应用全局状态 (替代 st.session_state)
    data        : 1D array, 非零体素的 B0 值
    coords      : Nx3 array, 非零体素的坐标
    label       : str, 显示用的文件名
    vol3d       : 3D array (可选)
    affine      : 4x4 affine (可选)
    source_path : str, B0 文件的真实磁盘路径 (可选)
    """
    app_state.update(
        b0_data=data,
        b0_coords=coords,
        b0_label=label,
        b0_vol3d=vol3d,
        b0_affine=affine,
    )

    if source_path and os.path.isfile(source_path):
        d = os.path.dirname(os.path.abspath(source_path))
        if not _is_temp_dir(d):
            app_state["subject_dir"] = d


def get_subject_dir(app_state: dict):
    """获取被试文件夹路径。"""
    d = app_state.get("subject_dir")
    if d and os.path.isdir(d):
        return d
    return None
