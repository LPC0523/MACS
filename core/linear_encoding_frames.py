"""One frame at a time: spatial frames, display decimation, colour limits.

IDEALIZED OFFLINE SIMULATION.

Everything here is deliberately Qt-free. The frame formulas, the decimation
rule and the colour limits are the parts that can be silently wrong in a way a
screenshot will not reveal -- a plot drawn from decimated data still looks like
a plot -- so they live where a test can reach them.

Six frames, six formulas
------------------------
    dynamic B0          C @ i_dynamic[k]
    total local B0      C @ i_total[k]
    nonlinear dynamic   R @ i_dynamic[k]
    dynamic phase       2*pi * C @ q_dynamic[k]
    total local phase   2*pi * C @ q_total[k]
    nonlinear phase     2*pi * R @ q_dynamic[k]

`C` is the full per-channel map on the ROI and `R` its nonlinear part, so the
nonlinear frames are a strict subset of the field -- never something added on.
The nonlinear ones exist to be looked at, not to be integrated into a k curve:
a field that is not linear has no single k vector, and putting it on a k plot
would assert that it does.

Only the requested sample is ever evaluated. An (Nx, Ny, Nz, Nt) array is not
built here or anywhere else; `test_linear_encoding_visualisation.py` measures
the peak allocation to keep it that way.
"""

from __future__ import annotations

import numpy as np

__all__ = ["FRAME_TYPES", "FRAME_LABELS", "frame_values", "frame_volume",
           "frame_stats", "symmetric_limits", "display_indices",
           "decimate", "axis_codes", "is_oblique", "plane_note",
           "MAX_DISPLAY_SAMPLES", "FrameCache"]

#: The six things a user can look at, in the order the selector shows them.
FRAME_TYPES = ("dynamic_field", "total_field", "nonlinear_field",
               "dynamic_phase", "total_phase", "nonlinear_phase")

FRAME_LABELS = {
    "dynamic_field": ("动态 B0 场", "Hz"),
    "total_field": ("本地线圈 static+dynamic B0 场", "Hz"),
    "nonlinear_field": ("非线性空间场分量（动态）", "Hz"),
    "dynamic_phase": ("动态相位", "rad"),
    "total_phase": ("本地线圈 static+dynamic 相位", "rad"),
    "nonlinear_phase": ("非线性空间场分量相位（动态）", "rad"),
}

#: Plot at most this many time points. Display only -- every integral, peak
#: and exported CSV uses the full series.
MAX_DISPLAY_SAMPLES = 5000

#: Colour limit used when a frame is entirely zero, so matplotlib is never
#: handed vmin == vmax.
_ZERO_FRAME_LIMIT = 1.0

NONLINEAR_NOTE = (
    "非线性空间场同样可以携带编码信息，但不能表示为单一笛卡尔 k 向量；"
    "是否对成像有益需要序列与模型化重建验证。")

K_NOTE = ("附加 B0 线性空间分量积分得到的理想化 k 偏移。"
          "不包含扫描仪内置成像梯度，不是完整扫描序列的 k-space 轨迹。")


def frame_values(model, sim, frame_type: str, index: int) -> np.ndarray:
    """(n_roi,) values for one frame. One matrix-vector product, no more."""
    if frame_type not in FRAME_TYPES:
        raise ValueError("unknown frame type %r; expected one of %s"
                         % (frame_type, ", ".join(FRAME_TYPES)))
    k = int(index)
    n = sim.t.shape[0]
    if not (0 <= k < n):
        raise IndexError("sample %d outside 0..%d" % (k, n - 1))

    if frame_type == "dynamic_field":
        return model.C @ sim.i_dynamic[k]
    if frame_type == "total_field":
        return model.C @ sim.i_total[k]
    if frame_type == "nonlinear_field":
        return model.residual @ sim.i_dynamic[k]
    if frame_type == "dynamic_phase":
        return 2.0 * np.pi * (model.C @ sim.q_dynamic[k])
    if frame_type == "total_phase":
        return 2.0 * np.pi * (model.C @ sim.q_total[k])
    return 2.0 * np.pi * (model.residual @ sim.q_dynamic[k])


def frame_volume(model, sim, frame_type: str, index: int) -> np.ndarray:
    """The same frame scattered back into one 3-D volume, NaN outside the ROI.

    NaN rather than zero: zero is a legitimate field value inside the region,
    and painting the outside with it would make an untouched area look like a
    nulled one.
    """
    return model.scatter(frame_values(model, sim, frame_type, index),
                         fill=np.nan)


def frame_stats(values, frame_type: str, index: int, time_s: float) -> dict:
    """What the caption has to say, so the number on screen is never bare."""
    v = np.asarray(values, dtype=np.float64)
    finite = v[np.isfinite(v)]
    label, unit = FRAME_LABELS.get(frame_type, (frame_type, ""))
    if finite.size == 0:
        return {"frame_type": frame_type, "label": label, "unit": unit,
                "index": int(index), "time_s": float(time_s),
                "valid_voxels": 0, "min": float("nan"), "max": float("nan"),
                "rms": float("nan"), "all_zero": True}
    return {
        "frame_type": frame_type, "label": label, "unit": unit,
        "index": int(index), "time_s": float(time_s),
        "valid_voxels": int(finite.size),
        "min": float(finite.min()), "max": float(finite.max()),
        "rms": float(np.sqrt(np.mean(finite ** 2))),
        "all_zero": bool(np.all(finite == 0.0)),
    }


def symmetric_limits(values, fallback=_ZERO_FRAME_LIMIT) -> dict:
    """Colour limits centred on zero, and a label saying they autoscale.

    A B0 map is signed and an asymmetric scale puts the colour of "no field"
    somewhere other than the middle of the bar, which reads as a bias that is
    not there. The limit is per frame, so colours are not comparable between
    time points -- the label says so rather than leaving it to be assumed.
    """
    v = np.asarray(values, dtype=np.float64)
    finite = v[np.isfinite(v)]
    lim = float(np.abs(finite).max()) if finite.size else 0.0
    if not np.isfinite(lim) or lim <= 0.0:
        lim = float(fallback)
        mode = "current-frame symmetric autoscale (frame is all zero; "
        mode += "fallback range used)"
    else:
        mode = "current-frame symmetric autoscale"
    return {"vmin": -lim, "vmax": lim, "limit": lim, "mode": mode,
            "comparable_across_time": False}


def display_indices(n_samples: int, must_include=(),
                    max_points: int = MAX_DISPLAY_SAMPLES) -> np.ndarray:
    """Deterministic decimation for plotting only.

    Keeps the first and last sample, plus anything named in `must_include` --
    the peak in particular, because a decimated plot whose peak fell between
    two kept samples understates the very number the caption quotes. Returns
    the full range untouched when it already fits.
    """
    n = int(n_samples)
    if n <= 0:
        return np.zeros(0, dtype=np.int64)
    keep = set(int(i) for i in must_include if 0 <= int(i) < n)
    keep.update({0, n - 1})
    if n <= int(max_points):
        return np.arange(n, dtype=np.int64)
    base = np.linspace(0, n - 1, int(max_points)).round().astype(np.int64)
    keep.update(base.tolist())
    return np.array(sorted(keep), dtype=np.int64)


def decimate(series, idx) -> np.ndarray:
    """Take `idx` along the time axis of a 1-D or 2-D series."""
    a = np.asarray(series)
    i = np.asarray(idx, dtype=np.int64)
    return a[i] if a.ndim == 1 else a[i, :]


def decimation_report(n_full: int, idx) -> dict:
    return {
        "full_samples": int(n_full),
        "displayed_samples": int(np.asarray(idx).size),
        "method": ("deterministic even-index decimation; first, last and the "
                   "peak sample are always kept. Display only -- every "
                   "integral, peak and exported CSV uses the full series."),
        "max_display_samples": int(MAX_DISPLAY_SAMPLES),
    }


def axis_codes(affine):
    """('R','A','S')-style codes for the three array axes."""
    try:
        import nibabel as nib

        return tuple(nib.aff2axcodes(np.asarray(affine, dtype=float)))
    except Exception:
        return ("?", "?", "?")


def is_oblique(affine, tol_deg: float = 2.0) -> bool:
    """Is any array axis more than `tol_deg` off its nearest world axis?

    An axis permutation or flip is not oblique -- the plane is still a genuine
    anatomical plane, just stored in a different order. A rotation is, and then
    the slice on screen is not sagittal, coronal or axial no matter what a
    label claims.
    """
    A = np.asarray(affine, dtype=float)[:3, :3]
    norms = np.linalg.norm(A, axis=0)
    if np.any(norms <= 0):
        return True
    cos = np.abs(A / norms)                 # |direction cosines| per column
    worst = float(np.min(cos.max(axis=0)))  # best alignment of the worst axis
    return bool(np.degrees(np.arccos(min(max(worst, -1.0), 1.0))) > tol_deg)


def plane_note(affine) -> str:
    """Caption for the slice panel. Never claims an anatomical plane it cannot."""
    codes = "".join(axis_codes(affine))
    if is_oblique(affine):
        return ("native voxel plane（轴向代码 %s）；该切面不等同于标准 RAS "
                "正交解剖平面，本页未做重采样。" % codes)
    return ("native voxel plane（轴向代码 %s）；仅为轴置换/翻转，未做重采样。"
            % codes)


class FrameCache:
    """At most `limit` frames, keyed by (result identity, type, index).

    Small on purpose. Caching the whole time series would be the 4-D array
    this design exists to avoid, just spelled differently.
    """

    def __init__(self, limit: int = 5):
        self.limit = int(limit)
        self._d = {}
        self._order = []

    def key(self, result_id, frame_type, index):
        return (str(result_id), str(frame_type), int(index))

    def get(self, k):
        if k in self._d:
            self._order.remove(k)
            self._order.append(k)
            return self._d[k]
        return None

    def put(self, k, v):
        if k in self._d:
            self._order.remove(k)
        self._d[k] = v
        self._order.append(k)
        while len(self._order) > self.limit:
            self._d.pop(self._order.pop(0), None)
        return v

    def clear(self):
        self._d.clear()
        self._order.clear()

    def __len__(self):
        return len(self._d)
