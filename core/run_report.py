"""Provenance for one shim run: `result.json`, written next to the outputs.

A set of currents on its own is not reviewable. Months later the questions are
always the same -- which DICOM folder, which ΔTE, which coil calibration, which
brain mask, was it actually uploaded -- and none of them can be answered from
the numbers. This records the whole chain at the moment it ran.

Everything here is read back from what the run actually produced. Nothing is
recomputed or assumed: a field that could not be determined is written as null
rather than filled with a plausible default, because a provenance file that
guesses is worse than one that admits a gap.
"""

from __future__ import annotations

import datetime
import json
import os
from pathlib import Path

import numpy as np

REPORT_NAME = "result.json"
SCHEMA_VERSION = 1


def _now() -> str:
    return datetime.datetime.now().replace(microsecond=0).isoformat()


def _jsonable(x):
    """numpy scalars and arrays are not JSON-serialisable on their own."""
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, np.ndarray):
        return [_jsonable(v) for v in x.tolist()]
    if isinstance(x, Path):
        return str(x)
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    return x


def run_dir(result, root: str | os.PathLike | None = None) -> Path:
    """`runs/<YYYYMMDD>/<HHMMSS>_<subject>/` under the subject's output folder."""
    now = datetime.datetime.now()
    base = Path(root) if root else Path(result.nifti_dir)
    subject = _safe(result.subject or "subject")
    return base / "runs" / now.strftime("%Y%m%d") / f"{now.strftime('%H%M%S')}_{subject}"


def _safe(name: str) -> str:
    for ch in r'/\:*?"<>| ':
        name = name.replace(ch, "_")
    return name or "subject"


def _read_sidecars(nifti_dir) -> dict:
    """SeriesDescription etc. from the JSON sidecars the converter wrote."""
    out = {"converted": [], "protocol_name": None, "echo_times_ms": [],
           "study_uid": None}
    d = Path(nifti_dir)
    if not d.is_dir():
        return out
    for js in sorted(d.glob("*.json")):
        if js.name == REPORT_NAME:
            continue
        try:
            meta = json.loads(js.read_text(encoding="utf-8"))
        except Exception:
            continue
        if "SeriesDescription" not in meta:
            continue        # not one of ours
        out["converted"].append(meta.get("SeriesDescription") or js.stem)
        out["protocol_name"] = out["protocol_name"] or (meta.get("ProtocolName") or None)
        out["study_uid"] = out["study_uid"] or (meta.get("StudyInstanceUID") or None)
        te = meta.get("EchoTime")
        if te is not None:
            try:
                out["echo_times_ms"].append(float(te))
            except (TypeError, ValueError):
                pass
    out["echo_times_ms"] = sorted(set(out["echo_times_ms"]))
    return out


def build(result, *, dicom_dir, delta_te, field_t, lambda_penalty,
          coil=None, bet_info=None) -> dict:
    """Assemble the report dict. Pure -- writes nothing."""
    from core.version import build_info

    info = build_info()
    side = _read_sidecars(result.nifti_dir)

    # `evaluate_masks` reports ROI and whole-brain separately; the headline
    # numbers are the ROI, since that is what was optimised.
    stats = result.stats or {}
    before = stats.get("roi_rmse_before")
    after = stats.get("roi_rmse_after")
    improvement = None
    if before and after is not None and before > 0:
        improvement = round(100.0 * (before - after) / before, 2)

    currents = []
    if result.currents is not None:
        names = list(result.channel_names or [])
        for i, amps in enumerate(np.asarray(result.currents).ravel()):
            currents.append({
                "channel": i + 1,
                "name": names[i] if i < len(names) else f"ch{i + 1}",
                "current_a": round(float(amps), 6),
            })

    mode = getattr(result, "mode", "static")

    report = {
        "schema": SCHEMA_VERSION,
        "mode": mode,
        "version": info["version"],
        "build": info["build_date"],
        "git_sha": info["git_sha"],
        "timestamp": _now(),

        "subject": {
            "id": result.subject or None,
            "study_uid": side["study_uid"],
            "dicom_folder": str(dicom_dir),
            "nifti_folder": str(result.nifti_dir),
        },

        "acquisition": {
            "delta_te_s": float(delta_te),
            "field_strength_t": float(field_t),
            "protocol_name": side["protocol_name"],
            # The zero of the field map (DICOM 0018,0084). Recorded because the
            # centre-frequency offset the system-shim page reports is an offset
            # from this frequency and no other, and by the time anyone reads
            # this report the console has long since re-locked its own f0.
            "imaging_frequency_hz": (
                float(result.imaging_frequency_hz)
                if getattr(result, "imaging_frequency_hz", None) else None),
            "converted_series": side["converted"],
            "echo_times_ms": side["echo_times_ms"],
        },

        "bet": dict(bet_info or {}),

        "roi": {
            "name": result.roi_name,
            "voxels": int(result.roi_mask.sum()) if result.roi_mask is not None else None,
            "brain_voxels": int(result.brain_mask.sum()) if result.brain_mask is not None else None,
        },

        "optimization": {
            "coil": result.coil_name,
            "channels": len(result.channel_names or []),
            "current_limit_a": float(getattr(coil, "current_limit_a", 0.0)) or None,
            "lambda": float(lambda_penalty),
            "rmse_before_hz": _round(before),
            "rmse_after_hz": _round(after),
            "improvement_percent": improvement,
            "b0_std_hz": _round(getattr(result, "b0_std_hz", None)),
            "channels_at_limit": int(result.channels_at_limit or 0),
            # The full before/after table, so a later comparison does not have
            # to re-derive std and peak-to-peak from the NIfTI files.
            "metrics": {k: _round(v) for k, v in stats.items()},
        },

        "currents": currents,

        "warnings": list(getattr(result, "warnings", []) or []),
        "durations_s": {k: round(float(v), 2) for k, v in (result.seconds or {}).items()},

        # Filled in by `record_upload` once the amplifier confirms.
        "upload": None,
    }
    if mode == "dynamic" and result.slice_currents is not None:
        per_slice = []
        for z in range(result.slice_currents.shape[0]):
            per_slice.append({
                "slice": z + 1,
                "currents_a": [round(float(v), 6) for v in result.slice_currents[z]],
                "rmse_hz": _round(result.slice_rmse_after[z]),
                "rmse_before_hz": _round(result.slice_rmse_before[z]),
                "voxels": int(result.slice_voxels[z]),
                "solved": bool(result.slice_solved[z]),
            })
        solved = result.slice_solved
        after = result.slice_rmse_after[solved] if solved.any() else []
        before = result.slice_rmse_before[solved] if solved.any() else []
        report["dynamic"] = {
            "axis": int(result.slice_axis),
            "n_slices": int(result.slice_currents.shape[0]),
            "n_channels": int(result.slice_currents.shape[1]),
            "slices_solved": int(solved.sum()),
            "slices_skipped": int((~solved).sum()),
            # The whole-brain figures the two modes can be compared on; the
            # per-slice mean is a different quantity and is labelled as such.
            "rmse_whole_brain_before_hz": _round(stats.get("roi_rmse_before")),
            "rmse_whole_brain_after_hz": _round(stats.get("roi_rmse_after")),
            "rmse_slice_mean_before_hz": _round(np.mean(before)) if len(before) else None,
            "rmse_slice_mean_after_hz": _round(np.mean(after)) if len(after) else None,
            "currents_csv": "dynamic_currents.csv",
        }
        report["currents_per_slice"] = per_slice

    return _jsonable(report)


def _round(x, nd: int = 2):
    try:
        return round(float(x), nd)
    except (TypeError, ValueError):
        return None


def write(report: dict, directory) -> Path:
    """Write result.json into `directory`, creating it if needed."""
    d = Path(directory)
    d.mkdir(parents=True, exist_ok=True)
    path = d / REPORT_NAME
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False),
                    encoding="utf-8")
    return path


def save(result, **kwargs) -> Path:
    """build() + write() into the run directory for this result."""
    root = kwargs.pop("root", None)
    d = run_dir(result, root)
    return write(build(result, **kwargs), d)


def record_upload(report_path, *, status: str, channels_written: int = 0,
                  dac_readback=None, failed_channels=None,
                  error: str = "") -> None:
    """Fill in the `upload` block after writing to the amplifier.

    Kept separate from `save` because the upload happens later, from a
    different page, and may never happen at all -- `upload: null` is a
    meaningful state meaning "computed but not applied".
    """
    p = Path(report_path)
    try:
        report = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return

    block = {
        "timestamp": _now(),
        "status": status,                       # ok | partial | failed
        "channels_written": int(channels_written),
    }
    if dac_readback is not None:
        block["dac_readback"] = _jsonable(dac_readback)
    if failed_channels:
        block["failed_channels"] = _jsonable(failed_channels)
    if error:
        block["error"] = error[:1000]

    report["upload"] = block
    try:
        p.write_text(json.dumps(report, indent=2, ensure_ascii=False),
                     encoding="utf-8")
    except OSError:
        pass
