"""
core/dicom_to_nifti.py
======================
Pure-Python pydicom DICOM -> NIfTI converter.

There is deliberately no dcm2niix path. That binary converted whole folders in
one subprocess call, which bypassed the SeriesDescription/ProtocolName filter
below -- so whether an EPI or scout series reached the phase/magnitude search
depended on whether dcm2niix happened to be installed. One code path means the
filter always applies, and development and packaged builds behave identically.
"""

from __future__ import annotations
import os
import sys
from pathlib import Path
import numpy as np
import nibabel as nib


def _safe_print(msg) -> None:
    """print() that can never abort the caller.

    The console on a Chinese Windows is GBK, and these logs contain check
    marks; a bare print then raises UnicodeEncodeError from inside the
    conversion loop and kills the whole B0 run. Logging is diagnostics -- it
    must never be able to fail the computation it is describing.
    """
    try:
        print(msg)
    except UnicodeEncodeError:
        try:
            enc = (getattr(sys.stdout, "encoding", None) or "ascii")
            print(str(msg).encode(enc, "replace").decode(enc, "replace"))
        except Exception:
            pass
    except Exception:
        pass


# ─────────────────────────────────────────────────────────────────────────
#  DICOM reading helpers
# ─────────────────────────────────────────────────────────────────────────

def _get(ds, tag, default=None):
    val = getattr(ds, tag, None)
    if val is None:
        return default
    if hasattr(val, "__iter__") and not isinstance(val, (str, bytes)):
        return list(val)
    return val

def _image_type(ds):
    return [str(x).upper() for x in _get(ds, "ImageType", [])]

def _series_id(ds):
    sn = str(_get(ds, "SeriesNumber", "0"))
    en = str(_get(ds, "EchoNumbers",  "1"))
    it = "_".join(_image_type(ds)[:2])
    return f"{sn}_echo{en}_{it}"

def _slice_position(ds):
    ipp = _get(ds, "ImagePositionPatient")
    iop = _get(ds, "ImageOrientationPatient")
    if ipp and iop:
        iop = np.array(iop, dtype=float)
        n   = np.cross(iop[:3], iop[3:])
        return float(np.dot(n, np.array(ipp, dtype=float)))
    return float(_get(ds, "InstanceNumber", 0))

def _pixel_data(ds):
    arr = ds.pixel_array.astype(np.float32)
    return arr * float(_get(ds, "RescaleSlope", 1.0)) + float(_get(ds, "RescaleIntercept", 0.0))

#: DICOM patient coordinates are LPS, NIfTI world coordinates are RAS.
_LPS_TO_RAS = np.array([-1.0, -1.0, 1.0])


def _build_affine(slices):
    """Voxel -> RAS world affine for a stack of DICOM slices.

    The array is assembled as [row, column, slice] (see `_run_pydicom`), so:

      * column 0 of the affine is the direction of increasing ROW index, which
        DICOM stores in ImageOrientationPatient[3:6] -- not [0:3];
      * column 1 is the direction of increasing COLUMN index, [0:3];
      * PixelSpacing is [row spacing, column spacing], matching that order;
      * everything is converted LPS -> RAS.

    Getting the first two columns the wrong way round silently transposes the
    anatomy: axis 0 becomes anterior-posterior while the header claims it is
    left-right, so every downstream geometric step -- atlas registration, coil
    resampling, ROI masks -- lands rotated, while voxel-space checks still look
    perfectly self-consistent.
    """
    ds0 = slices[0]
    ps = _get(ds0, "PixelSpacing", [1.0, 1.0])
    row_spacing, col_spacing = float(ps[0]), float(ps[1])
    iop = _get(ds0, "ImageOrientationPatient")

    if iop and len(iop) == 6:
        iop = np.array(iop, dtype=float)
        col_dir = iop[:3]                 # along a row: column index increases
        row_dir = iop[3:]                 # down a column: row index increases

        ipp0 = np.array(_get(ds0, "ImagePositionPatient", [0, 0, 0]), dtype=float)
        if len(slices) > 1:
            ipp_last = np.array(
                _get(slices[-1], "ImagePositionPatient", [0, 0, 0]), dtype=float)
            # Signed step: a norm would always come out positive and silently
            # flip the slice axis for descending acquisitions.
            slice_vec = (ipp_last - ipp0) / (len(slices) - 1)
        else:
            slice_vec = np.cross(col_dir, row_dir) * float(
                _get(ds0, "SliceThickness", 1.0))

        aff = np.eye(4)
        aff[:3, 0] = row_dir * row_spacing * _LPS_TO_RAS
        aff[:3, 1] = col_dir * col_spacing * _LPS_TO_RAS
        aff[:3, 2] = slice_vec * _LPS_TO_RAS
        aff[:3, 3] = ipp0 * _LPS_TO_RAS
    else:
        aff = np.diag([row_spacing, col_spacing,
                       float(_get(ds0, "SliceThickness", 1.0)), 1.0])
    return aff.astype(np.float64)

#: A B0 mapping acquisition names itself one of these. Matched case-insensitively.
#
#: The lookarounds are deliberately NOT `\b`. Underscore is a regex word
#: character, so `\bb0map\b` fails on the real series name `b0map_tra_shim` and
#: on `..._B0Map_ph` -- it would reject exactly the data we need. What must be
#: excluded is a name glued on in front (`PALB0Map_M`, `CalB0Map` are not field
#: maps), so the left edge breaks on letters while `_`, spaces and digits
#: separate. The right edge blocks only letters, so `B0Map2` still counts.
#
#: The left edge used to break on digits too, which silently dropped every
#: magnitude echo of a protocol called `B0fieldmap`: the `0` sits directly in
#: front of `field`, so `field\s*map` never matched, and only the phase series
#: survived because its own name carried a separate `_B0Map`. The run then
#: failed with "Could not identify phase/magnitude images", and -- worse, since
#: it was not an error -- delta-TE could no longer be read from the two echoes
#: and quietly fell back to whatever the page had typed in it.
_B0_SERIES_PATTERN = r"(?<![A-Za-z])(?:b0\s*map|field\s*map|gre.?field)(?![A-Za-z])"


def _is_b0_series(ds) -> bool | None:
    """Does this slice belong to a B0 mapping acquisition?

    True / False when SeriesDescription or ProtocolName says so, None when the
    tags are absent -- older exports get no filtering rather than being thrown
    away on missing metadata.
    """
    import re as _re

    desc = str(_get(ds, "SeriesDescription", "") or "")
    proto = str(_get(ds, "ProtocolName", "") or "")
    if not desc and not proto:
        return None
    pat = _re.compile(_B0_SERIES_PATTERN, _re.I)
    # ProtocolName covers the whole acquisition, so the magnitude echoes of a
    # B0 protocol are kept even when their own SeriesDescription is just
    # "..._Echo1". Filtering on SeriesDescription alone would drop them and
    # leave the phase map with nothing to pair with.
    return bool(pat.search(desc) or pat.search(proto))


def _filter_b0_series(datasets, log):
    """Drop series that are not part of a B0 mapping protocol.

    An EPI run also has Echo1/Echo2, and a scout also converts cleanly; keeping
    them means the phase/magnitude search downstream can pair the wrong images.
    """
    keep, dropped, unknown = [], {}, 0
    for ds in datasets:
        verdict = _is_b0_series(ds)
        if verdict is None:
            unknown += 1
            keep.append(ds)
            continue
        if verdict:
            keep.append(ds)
        else:
            name = str(_get(ds, "SeriesDescription", "") or
                       _get(ds, "ProtocolName", "") or "?")
            dropped[name] = dropped.get(name, 0) + 1

    if dropped:
        log("  skipped non-B0 series: " +
            ", ".join(f"{k} ({v} slices)" for k, v in sorted(dropped.items())))
    if unknown:
        log(f"  {unknown} slice(s) have no series description; kept unfiltered")
    if not keep:
        names = ", ".join(sorted(dropped)) or "none"
        raise RuntimeError(
            "这些 DICOM 里没有找到 B0 mapping 序列。\n"
            f"看到的序列：{names}\n"
            "请确认拖入的是双回波 GRE 场图序列（幅值 + 相位）。")
    return keep


def _write_sidecar(ds, nifti_path: str, shape) -> None:
    """Write a small JSON next to the NIfTI describing which series it is.

    Filenames alone cannot tell a B0 mapping series from an EPI run that also
    happens to have two echoes, so downstream selection reads SeriesDescription
    from here; we write it ourselves because nothing else produces one.
    """
    import json as _json

    meta = {
        "SeriesDescription": str(_get(ds, "SeriesDescription", "")),
        "ProtocolName": str(_get(ds, "ProtocolName", "")),
        "SeriesNumber": _get(ds, "SeriesNumber", None),
        "EchoNumbers": _get(ds, "EchoNumbers", None),
        "EchoTime": _get(ds, "EchoTime", None),
        "ImageType": [str(x) for x in (_get(ds, "ImageType", []) or [])],
        "Modality": str(_get(ds, "Modality", "")),
        "StudyInstanceUID": str(_get(ds, "StudyInstanceUID", "")),
        "SeriesInstanceUID": str(_get(ds, "SeriesInstanceUID", "")),
        "MagneticFieldStrength": _get(ds, "MagneticFieldStrength", None),
        # (0018,0084), MHz. The transmit frequency this field map was acquired
        # at, and therefore the zero of the deltaB0 it produces -- the centre
        # frequency the system-shim page reports is an offset from this number
        # and from no other. See `probe_acquisition`.
        "ImagingFrequency": _get(ds, "ImagingFrequency", None),
        "Shape": [int(x) for x in shape],
        "ConvertedBy": "MACS pydicom converter",
    }
    for k, v in list(meta.items()):
        # pydicom returns its own numeric types; make them JSON-safe.
        if v is not None and not isinstance(v, (str, list, int, float, bool)):
            try:
                meta[k] = float(v)
            except (TypeError, ValueError):
                meta[k] = str(v)

    side = nifti_path
    for suf in (".nii.gz", ".nii"):
        if side.endswith(suf):
            side = side[: -len(suf)]
            break
    try:
        with open(side + ".json", "w", encoding="utf-8") as f:
            _json.dump(meta, f, indent=2, ensure_ascii=False)
    except OSError:
        pass        # metadata is helpful, not required


def sidecar_frequency_hz(nifti_path: str):
    """Transmit frequency of one converted series, from its sidecar JSON.

    `probe_acquisition` answers "what frequency did this folder use", which has
    no answer once the folder holds more than one acquisition -- and a shim
    session is exactly that. Measured on a 5T phantom session: 21 accepted B0
    acquisitions over 90 minutes spanning 793 Hz, because every f0 calibration
    in between moved it. The folder-wide question is therefore the wrong one;
    the field map that was actually processed has one frequency, and this reads
    that one.

    Returns None if the sidecar is missing or has no frequency, so a caller can
    fall back rather than substitute a neighbouring acquisition's f0 -- which
    on the session above would be wrong by up to 793 Hz.
    """
    import json as _json

    if not nifti_path:
        return None
    side = str(nifti_path)
    for suf in (".nii.gz", ".nii"):
        if side.endswith(suf):
            side = side[: -len(suf)]
            break
    try:
        with open(side + ".json", encoding="utf-8") as f:
            mhz = _json.load(f).get("ImagingFrequency")
    except (OSError, ValueError):
        return None
    try:
        return float(round(float(mhz) * 1e6)) if mhz else None
    except (TypeError, ValueError):
        return None


def probe_acquisition(dicom_dir: str, log_cb=None) -> dict:
    """Read ΔTE, field strength and transmit frequency out of the DICOM headers.

    All three are facts of the acquisition, recorded in every slice. Asking the
    user to type them is asking them to re-enter something the data already
    knows, and a wrong ΔTE is invisible in the result: the field map keeps its
    shape and only its scale is wrong, so the currents come out proportionally
    wrong with nothing on screen looking odd.

    `imaging_frequency_hz` is (0018,0084) converted from MHz. It matters for a
    different reason than the other two -- nothing in the solve uses it. It is
    the zero of the field map: deltaB0 is local resonance minus *this*
    frequency, so the centre-frequency offset the system-shim page reports is
    an offset from this number specifically. The console's displayed f0 is not
    a substitute, because a prescan between the field map and the shim entry
    re-locks it, and adding the offset to a re-locked f0 double-counts the
    pre-shim mean by however much the prescan moved.

    Read only from slices `_is_b0_series` accepts, so an EPI run in the same
    study cannot contribute its own f0.

    Pixel data is not read, so this is a fast pass even over a full study.
    Returns {delta_te_s, echo_times_ms, field_t, imaging_frequency_hz,
    n_slices}; any value that could not be determined is None rather than a
    plausible default.
    """
    log = log_cb or (lambda *_: None)
    out = {"delta_te_s": None, "echo_times_ms": [], "field_t": None,
           "imaging_frequency_hz": None, "n_slices": 0}
    try:
        import pydicom
    except ImportError:
        return out

    echoes, fields, freqs, n = set(), set(), set(), 0
    for path in (str(q) for q in Path(dicom_dir).rglob("*") if q.is_file()):
        try:
            ds = pydicom.dcmread(path, stop_before_pixels=True, force=True)
        except Exception:
            continue
        if not hasattr(ds, "SOPClassUID") and not hasattr(ds, "Modality"):
            continue
        if _is_b0_series(ds) is False:
            continue                    # an EPI's echo times are not our ΔTE
        n += 1
        te = _get(ds, "EchoTime")
        if te is not None:
            try:
                echoes.add(round(float(te), 4))
            except (TypeError, ValueError):
                pass
        f = _get(ds, "MagneticFieldStrength")
        if f is not None:
            try:
                fields.add(round(float(f), 3))
            except (TypeError, ValueError):
                pass
        imf = _get(ds, "ImagingFrequency")
        if imf is not None:
            try:
                # MHz -> Hz. Rounded to 1 Hz before de-duplication: the tag is
                # a DS, so the stored decimals are the real resolution, and
                # comparing raw floats would split one frequency into several
                # over a series purely on the last digit.
                freqs.add(round(float(imf) * 1e6))
            except (TypeError, ValueError):
                pass

    out["n_slices"] = n
    out["echo_times_ms"] = sorted(echoes)
    if len(out["echo_times_ms"]) >= 2:
        te1, te2 = out["echo_times_ms"][0], out["echo_times_ms"][-1]
        out["delta_te_s"] = (te2 - te1) / 1000.0
        log(f"  ΔTE 来自 DICOM：TE {te1:g} / {te2:g} ms → "
            f"ΔTE = {out['delta_te_s'] * 1000:.3f} ms")
    elif out["echo_times_ms"]:
        log(f"  DICOM 里只有一个回波时间（{out['echo_times_ms'][0]:g} ms），"
            "无法推出 ΔTE")

    if len(fields) == 1:
        out["field_t"] = fields.pop()
        log(f"  场强来自 DICOM：{out['field_t']:g} T")
    elif len(fields) > 1:
        log(f"  DICOM 里有多个场强值 {sorted(fields)}，不采用")

    if len(freqs) == 1:
        out["imaging_frequency_hz"] = float(freqs.pop())
        log(f"  采集发射频率来自 DICOM：{out['imaging_frequency_hz'] / 1e6:.6f} MHz")
    elif len(freqs) > 1:
        # Two echoes of one field map share a frequency; several values mean
        # several acquisitions, which is the normal shape of a shim session
        # rather than a fault. There is then no folder-wide answer, so None is
        # returned and `sidecar_frequency_hz` resolves it per acquisition. The
        # span is logged because it is the size of the error someone would make
        # by reaching for the console's current f0 instead.
        span = max(freqs) - min(freqs)
        log(f"  文件夹内有 {len(freqs)} 个不同的采集频率（相差 {span} Hz），"
            "改为按实际处理的那一组场图取值")
    return out


def _series_filename(ds):
    desc = str(_get(ds, "SeriesDescription", "series")).strip()
    for ch in r'/\\:*?"<>| ': desc = desc.replace(ch, "_")
    it   = _image_type(ds)
    en   = str(_get(ds, "EchoNumbers", "1"))
    is_ph = "P" in it or "PHASE" in it or "phase" in desc.lower()
    sfx   = (f"_e{en}" if en != "1" else "") + ("_ph" if is_ph else "")
    return f"{desc}{sfx}"

def _acquisition_tag(ds):
    """Which acquisition a series belongs to, when several share a name.

    Only consulted to break a filename collision. A console that repeats one
    protocol numbers the components of acquisition N as N01/N02/N03 -- Echo1,
    Echo2, phase -- so integer division recovers the acquisition. Verified on a
    5T export: 24 acquisitions, every one holding exactly xx01/xx02/xx03, and
    the transmit frequency constant inside each.

    Returns a prefix that is *identical* for the components of one acquisition,
    which is what lets `_acquisition_key` still pair phase with magnitude. When
    the numbering does not fit that shape the series number is used whole: two
    files then never pair and the run fails saying so, which is the right
    outcome -- a field map built from two different acquisitions is wrong in a
    way no downstream check would catch.
    """
    try:
        sn = int(_get(ds, "SeriesNumber", 0))
    except (TypeError, ValueError):
        return "s0"
    return f"a{sn // 100:02d}" if sn >= 100 else f"s{sn}"


def _run_pydicom(dicom_dir: str, output_dir: str, log_cb=None) -> None:
    def _log(msg):
        if log_cb: log_cb(msg)
        _safe_print(msg)

    try:
        import pydicom
    except ImportError:
        raise ImportError(
            "DICOM 读取需要 pydicom。\n"
            "安装：  pip install pydicom"
        )

    os.makedirs(output_dir, exist_ok=True)
    candidates = [str(p) for p in Path(dicom_dir).rglob("*") if p.is_file()]
    _log(f"  Scanning {len(candidates)} files …")

    datasets = []
    for path in candidates:
        try:
            ds = pydicom.dcmread(path, stop_before_pixels=False, force=True)
            if hasattr(ds, "pixel_array"):
                datasets.append(ds)
        except Exception:
            continue

    if not datasets:
        raise FileNotFoundError(f"No readable DICOM files in {dicom_dir}")

    datasets = _filter_b0_series(datasets, _log)

    _log(f"  {len(datasets)} slices, grouping into series …")
    groups: dict[str, list] = {}
    for ds in datasets:
        groups.setdefault(_series_id(ds), []).append(ds)

    # Series are grouped by SeriesNumber, but named by SeriesDescription, and
    # those are not the same thing: a session that repeats one protocol gives
    # every repeat the same description. Writing them all to that one filename
    # silently keeps whichever happened to be converted last, and since the
    # three components are written independently, the survivors need not even
    # come from the same acquisition -- a field map built from two different
    # ones, with nothing on screen to show for it. Measured on a 5T session:
    # 72 accepted B0 series collapsing onto 18 names, one description repeated
    # 11 times. So collisions are detected first and every affected series gets
    # its acquisition as a prefix; folders holding a single acquisition, which
    # is what this app is normally handed, keep the names they had.
    ordered = {}
    for key, slices in groups.items():
        sl = sorted(slices, key=_slice_position)
        ordered[key] = (sl, _series_filename(sl[0]))

    seen: dict[str, int] = {}
    for _, name in ordered.values():
        seen[name] = seen.get(name, 0) + 1
    clashing = {n for n, c in seen.items() if c > 1}
    if clashing:
        _log(f"  [!] {len(clashing)} 个序列名重复（{len(ordered)} 个序列 → "
             f"{len(seen)} 个文件名），按采集编号加前缀以免互相覆盖")
        ordered = {k: (sl, f"{_acquisition_tag(sl[0])}_{name}")
                   for k, (sl, name) in ordered.items()}

    for key, (sl, name) in ordered.items():
        try:
            vol  = np.stack([_pixel_data(s) for s in sl], axis=-1)
            aff  = _build_affine(sl)
            out  = os.path.join(output_dir, f"{name}.nii.gz")
            nib.save(nib.Nifti1Image(vol.astype(np.float32), aff), out)
            _write_sidecar(sl[0], out, vol.shape)
            _log(f"  ✓ {name}.nii.gz  {vol.shape}")
        except Exception as e:
            _log(f"  ✗ series {key}: {e}")


# ─────────────────────────────────────────────────────────────────────────
#  Public API
# ─────────────────────────────────────────────────────────────────────────

def dicom_folder_to_nifti(dicom_dir: str, output_dir: str, log_cb=None) -> None:
    """Convert one DICOM folder -> NIfTI (+ JSON sidecar per series)."""
    _run_pydicom(dicom_dir, output_dir, log_cb)
