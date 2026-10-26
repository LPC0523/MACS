"""The field map's transmit frequency, read from DICOM (0018,0084).

    D:\\python.exe tests/test_imaging_frequency.py

This exists because the number it guards is one an operator types into a
console. The system-shim page reports a centre-frequency offset, and that
offset is meaningless without the frequency it is measured from: deltaB0 is
local resonance minus the frequency the *field map* was acquired at, not minus
whatever f0 the console currently displays. Any prescan in between re-locks the
console onto the unshimmed field's mean, and adding the offset there
double-counts that mean.

Measured, on a 5T phantom calibration folder (two phantoms, three table
positions, 21 accepted B0 acquisitions over 105 minutes): the tag is present on
every b0map series and absent on the vendor's PAL adjustment series, stored to
six decimals in MHz, i.e. 1 Hz. Accepted acquisitions span 823 Hz across the
folder. Most of that is legitimate -- a different phantom at a different table
position resonates elsewhere -- which is the point: there is no folder-wide f0
to read, so the frequency has to follow the pair that was processed. Within the
one table position that matters for the shim workflow the two field maps nine
minutes apart agree exactly, so this is not a claim that f0 always drifts.

So the tests are about provenance, not arithmetic:
  1. the frequency is read, and converted MHz -> Hz
  2. an EPI in the same study cannot contribute its own, different f0
  3. a folder with several acquisitions has no single answer
  4. a missing tag yields None rather than a plausible default
  5. but the processed pair always has one, via its sidecar
  6. and the whole point: which baseline the offset attaches to
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core.dicom_to_nifti import probe_acquisition

FAILS = []


def ok(cond, msg):
    print("  [%s] %s" % ("OK" if cond else "FAIL", msg))
    if not cond:
        FAILS.append(msg)


def _write(path, series, freq_mhz, te, *, with_freq=True):
    """One minimal DICOM slice. No pixel data -- probe never reads any."""
    import pydicom
    from pydicom.dataset import Dataset, FileMetaDataset
    from pydicom.uid import ExplicitVRLittleEndian

    ds = Dataset()
    ds.file_meta = FileMetaDataset()
    ds.file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    ds.file_meta.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.4"
    ds.file_meta.MediaStorageSOPInstanceUID = pydicom.uid.generate_uid()
    ds.SOPClassUID = "1.2.840.10008.5.1.4.1.1.4"
    ds.SOPInstanceUID = ds.file_meta.MediaStorageSOPInstanceUID
    ds.Modality = "MR"
    ds.SeriesDescription = series
    ds.ProtocolName = series
    ds.EchoTime = te
    ds.MagneticFieldStrength = 3.0
    if with_freq:
        # DS with six decimals in MHz, i.e. 1 Hz resolution -- what the
        # console actually stores.
        ds.ImagingFrequency = "%.6f" % freq_mhz
    ds.is_little_endian = True
    ds.is_implicit_VR = False
    ds.save_as(str(path), write_like_original=False)


def test_reads_and_converts():
    print("1. read + MHz -> Hz")
    with tempfile.TemporaryDirectory() as d:
        for i, te in enumerate((4.92, 7.38)):
            _write(Path(d) / ("b0_%d.dcm" % i), "gre_field_mapping",
                   123.253430, te)
        out = probe_acquisition(d)
        ok(out["imaging_frequency_hz"] == 123253430.0,
           "123.253430 MHz -> %r Hz" % out["imaging_frequency_hz"])
        ok(abs(out["delta_te_s"] - 0.00246) < 1e-9,
           "delta_te still read alongside (%s s)" % out["delta_te_s"])


def test_ignores_non_b0_series():
    print("2. an EPI's f0 is not the field map's")
    with tempfile.TemporaryDirectory() as d:
        _write(Path(d) / "b0_0.dcm", "gre_field_mapping", 123.253430, 4.92)
        _write(Path(d) / "b0_1.dcm", "gre_field_mapping", 123.253430, 7.38)
        # Same study, re-locked frequency, not a field map.
        _write(Path(d) / "epi_0.dcm", "restfMRI_mb8_PA_1.8mm", 123.253512, 30.0)
        out = probe_acquisition(d)
        ok(out["imaging_frequency_hz"] == 123253430.0,
           "EPI at 123.253512 ignored, got %r" % out["imaging_frequency_hz"])


def test_conflict_is_none():
    print("3. two field maps at different f0 -> None")
    with tempfile.TemporaryDirectory() as d:
        _write(Path(d) / "a_0.dcm", "B0fieldmap_ref", 123.253430, 4.92)
        _write(Path(d) / "b_0.dcm", "B0fieldmap_shim", 123.253570, 4.92)
        lines = []
        out = probe_acquisition(d, log_cb=lines.append)
        ok(out["imaging_frequency_hz"] is None,
           "ambiguous -> None, got %r" % out["imaging_frequency_hz"])
        ok(any("采集频率" in l and "相差" in l for l in lines),
           "the spread is reported, since it sizes the error someone would "
           "make by reaching for the console's f0 instead")


def test_missing_is_none():
    print("4. tag absent -> None, not a default")
    with tempfile.TemporaryDirectory() as d:
        for i, te in enumerate((4.92, 7.38)):
            _write(Path(d) / ("b0_%d.dcm" % i), "gre_field_mapping",
                   0.0, te, with_freq=False)
        out = probe_acquisition(d)
        ok(out["imaging_frequency_hz"] is None,
           "missing -> None, got %r" % out["imaging_frequency_hz"])
        ok(out["field_t"] == 3.0, "other fields unaffected")


def test_sidecar_resolves_one_acquisition():
    print("5. a folder of many acquisitions resolves per pair")
    # The shape of a real calibration folder: several field maps, each at its
    # own frequency, because the table moved and the phantom changed between
    # them. Folder-wide there is no answer; per acquisition there always is.
    import json

    with tempfile.TemporaryDirectory() as d:
        made = {}
        for name, mhz in (("b0map_tra_shim_Echo2", 210.912262),
                          ("b0map_tra_f0cal_Echo2", 210.911469)):
            p = Path(d) / (name + ".nii.gz")
            p.write_bytes(b"")
            (Path(d) / (name + ".json")).write_text(
                json.dumps({"SeriesDescription": name,
                            "ImagingFrequency": "%.6f" % mhz}),
                encoding="utf-8")
            made[name] = str(p)

        from core.dicom_to_nifti import sidecar_frequency_hz
        ok(sidecar_frequency_hz(made["b0map_tra_shim_Echo2"]) == 210912262.0,
           "the processed pair's own f0 is returned")
        ok(sidecar_frequency_hz(made["b0map_tra_f0cal_Echo2"]) == 210911469.0,
           "a different acquisition in the same folder keeps its own")
        ok(sidecar_frequency_hz(str(Path(d) / "absent.nii.gz")) is None,
           "no sidecar -> None")


def test_target_is_offset_from_the_map_not_the_console():
    print("6. the reason this is read at all")
    # Numbers stand in for one acquisition: the map was taken at f0_map, its
    # residual mean after shimming is `offset`, and a prescan in between moved
    # the console to f0_console by re-locking onto the unshimmed mean.
    f0_map, offset, preshim_mean = 123253430.0, 23.0, 61.0
    f0_console = f0_map + preshim_mean

    correct = f0_map + offset
    naive = f0_console + offset
    ok(naive - correct == preshim_mean,
       "adding the offset to the console's f0 overshoots by the pre-shim "
       "mean (%.0f Hz), which is why the map's own f0 is carried" % preshim_mean)
    ok(abs(naive - correct) > abs(offset),
       "and the error (%.0f Hz) exceeds the offset itself (%.0f Hz)"
       % (naive - correct, offset))


if __name__ == "__main__":
    try:
        import pydicom                                            # noqa: F401
    except ImportError:
        print("pydicom not installed -- skipping")
        sys.exit(0)
    test_reads_and_converts()
    test_ignores_non_b0_series()
    test_conflict_is_none()
    test_missing_is_none()
    test_sidecar_resolves_one_acquisition()
    test_target_is_offset_from_the_map_not_the_console()
    print()
    if FAILS:
        print("%d FAILED" % len(FAILS))
        sys.exit(1)
    print("all passed")
