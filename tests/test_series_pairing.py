"""Series filtering and phase/magnitude pairing, against real series names.

    D:\\python.exe tests/test_series_pairing.py

Both cases here come from an actual failure. An infant CBCP export named its
protocol `B0fieldmap`, whose magnitude echoes the filter dropped because the
`0` sat directly in front of `field`; and the same export carries two complete
B0 acquisitions, `_ref` and `_shim`, which the pairing code could mix.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core.B0_cal_algorithm import _acquisition_key, _find_phase_magnitude
from core.dicom_to_nifti import _B0_SERIES_PATTERN

FAILS = []


def ok(cond, msg):
    print("  [%s] %s" % ("OK" if cond else "FAIL", msg))
    if not cond:
        FAILS.append(msg)


def test_filter():
    import re
    pat = re.compile(_B0_SERIES_PATTERN, re.I)
    keep = [
        "B0fieldmap_ref_Echo1", "B0fieldmap_ref_Echo2",
        "B0fieldmap_shim_Echo1", "B0fieldmap_shim_Echo2",
        "B0fieldmap_ref_B0Map_ph", "B0fieldmap_shim_B0Map_ph",
        "b0map_tra_shim_B0Map_ph", "b0map_tra_20240913135300_203_ph",
        "gre_field_mapping", "GRE Field Map", "B0Map2",
    ]
    drop = [
        "PALB0Map_M", "CalB0Map", "gre_scout",
        "t1w_A3.22_iso0.8mm_CBCP", "t2w_A3.83_iso0.8mm_CBCP_NDC",
        "restfMRI_mb8_PA_1.8mm_CBCP_Ref",
        "se_epi_field_AP_1.8mm_CBCP_Ref",
        "restfMRI_mb8_PA_1.8mm_CBCP_shim_SaveBySlc",
    ]
    print("1. series filter")
    for n in keep:
        ok(bool(pat.search(n)), "keep %s" % n)
    for n in drop:
        ok(not pat.search(n), "drop %s" % n)


def test_key():
    print("2. acquisition grouping")
    pairs = [
        ("B0fieldmap_ref_Echo2.nii.gz", "B0fieldmap_ref_B0Map_ph.nii.gz"),
        ("B0fieldmap_shim_Echo1.nii.gz", "B0fieldmap_shim_B0Map_ph.nii.gz"),
    ]
    for a, b in pairs:
        ok(_acquisition_key(a) == _acquisition_key(b),
           "%s and %s share a key (%s)" % (a, b, _acquisition_key(a)))
    ok(_acquisition_key("B0fieldmap_ref_Echo2.nii.gz")
       != _acquisition_key("B0fieldmap_shim_Echo2.nii.gz"),
       "ref and shim keys differ")


def test_pairing():
    print("3. pairing inside one acquisition")
    names = ["B0fieldmap_ref_Echo1.nii.gz", "B0fieldmap_ref_Echo2.nii.gz",
             "B0fieldmap_ref_B0Map_ph.nii.gz",
             "B0fieldmap_shim_Echo1.nii.gz", "B0fieldmap_shim_Echo2.nii.gz",
             "B0fieldmap_shim_B0Map_ph.nii.gz"]
    with tempfile.TemporaryDirectory() as d:
        for n in names:
            (Path(d) / n).write_bytes(b"")
        lines = []
        ph, mg = _find_phase_magnitude(d, lines.append)
        ok(ph is not None and mg is not None, "found a pair")
        ok(_acquisition_key(Path(ph).name) == _acquisition_key(Path(mg).name),
           "phase and magnitude come from the same acquisition (%s / %s)"
           % (Path(ph).name, Path(mg).name))
        ok("echo2" in Path(mg).name.lower(), "magnitude is the second echo")
        ok(any("2 组完整" in l for l in lines),
           "both acquisitions were reported to the user")

    print("4. single acquisition, older naming")
    with tempfile.TemporaryDirectory() as d:
        for n in ("b0map_tra_202_e2.nii.gz", "b0map_tra_203_ph.nii.gz"):
            (Path(d) / n).write_bytes(b"")
        lines = []
        ph, mg = _find_phase_magnitude(d, lines.append)
        ok(ph and "ph" in Path(ph).name, "phase = %s" % Path(ph).name)
        ok(mg and "e2" in Path(mg).name, "magnitude = %s" % Path(mg).name)
        ok(not any("组完整" in l for l in lines), "no ambiguity warning")


def test_repeated_protocol_does_not_collide():
    """One protocol run many times must not collapse onto one filename.

    Series are grouped by SeriesNumber but named by SeriesDescription, and a
    session that repeats a protocol gives every repeat the same description.
    Measured on a 5T export: 72 accepted B0 series, 18 distinct descriptions,
    one of them used 11 times. Before the fix the converter wrote each name
    once per repeat and kept whichever landed last -- and because the three
    components are written independently, the survivors need not come from the
    same acquisition, which yields a field map assembled from two different
    scans with nothing to show for it.

    The prefix has to group as well as disambiguate: `_acquisition_key` strips
    role suffixes off the *end*, so a prefix survives it and the three
    components of one acquisition still pair.
    """
    print("5. a protocol repeated in one folder")
    from core.dicom_to_nifti import _acquisition_tag

    class _DS:
        def __init__(self, sn):
            self.SeriesNumber = sn

    # uMR numbering: acquisition N is N01 (Echo1), N02 (Echo2), N03 (phase).
    tags = [_acquisition_tag(_DS(n)) for n in (901, 902, 903)]
    ok(len(set(tags)) == 1 and tags[0] == "a09",
       "one acquisition's three components share a tag (%s)" % tags)
    ok(_acquisition_tag(_DS(1401)) == "a14",
       "a later acquisition gets a different one (a14)")
    ok(_acquisition_tag(_DS(903)) != _acquisition_tag(_DS(1403)),
       "so the two phase maps no longer share a filename")

    names = ["a09_b0map_tra_f0cal_Echo1.nii.gz",
             "a09_b0map_tra_f0cal_Echo2_e2.nii.gz",
             "a09_b0map_tra_f0cal_B0Map_ph.nii.gz",
             "a14_b0map_tra_f0cal_Echo1.nii.gz",
             "a14_b0map_tra_f0cal_Echo2_e2.nii.gz",
             "a14_b0map_tra_f0cal_B0Map_ph.nii.gz"]
    keys = {}
    for n in names:
        keys.setdefault(_acquisition_key(n), []).append(n)
    ok(len(keys) == 2, "the six files fall into two acquisitions (%d)" % len(keys))
    ok(all(len(v) == 3 for v in keys.values()),
       "three components each, so phase still pairs with its own magnitude")

    with tempfile.TemporaryDirectory() as d:
        for n in names:
            (Path(d) / n).write_bytes(b"")
        lines = []
        ph, mg = _find_phase_magnitude(d, lines.append)
        ok(_acquisition_key(Path(ph).name) == _acquisition_key(Path(mg).name),
           "the chosen pair is from one acquisition (%s / %s)"
           % (Path(ph).name, Path(mg).name))
        ok(any("2 组完整" in l for l in lines),
           "and both acquisitions are reported rather than one silently used")

    # Numbering that does not fit the convention must not be forced into it:
    # a wrong grouping pairs across acquisitions, which is the failure the
    # whole mechanism exists to prevent.
    ok(_acquisition_tag(_DS(2)) != _acquisition_tag(_DS(3)),
       "small series numbers stay distinct rather than collapsing to a00")


if __name__ == "__main__":
    test_filter()
    test_key()
    test_pairing()
    test_repeated_protocol_does_not_collide()
    print()
    if FAILS:
        print("%d FAILED" % len(FAILS))
        sys.exit(1)
    print("all passed")
