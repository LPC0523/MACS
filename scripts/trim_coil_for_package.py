"""Copy just the field maps a coil needs into the bundle, and write its config.

A raw coil calibration folder is ~600 MB, most of it the `positive_nii` /
`negative_nii` acquisitions that only mattered while deriving the field maps.
Inference reads one file per channel, so the bundle carries ~100 MB instead and
the app ships self-contained -- no "please locate your coil data" step, which
would just trade one path for another.

    python scripts/trim_coil_for_package.py \
        --src ../PFC_coil_profile_nii --name pfc_12ch --label "PFC 12ch"

    python scripts/trim_coil_for_package.py --all      # both known coils
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from pathlib import Path

APP = Path(__file__).resolve().parent.parent          # shimming_qt/
COILS_OUT = APP / "coils"
CONFIG_OUT = APP / "config" / "coils"

# Only the coil's own field map is kept. `*_resampled_to_brain.nii.gz` is that
# map pre-resampled onto one subject's grid; subjects here come in 14 different
# matrix sizes, so shipping it would make the engine resample an already
# resampled volume. The engine resamples the native map to whatever grid the
# current subject has, once.
KEEP_PATTERNS = ("*deltaFieldmap.nii*",)

KNOWN = [
    ("pfc_12ch", "PFC 12ch", "Prefrontal surface coil array, 12 channels",
     APP.parent / "PFC_coil_profile_nii", True),
    ("sf_12ch", "SF 12ch", "Surface coil array 2025-12-04 calibration, 12 channels",
     APP.parent / "SF_coil_profile_20251204_nii", False),
]


def human(n: int) -> str:
    return f"{n / 1e6:.1f} MB"


def trim(src: Path, slug: str, label: str, description: str,
         is_default: bool, dry_run: bool = False) -> bool:
    if not src.is_dir():
        print(f"  [SKIP] {label}: source not found -- {src}")
        return False

    ch_dirs = sorted(
        (p for p in src.iterdir()
         if p.is_dir() and re.fullmatch(r"ch\d+(_nii)?", p.name, re.I)),
        key=lambda p: int(re.sub(r"\D", "", p.name) or 0))
    if not ch_dirs:
        print(f"  [SKIP] {label}: no ch*_nii folders under {src}")
        return False

    dst_root = COILS_OUT / slug
    copied = kept_bytes = 0
    missing: list[str] = []

    for ch in ch_dirs:
        files = []
        for pat in KEEP_PATTERNS:
            files += sorted(ch.glob(pat))
        if not files:
            missing.append(ch.name)
            continue
        if not dry_run:
            (dst_root / ch.name).mkdir(parents=True, exist_ok=True)
        for f in files:
            kept_bytes += f.stat().st_size
            copied += 1
            if not dry_run:
                shutil.copy2(f, dst_root / ch.name / f.name)

    src_bytes = sum(f.stat().st_size for f in src.rglob("*") if f.is_file())
    print(f"  {label}: {len(ch_dirs)} channels, {copied} file(s) kept  "
          f"{human(src_bytes)} -> {human(kept_bytes)}"
          f"  ({100 * kept_bytes / max(src_bytes, 1):.0f}%)")
    if missing:
        # A channel with no usable map would silently shrink the solve.
        print(f"    !! no field map in: {', '.join(missing)}")
        return False

    cfg = {
        "name": label,
        "description": description,
        "channel_count": len(ch_dirs),
        # Relative to the application root, so it resolves both in a checkout
        # and inside the PyInstaller bundle where config/ moves to _internal/.
        "fieldmap_dir": f"coils/{slug}",
        "current_limit_a": 3.0,
        "default": is_default,
    }
    if not dry_run:
        CONFIG_OUT.mkdir(parents=True, exist_ok=True)
        (CONFIG_OUT / f"{slug}.json").write_text(
            json.dumps(cfg, indent=2), encoding="utf-8")
    print(f"    config -> config/coils/{slug}.json  (fieldmap_dir=coils/{slug})")
    return True


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src")
    ap.add_argument("--name", help="slug, e.g. pfc_12ch")
    ap.add_argument("--label", help="display name, e.g. 'PFC 12ch'")
    ap.add_argument("--description", default="")
    ap.add_argument("--default", action="store_true")
    ap.add_argument("--all", action="store_true", help="process the known coils")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if args.all:
        print(f"trimming known coils into {COILS_OUT}")
        ok = all(trim(src, slug, label, desc, dflt, args.dry_run)
                 for slug, label, desc, src, dflt in KNOWN)
    elif args.src and args.name:
        ok = trim(Path(args.src), args.name, args.label or args.name,
                  args.description, args.default, args.dry_run)
    else:
        ap.error("pass --all, or --src with --name")
        return 2

    if not ok:
        print("\none or more coils were not fully prepared -- see above")
        return 1

    total = sum(f.stat().st_size for f in COILS_OUT.rglob("*") if f.is_file()) \
        if COILS_OUT.is_dir() else 0
    print(f"\nbundle payload: {human(total)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
