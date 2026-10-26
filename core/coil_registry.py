"""Coil configurations, discovered from JSON instead of typed as a path.

The user should pick a coil from a list, not remember where its field maps
live. Each coil is one JSON file under `config/coils/`:

    {
      "name": "PFC 12ch",
      "description": "Prefrontal surface coil array",
      "channel_count": 12,
      "fieldmap_dir": "../../PFC_coil_profile_nii",   # relative to the JSON
      "current_limit_a": 3.0,
      "default": true
    }

`fieldmap_dir` holds one `ch<N>_nii/` sub-directory per channel, each with a
`*_resampled_to_brain.nii.gz` or `*deltaFieldmap.nii*`.

Configs are validated on load: a coil whose directory is missing, or whose
channel count does not match what is on disk, is reported rather than silently
offered in the UI and failing at optimisation time.
"""
from __future__ import annotations

import json
import os
import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

__all__ = ["Coil", "list_coils", "get_coil", "get_default", "config_dir",
           "user_config_dir", "discover_report", "inspect_folder",
           "add_coil", "remove_coil", "is_removable",
           "Issue", "CoilReport", "validate_coil", "validate_all",
           "fieldmap_hash", "find_fieldmap_nii"]


@dataclass
class Coil:
    name: str
    description: str = ""
    channel_count: int = 0
    fieldmap_dir: Path = field(default_factory=Path)
    current_limit_a: float = 3.0
    is_default: bool = False
    imported: bool = False
    source_json: Path = field(default_factory=Path)

    #: Optional content hash recorded when the coil was calibrated. When set,
    #: validation recomputes it and flags a mismatch -- field maps silently
    #: swapped for another revision produce currents that look plausible and
    #: are wrong, which nothing downstream can detect.
    fieldmap_version: str = ""

    #: How the field-map unit was established, and by whom. A built-in coil
    #: carries the confirmation in its config file, so the assertion travels
    #: with the calibration it was made about. A coil the user imports starts
    #: `unconfirmed` and stays that way until they say otherwise -- confirming
    #: the shipped 12ch maps must not silently vouch for a folder someone
    #: drops in next month.
    field_unit: str = "Hz/A"
    field_unit_status: str = "unconfirmed"
    field_unit_confirmed_date: str = ""
    field_unit_note: str = ""

    #: Empty when the coil is usable; otherwise why it is not.
    problem: str = ""

    @property
    def usable(self) -> bool:
        return not self.problem

    def channel_dirs(self) -> list[Path]:
        """ch1_nii, ch2_nii, ... in natural order."""
        if not self.fieldmap_dir.is_dir():
            return []
        dirs = [p for p in self.fieldmap_dir.iterdir()
                if p.is_dir() and re.fullmatch(r"ch\d+(_nii)?", p.name, re.I)]
        return sorted(dirs, key=lambda p: int(re.sub(r"\D", "", p.name) or 0))

    def __str__(self) -> str:
        tag = "" if self.usable else "  [unavailable]"
        return f"{self.name} ({self.channel_count}ch){tag}"


def _app_base() -> Path:
    """Root that bundled data sits under (PyInstaller _MEIPASS when frozen)."""
    import sys
    if getattr(sys, "frozen", False):
        return Path(getattr(sys, "_MEIPASS", Path(sys.executable).parent))
    return Path(__file__).resolve().parent.parent


def config_dir() -> Path:
    """Where the coils shipped with the app live (read-only in a bundle)."""
    return _app_base() / "config" / "coils"


def user_config_dir() -> Path:
    """Where coils the user imports live.

    Deliberately outside the installation: a packaged build keeps config/ inside
    _internal/, which is replaced wholesale on reinstall or update, so anything
    written there would quietly vanish.
    """
    from core.paths import user_data_dir
    d = Path(user_data_dir()) / "coils"
    d.mkdir(parents=True, exist_ok=True)
    return d


def user_data_root() -> Path:
    """Where copied field maps for imported coils are stored."""
    from core.paths import user_data_dir
    d = Path(user_data_dir()) / "coil_data"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _slugify(name: str) -> str:
    s = re.sub(r"[^\w\-]+", "_", name.strip(), flags=re.U).strip("_").lower()
    return s or "coil"


def inspect_folder(folder: str | os.PathLike) -> dict:
    """Look at a candidate coil folder without importing it.

    Returns {ok, channels, usable, missing, suggested_name, problem}. The UI
    shows this before asking for a name, so the user is not made to fill in a
    form for a folder that was never going to work.
    """
    import nibabel as nib

    p = Path(folder)
    out = {"ok": False, "channels": 0, "usable": 0, "missing": [],
           "suggested_name": p.name, "problem": ""}
    if not p.is_dir():
        out["problem"] = f"不是文件夹：{p}"
        return out

    ch_dirs = sorted(
        (d for d in p.iterdir()
         if d.is_dir() and re.fullmatch(r"ch\d+(_nii)?", d.name, re.I)),
        key=lambda d: int(re.sub(r"\D", "", d.name) or 0))
    out["channels"] = len(ch_dirs)
    if not ch_dirs:
        out["problem"] = ("这个文件夹里没有 ch1_nii / ch2_nii … 这样的通道子目录。\n"
                          "请选择线圈标定输出的根目录。")
        return out

    for d in ch_dirs:
        cand = sorted(d.glob("*deltaFieldmap.nii*"))
        if not cand:
            out["missing"].append(f"{d.name}（没有 deltaFieldmap）")
            continue
        try:
            img = nib.load(str(cand[0]))
            arr = np.asanyarray(img.dataobj, dtype=np.float32)
            if not np.any(arr):
                out["missing"].append(f"{d.name}（场图全为 0）")
                continue
        except Exception as e:
            out["missing"].append(f"{d.name}（无法读取：{type(e).__name__}）")
            continue
        out["usable"] += 1

    if out["usable"] == 0:
        out["problem"] = ("每个通道目录里都没有可用的 *deltaFieldmap.nii.gz。\n"
                          "本软件用它作为线圈场图，原始采集数据（positive/negative）不能直接使用。")
    elif out["missing"]:
        # Import would silently optimise fewer channels than the coil has.
        out["problem"] = ("以下通道没有可用场图，导入后会缺失：\n  "
                          + "\n  ".join(out["missing"]))
    else:
        out["ok"] = True
    return out


def add_coil(folder: str | os.PathLike, name: str,
             current_limit_a: float = 3.0,
             make_default: bool = False,
             progress_cb=None) -> Coil:
    """Import a coil: copy its field maps and write a config.

    Only `*deltaFieldmap.nii.gz` is copied (a few MB per channel); the raw
    acquisitions that sit alongside are not needed for inference. Raises
    ValueError with a readable message if the folder is not usable.
    """
    log = progress_cb or (lambda *_: None)
    info = inspect_folder(folder)
    if info["usable"] == 0:
        raise ValueError(info["problem"] or "这个文件夹不能作为线圈使用。")

    name = (name or info["suggested_name"]).strip()
    if not name:
        raise ValueError("请为线圈起一个名字。")
    if any(c.name == name for c in _discover()):
        raise ValueError(f"已存在同名线圈「{name}」，请换一个名字。")

    slug = _slugify(name)
    cfg_path = user_config_dir() / f"{slug}.json"
    n = 1
    while cfg_path.exists():
        n += 1
        slug = f"{_slugify(name)}_{n}"
        cfg_path = user_config_dir() / f"{slug}.json"

    src = Path(folder)
    dst = user_data_root() / slug
    if dst.exists():
        shutil.rmtree(dst, ignore_errors=True)

    ch_dirs = sorted(
        (d for d in src.iterdir()
         if d.is_dir() and re.fullmatch(r"ch\d+(_nii)?", d.name, re.I)),
        key=lambda d: int(re.sub(r"\D", "", d.name) or 0))

    copied = 0
    for i, d in enumerate(ch_dirs, 1):
        cand = sorted(d.glob("*deltaFieldmap.nii*"))
        if not cand:
            continue
        (dst / d.name).mkdir(parents=True, exist_ok=True)
        shutil.copy2(cand[0], dst / d.name / cand[0].name)
        copied += 1
        log(f"  复制通道 {i}/{len(ch_dirs)}：{d.name}")

    cfg = {
        "name": name,
        "description": f"用户导入 · 来源 {src}",
        "channel_count": copied,
        "fieldmap_dir": str(dst),          # absolute: it lives in the user dir
        "current_limit_a": float(current_limit_a),
        "default": bool(make_default),
        "imported": True,
    }
    cfg_path.write_text(json.dumps(cfg, indent=2, ensure_ascii=False),
                        encoding="utf-8")
    log(f"  已保存配置 {cfg_path}")

    coil = _load_one(cfg_path)
    if not coil.usable:
        cfg_path.unlink(missing_ok=True)
        shutil.rmtree(dst, ignore_errors=True)
        raise ValueError(f"导入后校验失败：{coil.problem}")
    return coil


def is_removable(coil: Coil) -> bool:
    """Only user-imported coils can be deleted; the shipped ones stay."""
    try:
        return coil.source_json.parent.resolve() == user_config_dir().resolve()
    except OSError:
        return False


def remove_coil(name: str, delete_data: bool = True) -> bool:
    """Delete an imported coil. Returns False for built-in coils."""
    coil = get_coil(name)
    if coil is None or not is_removable(coil):
        return False
    coil.source_json.unlink(missing_ok=True)
    if delete_data:
        try:
            if coil.fieldmap_dir.resolve().is_relative_to(user_data_root().resolve()):
                shutil.rmtree(coil.fieldmap_dir, ignore_errors=True)
        except (OSError, AttributeError):
            pass
    return True


def _repo_roots() -> list[Path]:
    """Checkout directories above us, most specific first."""
    return [p for p in Path(__file__).resolve().parents
            if (p / "shimming_qt").is_dir() or (p / "cpp_bet").is_dir()]


def _resolve_fieldmap_dir(spec: str, json_path: Path) -> Path:
    """Locate a coil's field maps under both layouts.

    The field-map sets are ~600 MB each, so they are not bundled: the configs
    carry a path relative to the JSON, which is right in a checkout but points
    outside the bundle once PyInstaller moves the JSON into _internal/. Try the
    relative form first, then the same tail under the app root and each repo
    root, and finally an absolute path. The first that exists wins; a coil that
    resolves nowhere is reported rather than silently offered.
    """
    p = Path(spec)
    if p.is_absolute():
        return p

    candidates = [(json_path.parent / p)]
    tail = Path(*[part for part in p.parts if part not in ("..", ".")])
    if tail.parts:
        candidates.append(_app_base() / tail)
        candidates += [root / tail for root in _repo_roots()]
        candidates += [root.parent / tail for root in _repo_roots()[:2]]

    for c in candidates:
        try:
            r = c.resolve()
        except OSError:
            continue
        if r.is_dir():
            return r
    return (json_path.parent / p).resolve()


def _load_one(path: Path) -> Coil:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        return Coil(name=path.stem, source_json=path,
                    problem=f"cannot read config: {e}")

    fm = raw.get("fieldmap_dir", "")
    fm_path = _resolve_fieldmap_dir(fm, path)

    coil = Coil(
        name=raw.get("name", path.stem),
        description=raw.get("description", ""),
        channel_count=int(raw.get("channel_count", 0)),
        fieldmap_dir=fm_path,
        current_limit_a=float(raw.get("current_limit_a", 3.0)),
        is_default=bool(raw.get("default", False)),
        imported=bool(raw.get("imported", False)),
        source_json=path,
        fieldmap_version=str(raw.get("fieldmap_version", "") or ""),
        field_unit=str(raw.get("field_unit", "Hz/A") or "Hz/A"),
        # Absent means unconfirmed. An imported coil has no such entry unless
        # the user adds one, which is the point.
        field_unit_status=str(raw.get("field_unit_status", "unconfirmed")
                              or "unconfirmed"),
        field_unit_confirmed_date=str(
            raw.get("field_unit_confirmed_date", "") or ""),
        field_unit_note=str(raw.get("field_unit_note", "") or ""),
    )

    if not fm_path.is_dir():
        coil.problem = f"field-map folder not found: {fm_path}"
        return coil

    found = len(coil.channel_dirs())
    if found == 0:
        coil.problem = f"no ch*_nii sub-folders under {fm_path}"
    elif coil.channel_count and found != coil.channel_count:
        # Mismatch means the config and the data have drifted apart; using it
        # would silently optimise the wrong number of channels.
        coil.problem = (f"config says {coil.channel_count} channels but "
                        f"{found} found on disk")
    elif not coil.channel_count:
        coil.channel_count = found
    return coil


def _discover() -> list[Coil]:
    """Built-in coils first, then anything the user imported."""
    out = []
    for d in (config_dir(), user_config_dir()):
        if d.is_dir():
            out += [_load_one(p) for p in sorted(d.glob("*.json"))]
    return out


def list_coils(usable_only: bool = True) -> list[Coil]:
    coils = _discover()
    return [c for c in coils if c.usable] if usable_only else coils


def get_coil(name: str) -> Coil | None:
    for c in _discover():
        if c.name == name or c.source_json.stem == name:
            return c
    return None


def get_default() -> Coil | None:
    """The coil marked default, else the first usable one."""
    usable = list_coils()
    for c in usable:
        if c.is_default:
            return c
    return usable[0] if usable else None


def discover_report() -> str:
    """One line per coil, including why unusable ones were rejected."""
    coils = _discover()
    if not coils:
        return f"no coil configs found in {config_dir()}"
    out = [f"coil configs in {config_dir()}:"]
    for c in coils:
        mark = "OK  " if c.usable else "SKIP"
        line = f"  [{mark}] {c.name}  {c.channel_count}ch  {c.fieldmap_dir}"
        if c.is_default:
            line += "  (default)"
        if c.problem:
            line += f"\n           -> {c.problem}"
        out.append(line)
    return "\n".join(out)


# ─────────────────────────────────────────────────────────────────────────
#  Validation
# ─────────────────────────────────────────────────────────────────────────
# `_load_one` answers "can this coil be offered at all". Validation answers the
# harder question: is what is on disk still the thing that was calibrated. A
# coil that loads fine but whose field maps were replaced, truncated, or saved
# on a different grid produces currents that look entirely reasonable and are
# wrong -- there is no downstream check that would catch it.

#: Field maps are Hz/A on the calibration grid; a channel that is flat carries
#: no information and would just be a free variable in the optimisation.
_FLAT_CHANNEL_EPS = 1e-9


@dataclass
class Issue:
    level: str          # "error" -> unusable, "warning" -> usable but suspect
    message: str

    def __str__(self) -> str:
        return f"[{self.level}] {self.message}"


@dataclass
class CoilReport:
    coil_name: str
    issues: list = field(default_factory=list)
    channels_checked: int = 0
    shapes: list = field(default_factory=list)

    @property
    def level(self) -> str:
        if any(i.level == "error" for i in self.issues):
            return "error"
        return "warning" if self.issues else "ok"

    @property
    def ok(self) -> bool:
        return self.level == "ok"

    @property
    def usable(self) -> bool:
        """Warnings still let the coil be selected; errors do not."""
        return self.level != "error"

    @property
    def mark(self) -> str:
        return {"ok": "✓", "warning": "⚠", "error": "✗"}[self.level]

    def summary(self) -> str:
        if self.ok:
            return f"{self.mark} {self.coil_name}：{self.channels_checked} 通道，全部正常"
        head = f"{self.mark} {self.coil_name}"
        return head + "\n    " + "\n    ".join(i.message for i in self.issues)


def find_fieldmap_nii(ch_dir: Path) -> Path | None:
    """The field map inside one channel directory, preferring the raw Hz/A map.

    `*_resampled_to_brain` files are derived products cached by an older
    version; the deltaFieldmap is the calibration output and is what the
    optimiser resamples from now.
    """
    for pattern in ("*deltaFieldmap.nii*", "*_resampled_to_brain.nii*", "*.nii*"):
        hits = sorted(ch_dir.glob(pattern))
        if hits:
            return hits[0]
    return None


def fieldmap_hash(coil: Coil) -> str:
    """Content hash over every channel field map, in channel order.

    Hashes the raw file bytes rather than the array so it stays cheap and does
    not depend on nibabel's version. Order matters: swapping two channels must
    change the hash.
    """
    import hashlib

    h = hashlib.sha256()
    for d in coil.channel_dirs():
        nii = find_fieldmap_nii(d)
        h.update(d.name.encode("utf-8"))
        if nii is None:
            h.update(b"\0missing")
            continue
        with open(nii, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
    return h.hexdigest()


def validate_coil(coil: Coil, deep: bool = True) -> CoilReport:
    """Check a coil against what is actually on disk.

    `deep=False` skips reading NIfTI headers and hashing, for the startup pass
    over many coils; the selected coil is then re-checked deeply.
    """
    import nibabel as nib

    rep = CoilReport(coil_name=coil.name)

    # Whatever _load_one already rejected it for.
    if coil.problem:
        rep.issues.append(Issue("error", coil.problem))
        return rep

    if not coil.fieldmap_dir.is_dir():
        rep.issues.append(Issue("error", f"场图目录不存在：{coil.fieldmap_dir}"))
        return rep

    dirs = coil.channel_dirs()
    rep.channels_checked = len(dirs)

    # 1. Channel count agrees with the config.
    if not dirs:
        rep.issues.append(Issue(
            "error", f"场图目录下没有 ch1_nii / ch2_nii … 子目录：{coil.fieldmap_dir}"))
        return rep
    if coil.channel_count and len(dirs) != coil.channel_count:
        rep.issues.append(Issue(
            "error", f"通道数不匹配：配置 {coil.channel_count}ch，"
                     f"场图目录只有 {len(dirs)} 个通道"))

    # 3. Current limit. Checked before the expensive part so a typo in the JSON
    #    is reported even when the field maps are unreadable.
    if coil.current_limit_a <= 0:
        rep.issues.append(Issue(
            "error", f"电流限值必须为正：当前 {coil.current_limit_a} A"))
    elif coil.current_limit_a > 10:
        rep.issues.append(Issue(
            "warning", f"电流限值偏高：{coil.current_limit_a} A"
                       "（常见放大器上限 3~5 A，请确认不是笔误）"))

    if not deep:
        return rep

    # 2. Every channel has a readable field map, all on one grid.
    shapes: dict = {}
    for idx, d in enumerate(dirs, start=1):
        nii = find_fieldmap_nii(d)
        if nii is None:
            rep.issues.append(Issue("error", f"CH{idx}（{d.name}）场图文件缺失"))
            continue
        try:
            img = nib.load(str(nii))
            shape = tuple(int(x) for x in img.shape[:3])
        except Exception as e:
            rep.issues.append(Issue(
                "error", f"CH{idx}（{d.name}）场图无法读取：{type(e).__name__}: {e}"))
            continue
        shapes.setdefault(shape, []).append(idx)

        # A channel of all zeros contributes nothing but still consumes a
        # degree of freedom, so the optimiser will happily assign it current.
        try:
            arr = np.asanyarray(img.dataobj, dtype=np.float32)
            if float(np.nanmax(np.abs(arr))) < _FLAT_CHANNEL_EPS:
                rep.issues.append(Issue("error", f"CH{idx}（{d.name}）场图全为 0"))
        except Exception:
            pass        # header read fine; a data read failure is reported above

    rep.shapes = sorted(shapes)
    if len(shapes) > 1:
        detail = "；".join(f"{s} ← CH{','.join(map(str, chs))}"
                           for s, chs in sorted(shapes.items()))
        rep.issues.append(Issue("error", f"各通道场图尺寸不一致：{detail}"))

    # 4. Field-map revision, when the config pinned one.
    if coil.fieldmap_version:
        actual = fieldmap_hash(coil)
        if actual != coil.fieldmap_version:
            rep.issues.append(Issue(
                "error",
                f"场图版本不匹配：配置记录 {coil.fieldmap_version[:8]}，"
                f"实际 {actual[:8]}。场图文件被替换过，请更新线圈配置后再用。"))

    return rep


def validate_all(deep: bool = True) -> list:
    """Validate every discovered coil, usable or not."""
    return [validate_coil(c, deep=deep) for c in _discover()]


if __name__ == "__main__":
    print(discover_report())
    print()
    for r in validate_all():
        print(r.summary())
