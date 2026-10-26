"""Thin Python wrapper over the C++ HD-BET/ONNX brain extractor (`_cpp_bet`).

Drop-in for `B0_cal_algorithm._bet_robust`: same call signature, same output
file naming (`<stem>_mask.nii.gz`), returns the mask path.

Loading the ONNX graph costs a second or two, so one extractor is cached per
(model, threads, mode) and reused for the whole session.

Environment overrides:
    CPP_BET_MODEL    explicit path to the .onnx file
    CPP_BET_MODE     'quality' (default) or 'fast'
    CPP_BET_THREADS  intra-op thread count (default: 4)
"""
from __future__ import annotations

import os
import sys
import threading
from pathlib import Path

__all__ = ["CppBETBridge", "bet", "available", "why_unavailable"]

_LOCK = threading.Lock()
_CACHE: dict = {}
_IMPORT_ERROR: str | None = None


# ──────────────────────────────────────────────────────────────────────
#  Locating the compiled module and the model
# ──────────────────────────────────────────────────────────────────────
def _app_base() -> Path:
    """Root that bundled data sits under (PyInstaller _MEIPASS when frozen)."""
    if getattr(sys, "frozen", False):
        return Path(getattr(sys, "_MEIPASS", Path(sys.executable).parent))
    return Path(__file__).resolve().parent.parent        # .../shimming_qt


def _repo_roots() -> list[Path]:
    """Walk up looking for the checkout that holds cpp_bet/.

    The two apps sit at different depths (shimming_qt/core vs
    shimming_app/shimming_app/core), so a fixed number of `.parent` hops
    silently resolves to the wrong place in one of them.
    """
    out = []
    for p in Path(__file__).resolve().parents:
        if (p / "cpp_bet").is_dir():
            out.append(p)
    return out


def _candidate_dirs() -> list[Path]:
    """Places the built _cpp_bet module may live, most specific first."""
    here = Path(__file__).resolve().parent               # .../core
    base = _app_base()
    out: list[Path] = []
    env = os.environ.get("CPP_BET_DIR")
    if env:
        out.append(Path(env))
    out += [base, base / "cpp_bet", here, here / "_cpp_bet"]
    for repo in _repo_roots():
        out += [repo / "cpp_bet" / "build" / "Release",
                repo / "cpp_bet" / "build",
                repo / "cpp_bet"]
    seen, uniq = set(), []
    for p in out:
        k = str(p).lower()
        if k not in seen and p.is_dir():
            seen.add(k)
            uniq.append(p)
    return uniq


def _import_module():
    """Import _cpp_bet, making sure the ONNX Runtime DLLs are findable."""
    global _IMPORT_ERROR
    try:
        import _cpp_bet  # already on sys.path
        return _cpp_bet
    except ImportError:
        pass

    tried = []
    for d in _candidate_dirs():
        pyd = list(d.glob("_cpp_bet*.pyd")) + list(d.glob("_cpp_bet*.so"))
        if not pyd:
            continue
        tried.append(str(d))
        # onnxruntime.dll sits beside the .pyd (CMake copies it there).
        if hasattr(os, "add_dll_directory"):
            try:
                os.add_dll_directory(str(d))
            except OSError:
                pass
        if str(d) not in sys.path:
            sys.path.insert(0, str(d))
        try:
            import _cpp_bet
            return _cpp_bet
        except ImportError as e:
            _IMPORT_ERROR = f"{d}: {e}"

    if not tried:
        _IMPORT_ERROR = ("_cpp_bet not built. Build it with:\n"
                         "  cmake -S cpp_bet -B cpp_bet/build -A x64 "
                         "-DCMAKE_TOOLCHAIN_FILE=<vcpkg>/scripts/buildsystems/vcpkg.cmake\n"
                         "  cmake --build cpp_bet/build --config Release")
    raise ImportError(_IMPORT_ERROR or "_cpp_bet import failed")


def _find_model() -> Path:
    env = os.environ.get("CPP_BET_MODEL")
    if env:
        p = Path(env)
        if p.is_file():
            return p
        raise FileNotFoundError(f"CPP_BET_MODEL points at a missing file: {p}")

    here = Path(__file__).resolve().parent
    base = _app_base()

    searched = [base / "model", here / "model", here]
    searched += [r / "cpp_bet" / "model" for r in _repo_roots()]

    # Preference is explicit and matches what ships. The development tree holds
    # both the student and the 117 MB teacher; picking "whatever is first"
    # made source runs use the teacher (238 s per volume) while the packaged
    # app used the student (1.2 s) -- same code, wildly different behaviour.
    PREFERRED = ("bet_micro.onnx", "bet_small.onnx", "bet_medium.onnx")
    for d in searched:
        if not d.is_dir():
            continue
        for name in PREFERRED:
            if (d / name).is_file():
                return d / name
        students = sorted(p for p in d.glob("bet_*.onnx") if "int8" not in p.name)
        if students:
            return students[0]
        teacher = d / "hdbet_light.onnx"
        if teacher.is_file():
            return teacher
        others = sorted(d.glob("*.onnx"))
        if others:
            return others[0]

    raise FileNotFoundError(
        "No .onnx model found. Looked in: "
        + ", ".join(str(d) for d in searched)
        + ". Generate one with `python cpp_bet/export_onnx.py`, or set "
          "CPP_BET_MODEL.")


# ──────────────────────────────────────────────────────────────────────
class CppBETBridge:
    """Wraps one initialised `_cpp_bet.BrainExtractor`."""

    def __init__(self, model_path: str | os.PathLike | None = None,
                 num_threads: int | None = None, mode: str | None = None):
        mod = _import_module()
        self.model_path = str(model_path or _find_model())
        self.num_threads = int(num_threads or os.environ.get("CPP_BET_THREADS", 4))
        self.mode = (mode or os.environ.get("CPP_BET_MODE", "quality")).lower()

        self._ex = mod.BrainExtractor()
        if not self._ex.initialize(self.model_path, self.num_threads):
            raise RuntimeError(f"BrainExtractor.initialize failed: "
                               f"{self._ex.last_error()}")
        self._ex.set_mode(self.mode)

    @property
    def stats(self):
        return self._ex.last_stats

    def bet(self, input_path: str, output_path: str, f: float = 0.5,
            m: bool = False, log_cb=None) -> str:
        """Extract the brain. Returns the mask path.

        `f` is accepted for signature compatibility. It is *not* an intensity
        threshold here -- the network decides -- so it is deliberately ignored
        rather than silently reinterpreted.
        """
        log = log_cb or (lambda *_: None)

        stem = (output_path[:-len(".nii.gz")]
                if output_path.endswith(".nii.gz") else output_path)
        mask_path = stem + "_mask.nii.gz"

        rc = self._ex.extract(str(input_path), mask_path,
                              True, str(output_path))
        if rc != 0:
            raise RuntimeError(f"cpp_bet extract failed (code {rc}): "
                               f"{self._ex.last_error()}")

        s = self._ex.last_stats
        # Name the model: dev trees can hold several, and a 200x runtime
        # difference between them should never be a mystery.
        log(f"  [bet] {os.path.basename(self.model_path)} / {self.mode}: "
            f"{s.brain_voxels} voxels, {s.patches} patch(es), "
            f"{s.seconds_total:.1f}s ({s.seconds_inference:.1f}s inference)")
        return mask_path


# ──────────────────────────────────────────────────────────────────────
def _get(mode: str | None = None) -> CppBETBridge:
    key = (os.environ.get("CPP_BET_MODEL", ""),
           os.environ.get("CPP_BET_THREADS", "4"),
           (mode or os.environ.get("CPP_BET_MODE", "quality")).lower())
    with _LOCK:
        if key not in _CACHE:
            _CACHE[key] = CppBETBridge(mode=key[2])
        return _CACHE[key]


def available() -> bool:
    """True if the module and model are both present and loadable."""
    try:
        _get()
        return True
    except Exception:
        return False


def why_unavailable() -> str:
    try:
        _get()
        return ""
    except Exception as e:
        return f"{type(e).__name__}: {e}"


def diagnose() -> str:
    """Human-readable deployment check: what is present, what is missing.

    Run this after building or packaging instead of waiting for a pipeline to
    fail at BET time:
        python -c "from core.cpp_bet_bridge import diagnose; print(diagnose())"
    """
    import platform

    lines = [f"python      : {sys.version.split()[0]} ({platform.machine()})"]

    # 1. the compiled module
    found = None
    for d in _candidate_dirs():
        hits = list(d.glob("_cpp_bet*.pyd")) + list(d.glob("_cpp_bet*.so"))
        if hits:
            found = hits[0]
            break
    if found:
        lines.append(f"_cpp_bet    : {found}")
        tag = found.name.split(".")[1] if found.name.count(".") >= 2 else ""
        want = f"cp{sys.version_info.major}{sys.version_info.minor}"
        if tag and want not in tag:
            lines.append(f"  !! built for {tag}, this interpreter is {want} "
                         "-- rebuild the module")
    else:
        lines.append("_cpp_bet    : NOT FOUND (build cpp_bet, see cpp_bet/README.md)")
        lines.append(f"  searched  : {[str(d) for d in _candidate_dirs()]}")

    # 2. the ONNX Runtime DLL next to it
    if found:
        dll = found.parent / "onnxruntime.dll"
        lines.append(f"onnxruntime : {'OK ' + str(dll) if dll.is_file() else 'MISSING next to _cpp_bet'}")

    # 3. the model
    try:
        model = _find_model()
        lines.append(f"model       : {model} ({model.stat().st_size/1e6:.0f} MB)")
    except Exception as e:
        lines.append(f"model       : {e}")

    # 4. can it actually load?
    err = why_unavailable()
    lines.append(f"status      : {'READY' if not err else 'UNAVAILABLE -- ' + err}")
    if not err:
        b = _get()
        lines.append(f"  mode      : {b.mode}   threads: {b.num_threads}")
    return "\n".join(lines)


def bet(input_path: str, output_path: str, f: float = 0.5, m: bool = False,
        log_cb=None, mode: str | None = None) -> str:
    """Module-level convenience wrapper. Returns the mask path."""
    return _get(mode).bet(input_path, output_path, f=f, m=m, log_cb=log_cb)
