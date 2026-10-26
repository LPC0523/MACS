"""The scanner's current shim setting, as the starting point for optimisation.

The active coils do not shim a raw field -- they shim whatever is left after
the scanner's own shim system has already acted. So the solve is not "find the
best shim", it is "find the best *change* from here", with the system terms
allowed to move only within what the scanner will accept.

Delta, not absolute
-------------------
The optimiser returns a delta and the UI shows `baseline + delta` alongside it.
Solving for an absolute setting instead would let the answer land far from
where prescan left things, and prescan is what the transmit and receive
frequency are locked to: a large jump moves the water peak off the locked
frequency (EPI readout shifts), invalidates the measured flip angle, and on
many consoles is simply rejected by the internal limit check. Bounding the
delta keeps all three from happening.

Which terms
-----------
Eight: three first-order and five second-order, exactly the variables in
`2nd_order_SH/calibrated_shim_simulation.ipynb`, whose calibration factors this
code shares. Zeroth order (A00, the centre frequency) is deliberately excluded
-- see `hybrid_shim.suggest_centre_frequency`.

Units
-----
uT/m for first order, uT/m² for second, matching the calibrated notebook, whose
measured deltas run to several hundred (A20 +195, B21 -856, A22 +288) against a
baseline in the hundreds to low thousands (B11 -1758, A20 +505). Bounds must be
on that scale. `config/shim_delta_bounds.json` is where site limits go.

The +-2000 envelope is a policy, not a placeholder
--------------------------------------------------
+-2000 on both the delta and the absolute setting is a deliberately
conservative deployment envelope chosen to sit inside what the scanner will
actually accept. It is expected to bind, and binding is the envelope doing its
job -- a solve that stops at +-2000 has been held back on purpose. Measured on
this cohort, an SH-only fit to the PFC alone asks for B21 of 3000 to 4400, so
the envelope is active on real data rather than nominally present.

Treat a pinned term as a constraint that was enforced, never as a sign of a
missing config or a unit error. The one case that genuinely is a unit problem
-- limits entered on the console's own single-digit scale -- looks different
and is detected separately by `bounds_bind_everything`: there *almost every*
term pins at once, whereas the policy envelope binds one or two.
"""

from __future__ import annotations

import datetime
import json
from dataclasses import dataclass, field
from pathlib import Path

from core.hybrid_shim import SH1, SH2, SH12, ORDER, UNIT

#: Display order and labels for the eight terms, matching the UI table.
TERMS = [
    ("A10", "Z0"),
    ("A11", "X"),
    ("B11", "Y"),
    ("A20", "Z²"),
    ("A21", "ZX"),
    ("B21", "ZY"),
    ("A22", "X²-Y²"),
    ("B22", "XY"),
]

LABEL = dict(TERMS)

#: Every spelling seen in exported shim files, lower-cased, mapped to our name.
_ALIASES = {
    "a10": "A10", "z": "A10", "z0": "A10", "z1": "A10",
    "a11": "A11", "x": "A11", "x1": "A11",
    "b11": "B11", "y": "B11", "y1": "B11",
    "a20": "A20", "z2": "A20", "z²": "A20", "zz": "A20",
    "a21": "A21", "zx": "A21", "xz": "A21",
    "b21": "B21", "zy": "B21", "yz": "B21",
    "a22": "A22", "x2y2": "A22", "x2-y2": "A22", "x²-y²": "A22", "x2_y2": "A22",
    "b22": "B22", "xy": "B22", "2xy": "B22",
}

#: Recognised but not an optimisation variable. Note `z0` is NOT here: in this
#: project's console labelling `Z0` is the first-order Z term (A10), which is
#: what `_ALIASES` maps it to. The zeroth order is only ever spelled A00/f0.
_CENTRE_FREQ_KEYS = {"a00", "f0", "cf", "centre_frequency", "center_frequency"}

#: The deployment envelope on the *change*. Not a fallback for a missing
#: setting -- this is the chosen conservative limit, and `config/` ships an
#: explicit file stating the same number so the value is visible rather than
#: implicit. Expected to bind on some subjects.
_BUILTIN_BOUND = 2000.0
DEFAULT_BOUNDS = {n: _BUILTIN_BOUND for n in SH12}

#: The envelope on the *absolute* setting, per term. Intersecting it with the
#: delta envelope is what makes the usable window asymmetric: a delta that is
#: individually small can still be unusable when the baseline is already near
#: the edge (B11 = -1757.7 leaves 242 of downward room against 3758 upward).
#: Overridable per term in the same config file.
DEFAULT_ABS_LIMIT = 2000.0
DEFAULT_ABS_LIMITS = {n: DEFAULT_ABS_LIMIT for n in SH12}

BOUNDS_FILENAME = "shim_delta_bounds.json"
STATE_FILENAME = "last_shim_state.json"


def canonical(name: str) -> str | None:
    """Our term name for whatever the file called it, or None if unrecognised."""
    key = str(name).strip().lower().replace(" ", "")
    return _ALIASES.get(key)


def is_centre_frequency(name: str) -> bool:
    return str(name).strip().lower().replace(" ", "") in _CENTRE_FREQ_KEYS


def _config_dirs() -> list:
    """Where shim_bounds.json may live: the user's copy wins over the bundled one."""
    from core.coil_registry import _app_base, user_config_dir

    return [Path(user_config_dir()).parent, Path(_app_base()) / "config"]


def load_bounds() -> tuple:
    """Site limits from the config file. Returns (delta_bounds, abs_limits, source)."""
    for d in _config_dirs():
        p = Path(d) / BOUNDS_FILENAME
        if not p.is_file():
            continue
        try:
            raw = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        out = dict(DEFAULT_BOUNDS)
        for k, v in (raw.get("bounds", raw) or {}).items():
            n = canonical(k)
            if n is None:
                continue
            try:
                out[n] = abs(float(v))
            except (TypeError, ValueError):
                continue
        lim = dict(DEFAULT_ABS_LIMITS)
        for k, v in (raw.get("absolute_limits") or {}).items():
            n = canonical(k)
            if n is None:
                continue
            try:
                lim[n] = abs(float(v))
            except (TypeError, ValueError):
                continue
        return out, lim, str(p)
    return dict(DEFAULT_BOUNDS), dict(DEFAULT_ABS_LIMITS), "内置默认 (Δ±2000, 绝对值±2000)"


def write_default_bounds_file(path=None) -> Path:
    """Write a commented starting point for site-specific limits."""
    p = Path(path) if path else Path(_config_dirs()[0]) / BOUNDS_FILENAME
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({
        "_comment": (
            "每项允许偏离当前值的幅度（半宽），即差值约束 |Δθ| ≤ Δmax。"
            "单位与 Step 2 一致：一阶 uT/m，二阶 uT/m²。填 0 表示锁定该项。"),
        "_policy": (
            "±2000 是主动选择的保守部署边界，位于 scanner 真实能力范围之内，"
            "不是占位值、不是缺省兜底。它**预期会 binding**：本队列实测中，"
            "仅用 SH 拟合 PFC 需要 B21 达 3000–4400，因此该边界在真实数据上是"
            "起作用的。某一项顶到边界＝安全策略生效，不是单位错误或配置缺失。"),
        "_scale_warning": (
            "唯一真正属于单位错误的情形是：这里填了主机 console 刻度（个位数），"
            "而上面的标定单位量级是几百"
            "（calibrated_shim_simulation.ipynb 实测 Z0≈44, Z²≈195, ZY≈-856, "
            "X²-Y²≈288）。那种情况下八项会**同时**全部卡死，与策略性 binding "
            "（一两项顶边界）形态不同，由 bounds_bind_everything 单独识别。"),
        "_absolute_note": (
            "absolute_limits 是主机自身的限幅，约束的是「当前值 + Δ」而不是 Δ。"
            "两个约束取交集：基线非零时，可用的 Δ 自动变窄且**逐项不对称**——"
            "见 SystemShimState.delta_bounds()。"),
        "bounds": {LABEL[n]: DEFAULT_BOUNDS[n] for n, _ in TERMS},
        "absolute_limits": {LABEL[n]: DEFAULT_ABS_LIMITS[n] for n, _ in TERMS},
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    return p


#: Fraction of unlocked terms that must be pinned before we call it a unit
#: problem. Not `all`: one term genuinely landing inside its limit is common
#: even when the limits are far too tight, and requiring unanimity means the
#: warning never fires in exactly the case it exists for.
_BIND_FRACTION = 0.7


def bounds_bind_everything(deltas, bounds, terms) -> bool:
    """True when nearly every unlocked term ended up pinned to its limit.

    Not a test for "did the envelope bind". The +-2000 envelope is meant to
    bind and routinely does on one or two terms; that is the safety policy
    working. This looks for the different signature of limits entered in the
    console's own scale -- single digits where the calibrated units run to
    hundreds -- where the solve returns the largest change it is allowed on
    *every* term at once and still reports success.
    """
    unlocked = [(n, d) for n, d in zip(terms, deltas)
                if abs(float(bounds.get(n, 0.0))) > 0]
    if len(unlocked) < 3:
        return False        # too few to distinguish from a genuine solution
    pinned = sum(abs(abs(float(d)) - abs(float(bounds[n]))) < 1e-6
                 for n, d in unlocked)
    return pinned >= _BIND_FRACTION * len(unlocked)


@dataclass
class SystemShimState:
    """Baseline shim values and how far each may move."""

    #: Term -> the scanner's current setting, in the units above.
    baseline: dict = field(default_factory=lambda: {n: 0.0 for n in SH12})
    #: Term -> allowed movement either side of baseline (a half-width, >= 0).
    bounds: dict = field(default_factory=lambda: dict(DEFAULT_BOUNDS))
    #: Term -> the console's limit on the absolute setting, |baseline + delta|.
    abs_limits: dict = field(default_factory=lambda: dict(DEFAULT_ABS_LIMITS))

    source: str = "manual"          # manual | scanner | file
    scanner: str = ""
    exported_at: str = ""
    bounds_source: str = ""
    #: The centre frequency, if the file carried one. Recorded, never optimised.
    centre_frequency: float | None = None
    #: Anything that had to be interpreted while reading a file.
    notes: list = field(default_factory=list)

    # ── construction ─────────────────────────────────────────────────
    @classmethod
    def default(cls) -> "SystemShimState":
        bounds, lim, src = load_bounds()
        return cls(bounds=bounds, abs_limits=lim, bounds_source=src)

    @classmethod
    def from_dict(cls, raw: dict) -> "SystemShimState":
        bounds, lim, bsrc = load_bounds()
        st = cls(baseline={n: 0.0 for n in SH12}, bounds=bounds,
                 abs_limits=lim, bounds_source=bsrc)

        # Both shapes are accepted: a flat {"Z": 2.1, ...} as exported by hand,
        # and a wrapped {"system_shim_state": {...}, "bounds": {...}}.
        block = raw.get("system_shim_state", raw)
        unknown, matched = [], 0
        for key, value in (block or {}).items():
            if str(key).startswith("_") or key in ("bounds", "absolute_limits",
                                                   "units", "source", "scanner",
                                                   "exported_at"):
                continue
            if is_centre_frequency(key):
                try:
                    st.centre_frequency = float(value)
                except (TypeError, ValueError):
                    pass
                st.notes.append(
                    f"「{key}」是中心频率（零阶），已记录但不参与优化——"
                    "改动它会让 prescan 锁定的发射/接收频率失效。")
                continue
            name = canonical(key)
            if name is None:
                unknown.append(str(key))
                continue
            try:
                st.baseline[name] = float(value)
                matched += 1
            except (TypeError, ValueError):
                unknown.append(f"{key}={value!r}")

        for key, value in (raw.get("absolute_limits") or {}).items():
            name = canonical(key)
            if name is None:
                continue
            try:
                st.abs_limits[name] = abs(float(value))
            except (TypeError, ValueError):
                pass

        for key, value in (raw.get("bounds") or {}).items():
            name = canonical(key)
            if name is None:
                if not is_centre_frequency(key):
                    unknown.append(f"bounds.{key}")
                continue
            try:
                st.bounds[name] = abs(float(value))
            except (TypeError, ValueError):
                unknown.append(f"bounds.{key}={value!r}")
        if raw.get("bounds"):
            st.bounds_source = "文件内 bounds 覆盖"

        if unknown:
            st.notes.append("以下字段无法识别，已忽略：" + ", ".join(unknown[:10]))
        if matched == 0:
            st.notes.append(
                "没有识别到任何匀场项，八个系数都按 0 处理。可用字段名："
                "A10/A11/B11/A20/A21/B21/A22/B22，或 Z/X/Y/Z2/ZX/ZY/X2Y2/XY。")

        st.source = str(raw.get("source", "file"))
        st.scanner = str(raw.get("scanner", ""))
        st.exported_at = str(raw.get("exported_at", ""))
        return st

    @classmethod
    def load(cls, path) -> "SystemShimState":
        p = Path(path)
        try:
            raw = json.loads(p.read_text(encoding="utf-8"))
        except FileNotFoundError:
            raise ValueError(f"找不到文件：{p}")
        except json.JSONDecodeError as e:
            raise ValueError(
                f"这个文件不是合法的 JSON：{p.name}\n第 {e.lineno} 行：{e.msg}")
        if not isinstance(raw, dict):
            raise ValueError(f"JSON 顶层应当是一个对象，实际是 {type(raw).__name__}。")
        return cls.from_dict(raw)

    # ── serialisation ────────────────────────────────────────────────
    def to_dict(self) -> dict:
        out = {
            "system_shim_state": {n: round(float(self.baseline.get(n, 0.0)), 6)
                                  for n in SH12},
            "bounds": {n: round(float(self.bounds.get(n, DEFAULT_BOUNDS[n])), 6)
                       for n in SH12},
            "absolute_limits": {n: round(float(self.abs_limits.get(
                n, DEFAULT_ABS_LIMIT)), 6) for n in SH12},
            "units": {"first_order": UNIT[1], "second_order": UNIT[2]},
            "source": self.source,
            "scanner": self.scanner,
            "exported_at": (self.exported_at or
                            datetime.datetime.now().replace(microsecond=0).isoformat()),
        }
        if self.centre_frequency is not None:
            out["centre_frequency_hz"] = float(self.centre_frequency)
        return out

    def save(self, path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False),
                     encoding="utf-8")
        return p

    # ── use ──────────────────────────────────────────────────────────
    def delta_bounds(self, terms=None) -> list:
        """(low, high) on the *change* for each term, in `terms` order.

        Two envelopes, intersected. This intersection is the formal constraint
        every solver in the project is given -- there is no separate, looser
        one anywhere:

          |delta|            <= bounds[n]        keeps prescan roughly valid
          |baseline + delta| <= abs_limits[n]    stays inside the deployment
                                                 envelope for the absolute
                                                 setting

        The second is what makes the window asymmetric. With B11 sitting at
        -1757.7 against a +-2000 limit there is 242 of room downward and 3758
        upward, so a symmetric +-Δ would happily return a value outside the
        envelope. Applying only the delta bound is the bug this fixes.

        Both envelopes are expected to bind. A returned window narrower than
        `bounds[n]`, or one that is not centred on zero, is the policy being
        enforced against a non-zero baseline, not a defect.
        """
        terms = list(terms or SH12)
        out = []
        for n in terms:
            d = abs(float(self.bounds.get(n, DEFAULT_BOUNDS.get(n, 0.0))))
            lim = abs(float(self.abs_limits.get(n, DEFAULT_ABS_LIMIT)))
            base = float(self.baseline.get(n, 0.0))
            lo = max(-d, -lim - base)
            hi = min(+d, +lim - base)
            if lo > hi:
                # The baseline is already outside the console limit. Freeze the
                # term rather than emit a delta that cannot help.
                lo = hi = 0.0
            out.append((lo, hi))
        return out

    def headroom(self, terms=None) -> dict:
        """Per term: (down, up) room the console limit leaves, before Δ capping."""
        terms = list(terms or SH12)
        out = {}
        for n in terms:
            lim = abs(float(self.abs_limits.get(n, DEFAULT_ABS_LIMIT)))
            base = float(self.baseline.get(n, 0.0))
            out[n] = (-lim - base, lim - base)
        return out

    def check_absolute(self, delta, terms=None) -> list:
        """Terms whose resulting absolute value would exceed the console limit.

        Run against the *scanner* delta, which is the model delta minus what
        the coils already contribute -- that difference is not known until
        after the solve, so the box constraint alone cannot guarantee it.
        """
        terms = list(terms or SH12)
        if not isinstance(delta, dict):
            delta = {n: float(v) for n, v in zip(terms, delta)}
        bad = []
        for n in terms:
            lim = abs(float(self.abs_limits.get(n, DEFAULT_ABS_LIMIT)))
            val = float(self.baseline.get(n, 0.0)) + float(delta.get(n, 0.0))
            if abs(val) > lim + 1e-6:
                bad.append((n, val, lim))
        return bad

    def absolute(self, delta, terms=None) -> dict:
        """baseline + delta -- what would actually be set on the scanner."""
        terms = list(terms or SH12)
        if not isinstance(delta, dict):
            delta = {n: float(v) for n, v in zip(terms, delta)}
        return {n: float(self.baseline.get(n, 0.0)) + float(delta.get(n, 0.0))
                for n in terms}

    def is_frozen(self, term: str) -> bool:
        """A zero allowance means the term is held at its baseline."""
        return abs(float(self.bounds.get(term, 0.0))) <= 0.0

    def summary(self) -> str:
        return "  ".join(
            f"{LABEL[n]}={self.baseline.get(n, 0.0):g}±{self.bounds.get(n, 0.0):g}"
            for n, _ in TERMS)
