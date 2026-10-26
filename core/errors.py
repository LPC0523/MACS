"""Two-layer failure reporting: what the user should do, and what went wrong.

Every message the pipeline can end on has two audiences. The operator standing
at the scanner needs one sentence naming the thing they can change -- a folder,
a checkbox, a number they typed. The engineer reading the log afterwards needs
paths, regexes and exception text.

Putting both in one string serves neither: the actionable sentence gets buried
in a stack trace, so the operator reads none of it, and abbreviating for the
operator throws away what diagnosis depends on. `Advice` keeps them separate,
and `classify()` turns an arbitrary exception from anywhere in the pipeline
into one -- so a failure raised four call levels down still surfaces as
something the user can act on rather than as its own repr.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass
class Advice:
    #: One sentence, imperative, naming what to change. Shown in the red card.
    headline: str
    #: Paths, patterns, exception text. Shown only when "详情" is expanded.
    detail: str = ""
    #: Stable identifier, for logs and the run report.
    code: str = "unknown"
    #: Where to read more, when there is somewhere.
    doc: str = ""

    def as_text(self) -> str:
        out = self.headline
        if self.detail:
            out += "\n\n详情：\n" + self.detail
        return out


def _fmt(exc: BaseException) -> str:
    return f"{type(exc).__name__}: {exc}"


# Ordered: the first rule whose `match` fires wins, so put specific patterns
# before general ones.
_RULES: list = []


def _rule(code: str, pattern: str, headline, detail=None, doc: str = ""):
    _RULES.append({
        "code": code,
        "re": re.compile(pattern, re.I | re.S),
        "headline": headline,
        "detail": detail,
        "doc": doc,
    })


# ── Brain extraction ──────────────────────────────────────────────────
_rule(
    "bet_cannot_open",
    r"cannot open|cannot write|cpp_bet extract failed",
    "脑提取失败。请确认 DICOM 文件夹不在压缩包、U 盘或只读位置。",
    lambda e: f"cpp_bet 无法打开文件。原始错误：{_fmt(e)}",
)

_rule(
    "bet_no_backend",
    r"没有可用的高精度后端|no bet backend",
    "脑提取模型未就绪。请联系管理员安装 HD-BET 或重新编译 ONNX 模型。",
    lambda e: str(e),
    doc="docs/BET_SETUP.md",
)

# ── DICOM / series selection ──────────────────────────────────────────
_rule(
    "no_b0_series",
    r"没有找到 B0 mapping 序列|no b0 mapping series",
    "未检测到场图序列。请确认拖入的是双回波 GRE 场图数据（幅值 + 相位），"
    "而非 T1/T2/定位像。",
    lambda e: str(e),
)

_rule(
    "no_phase_magnitude",
    r"could not identify phase/magnitude",
    "无法识别幅值/相位图像对。可能原因：序列被过滤器误伤、DICOM 缺少 ImageType 标签、"
    "或数据来自尚未适配的扫描仪。",
    lambda e: str(e),
)

_rule(
    "no_dicom",
    r"no readable dicom|not a dicom|没有可读的 DICOM",
    "这个文件夹里没有可读的 DICOM 文件。请拖入扫描导出的原始 DICOM 目录。",
    lambda e: _fmt(e),
)

# ── Environment ───────────────────────────────────────────────────────
_rule(
    "ants_missing",
    r"antspy|no module named ['\"]?ants",
    "配准组件 ANTsPy 未安装，前额叶 ROI 无法生成。可改用「全脑」模式继续。",
    lambda e: _fmt(e),
)

_rule(
    "ants_path",
    r"could not create imageio object",
    "配准读取图像失败。若数据路径含中文，请把它移到纯英文路径后重试。",
    lambda e: _fmt(e),
)

_rule(
    "out_of_memory",
    r"memoryerror|cannot allocate|bad_alloc",
    "内存不足。请关闭其他程序后重试，或改用体素更少的 ROI。",
    lambda e: _fmt(e),
)

_rule(
    "permission",
    r"permission denied|access is denied|errno 13",
    "没有写入权限。请把数据放到有写权限的位置（不要放在 Program Files 或只读盘）。",
    lambda e: _fmt(e),
)


def classify(exc: BaseException, *, context: str = "") -> Advice:
    """Turn any pipeline exception into something the user can act on."""
    text = f"{type(exc).__name__}: {exc}"

    for rule in _RULES:
        if rule["re"].search(text):
            detail = rule["detail"](exc) if callable(rule["detail"]) else (
                rule["detail"] or _fmt(exc))
            if context:
                detail = f"{context}\n{detail}"
            return Advice(headline=rule["headline"], detail=detail,
                          code=rule["code"], doc=rule["doc"])

    # Nothing matched. Say so honestly rather than inventing a cause -- a wrong
    # suggestion costs more than no suggestion.
    detail = _fmt(exc)
    if context:
        detail = f"{context}\n{detail}"
    return Advice(
        headline="处理过程中出现未预期的错误。日志最后几行通常指出失败位置。",
        detail=detail, code="unclassified")


# ── Warnings that are not failures ────────────────────────────────────

@dataclass
class Warning_:
    headline: str
    detail: str = ""
    code: str = ""


def check_b0_scale(std_hz: float, delta_te_s: float) -> Warning_ | None:
    """A field map whose spread is impossible usually means ΔTE's unit is wrong.

    Reported rather than corrected: ΔTE is a protocol fact we cannot read from
    the data, so guessing it would replace a visible error with an invisible one.
    """
    if std_hz > 300:
        return Warning_(
            headline=(f"脑内 B0 标准差异常偏高（{std_hz:.0f} Hz，典型 30~80 Hz）。"
                      f"ΔTE 可能填成了毫秒而非秒。当前 ΔTE = {delta_te_s * 1000:.3f} ms。"),
            detail=(f"std = {std_hz:.1f} Hz，ΔTE = {delta_te_s} s。\n"
                    f"若实际 ΔTE 为 {delta_te_s * 1000:.3f} ms，"
                    f"应填 {delta_te_s / 1000:.6f} s。"),
            code="dte_maybe_ms")
    if std_hz < 3:
        return Warning_(
            headline=(f"脑内 B0 标准差几乎为零（{std_hz:.1f} Hz），场图接近平坦。"
                      f"ΔTE 可能填得过大。当前 ΔTE = {delta_te_s * 1000:.3f} ms。"),
            detail=f"std = {std_hz:.2f} Hz，ΔTE = {delta_te_s} s。",
            code="dte_maybe_too_large")
    return None
