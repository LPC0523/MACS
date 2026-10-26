"""Figures speak English, and Chinese that reaches one still has glyphs.

    D:\\python.exe tests/test_cjk_labels.py

Matplotlib ships DejaVu Sans, which has no CJK glyphs, so Chinese inside a
figure renders as a row of empty rectangles -- how the dynamic-shim current
table shipped in v1.5.3, where the axis labels read 「通道」/「层 (slice)」/
「电流 (A)」. Qt draws the panel around the figure with a system font and was
never affected, which is why only the plots looked broken.

Two things are checked, because the fix has two halves:

1. No widget hands a Chinese *literal* to matplotlib. The figures are
   English, like the linear-encoding page's already were.
2. Chinese that arrives as *data* -- a coil whose channel names are Chinese,
   say -- can still be drawn, because `core.theme` puts an installed CJK face
   in front of matplotlib's default font stack at import time.
"""

from __future__ import annotations

import ast
import re
import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")

from matplotlib.figure import Figure  # noqa: E402
from matplotlib.font_manager import findfont, FontProperties  # noqa: E402

from core.theme import MPL_CJK_FONT  # noqa: E402  (sets the rcParams on import)

FAILS = []

# The calls that put text on a figure. Anything else in the file may be
# Chinese -- comments, Qt labels, tooltips -- and Qt has its own font.
LABEL_CALLS = {"set_title", "set_xlabel", "set_ylabel", "set_label",
               "set_xticklabels", "set_yticklabels", "suptitle", "text",
               "annotate", "legend"}
CJK = re.compile(r"[\u3000-\u303f\u4e00-\u9fff\uff00-\uffef]")


def ok(cond, msg):
    print("  [%s] %s" % ("OK" if cond else "FAIL", msg))
    if not cond:
        FAILS.append(msg)


def literals_in_label_calls():
    """CJK string literals the widgets pass to matplotlib, with source lines.

    Parsed rather than grepped, so a title split over two source lines --
    which is how the current table writes its own -- is not missed.
    """
    hits = []
    for path in sorted((ROOT / "widgets").glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not (isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute)
                    and node.func.attr in LABEL_CALLS):
                continue
            for sub in ast.walk(node):
                if (isinstance(sub, ast.Constant) and isinstance(sub.value, str)
                        and CJK.search(sub.value)):
                    hits.append(f"{path.name}:{node.lineno}  "
                                f"{node.func.attr}({sub.value!r})")
    return hits


print("1. figures carry no Chinese literals")
hits = literals_in_label_calls()
ok(not hits, f"no CJK in matplotlib label calls ({len(hits)} found)")
for h in hits:
    print("     ", h)

print("2. Chinese arriving as data can still be drawn")
print(f"  font chosen by core.theme: {MPL_CJK_FONT or '(none found)'}")
ok(bool(MPL_CJK_FONT), "a CJK font is installed and selected")

# Rendering is the real check: a face can be selected and still lack a
# character. Missing glyphs come out as warnings, one per character.
fig = Figure(figsize=(6, 3))
ax = fig.add_subplot(111)
ax.set_xticks([0])
ax.set_xticklabels(["前额叶通道 1"])     # a channel name out of a coil file
ax.set_ylabel("Current (A)")
ax.set_ylim(-1, 1)                       # a negative tick, for unicode_minus
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    fig.canvas.draw()
missing = [str(w.message) for w in caught
           if "missing from font" in str(w.message).lower()]
ok(not missing, f"all glyphs drawn ({len(missing)} missing)")
for msg in missing[:10]:
    print("     ", msg)

if MPL_CJK_FONT:
    resolved = Path(findfont(FontProperties(family=MPL_CJK_FONT))).name
    ok(resolved.lower() != "dejavusans.ttf",
       f"matplotlib resolves the family to a real file ({resolved})")

print("\n%s  (%d failure%s)" % ("FAIL" if FAILS else "PASS",
                               len(FAILS), "" if len(FAILS) == 1 else "s"))
sys.exit(1 if FAILS else 0)
