"""The fixed train / validation / test split. One seed, grouped by person.

The cohort is 44 scans but only 32 people: several were scanned twice, once
with the passive earshim and once without (`S30` / `S30_ear`), and three
sessions belong to one participant (`S13_Ref` / `S13_OPT` / `S13_TMS`). Two
scans of the same head share anatomy, susceptibility sources and very nearly
the same field, so putting one in train and the other in test would let a
hyper-parameter chosen on train be validated against a near-copy of itself.
The split is therefore over people, and every scan of a person lands in the
same fold.

The target is 22 / 11 / 11 scans. People carry one, two or three scans each, so
those counts cannot always be hit exactly; the assignment gets as close as it
can and `describe()` reports what was actually achieved rather than the target.

Route 1 is parameter-free and runs on all 44 regardless -- the labels only
group its output. From route 2 on, hyper-parameters may be chosen on train and
validation only, and test is run once, after everything is frozen.
"""

from __future__ import annotations

import re

import numpy as np

__all__ = ["SEED", "TARGET", "person_of", "split_subjects", "describe"]

#: Fixed for the life of the project. Changing it invalidates every number
#: that was ever reported against a fold.
SEED = 20260805

#: Nominal scans per fold, in order (train, validation, test), for the full
#: 44-scan cohort. Only the ratio is enforced: hand in fewer scans -- four are
#: excluded for carrying radian-scale data in a file named hertz -- and the
#: quotas scale down with them.
TARGET = (22, 11, 11)

FOLDS = ("train", "validation", "test")

# Session markers, not different people. `_ear` is the passive earshim
# condition; the rest are repeat sessions of one participant.
_SESSION_SUFFIX = re.compile(
    r"(?:_ear|_no_?ear|_ref|_opt|_tms|_pre|_post|_\d+)$", re.IGNORECASE)


def person_of(name: str) -> str:
    """The participant a scan belongs to.

    Strips session markers repeatedly, so `S13_Ref` and `S13_TMS` both reduce
    to `S13`. Case is normalised because the cohort mixes `S99_ear` with
    `S20_ear`.
    """
    s = str(name).strip()
    while True:
        stripped = _SESSION_SUFFIX.sub("", s)
        if stripped == s or not stripped:
            break
        s = stripped
    return s.upper()


def split_subjects(names) -> dict:
    """Assign every scan a fold. Returns {scan_name: fold}.

    Deterministic given `names` as a set: people are sorted, then shuffled with
    `SEED`, so the result does not depend on the order scans were discovered in.
    """
    names = list(names)
    by_person: dict[str, list[str]] = {}
    for n in names:
        by_person.setdefault(person_of(n), []).append(n)

    people = sorted(by_person)
    rng = np.random.default_rng(SEED)
    rng.shuffle(people)

    total = len(names)
    quota = [t * total / sum(TARGET) for t in TARGET]
    filled = [0.0, 0.0, 0.0]
    out: dict[str, str] = {}

    # Largest people first, so the coarse-grained ones are placed while there
    # is still room to correct for them; ties broken by the shuffled order.
    order = sorted(people, key=lambda p: -len(by_person[p]))
    for p in order:
        k = len(by_person[p])
        # Whichever fold is furthest below its quota, measured after the
        # person would be added -- this avoids overshooting a nearly-full fold
        # with a three-scan participant.
        deficit = [(quota[i] - filled[i]) - 0.5 * k for i in range(3)]
        i = int(np.argmax(deficit))
        filled[i] += k
        for scan in sorted(by_person[p]):
            out[scan] = FOLDS[i]
    return out


def describe(assignment: dict) -> str:
    """One line per fold: people, scans, and the target it was aiming at."""
    lines = []
    total = len(assignment)
    for i, f in enumerate(FOLDS):
        scans = sorted(k for k, v in assignment.items() if v == f)
        people = sorted({person_of(s) for s in scans})
        # The target scales with however many scans were actually handed in.
        # 22/11/11 is the nominal cohort; when scans are excluded the ratio is
        # what carries over, not the absolute counts.
        want = TARGET[i] * total / sum(TARGET)
        lines.append("%-11s %2d scans (target %4.1f)  %2d people   %s"
                     % (f, len(scans), want, len(people), ", ".join(scans)))
    return "\n".join(lines)
