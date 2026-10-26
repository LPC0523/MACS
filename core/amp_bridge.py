"""Handoff of computed currents from the shim pages to the amplifier page.

The Teensy page is maintained separately, by someone else. Merging the two
would mean every change on their side becomes a change on ours, and every new
board they support becomes a repackage here. So this is a bridge, not a merge:
the shim side publishes a self-describing payload into the shared app_state,
the Teensy side picks it up, and neither knows anything about the other's
internals.

The payload is deliberately plain data -- no numpy arrays, no result objects --
so that when the Teensy program does move into its own process, the transport
can become a JSON file or a socket with this module as the only thing that
changes.

`app_state["pending_currents"]`:

    {
      "coil": "PFC 12ch",
      "channel_names": ["ch1", ...],       # as computed
      "channel_values": [0.12, -0.34, ...],# amps, in channel order
      "current_limit_a": 3.0,
      "timestamp": "2026-08-07T14:32:00",
      "source": "one_click_shim",          # or "hybrid_shim"
      "report_path": "...\\result.json",   # where to record the outcome
      "consumed": False                    # set once the Teensy page shows it
    }
"""

from __future__ import annotations

import datetime
from pathlib import Path

KEY = "pending_currents"


def _now() -> str:
    return datetime.datetime.now().replace(microsecond=0).isoformat()


def publish(app_state: dict, *, currents, channel_names=None, coil_name: str = "",
            current_limit_a: float = 0.0, source: str = "one_click_shim",
            report_path=None, slice_currents=None) -> dict:
    """Put a fresh set of currents where the amplifier page will find them.

    `slice_currents`, when given, is the (n_slices, n_channels) table from a
    dynamic run. `channel_values` still carries a single representative vector
    so the existing static upload path keeps working unchanged; the table is an
    extra field, not a replacement.
    """
    values = [round(float(v), 6) for v in list(currents)]
    names = list(channel_names or []) or [f"ch{i + 1}" for i in range(len(values))]

    table = None
    if slice_currents is not None:
        table = [[round(float(v), 6) for v in row] for row in slice_currents]

    payload = {
        "coil": coil_name,
        "channel_names": names,
        "channel_values": values,
        "slice_currents": table,
        "n_slices": len(table) if table else 0,
        "mode": "dynamic" if table else "static",
        "current_limit_a": float(current_limit_a) or None,
        "timestamp": _now(),
        "source": source,
        "report_path": str(report_path) if report_path else None,
        "consumed": False,
    }
    app_state[KEY] = payload

    # Also fill the fields the Teensy page already reads, so an older upload
    # path keeps working whether or not it knows about this bridge.
    app_state["optimal_currents"] = values
    app_state["ch_names_active"] = names
    app_state["currents_source"] = source
    return payload


def take(app_state: dict) -> dict | None:
    """Return the pending payload once, marking it seen.

    Returns None when there is nothing new, so the amplifier page can tell
    "the user just sent me currents" from "these have been sitting here since
    the last run" -- only the former should announce itself.
    """
    payload = app_state.get(KEY)
    if not payload or payload.get("consumed"):
        return None
    payload["consumed"] = True
    return payload


def peek(app_state: dict) -> dict | None:
    """The pending payload without consuming it."""
    return app_state.get(KEY)


def record_upload(app_state: dict, *, status: str, channels_written: int = 0,
                  dac_readback=None, failed_channels=None,
                  error: str = "") -> Path | None:
    """Append the upload outcome to the run report this payload came from.

    Returns the report path, or None when the currents did not come from a
    recorded run (a hand-entered set, say) -- there is then nothing to append
    to, which is not an error.
    """
    payload = app_state.get(KEY) or {}
    path = payload.get("report_path")
    if not path:
        return None

    from core import run_report

    run_report.record_upload(path, status=status,
                             channels_written=channels_written,
                             dac_readback=dac_readback,
                             failed_channels=failed_channels,
                             error=error)
    return Path(path)
