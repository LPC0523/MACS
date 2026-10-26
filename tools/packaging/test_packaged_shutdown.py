"""Does the packaged exe close cleanly? Exit code 0, no forced termination.

    python tools/research/test_packaged_shutdown.py

Every other Qt check in this project ends its process with `os._exit(0)`,
because PySide6 crashes during interpreter teardown when run from source --
after the window is built, navigated and closed, all of which succeed. That
workaround makes those checks say nothing about what a user sees when they
click the X.

This one deliberately does not use it. It launches `dist/MACS/MACS.exe`, waits
for the real main window to exist, sends WM_CLOSE the way the window manager
does, and reads the process exit code. A clean shutdown is exit 0. Anything
else -- a teardown crash, a hang that has to be killed, a non-zero return --
is a defect a user would hit on every session, so it fails here.

Nothing is force-killed unless the app refuses to close within the timeout, and
if that happens it is reported as a failure rather than papered over.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
EXE = ROOT / "dist" / "MACS" / "MACS.exe"

#: How long to wait for the main window to appear before giving up. Cold start
#: on this build loads ANTsPy and onnxruntime, which is not instant.
WINDOW_TIMEOUT_S = 120
#: How long to wait for the process to exit after WM_CLOSE.
CLOSE_TIMEOUT_S = 60

PS = r"""
$ErrorActionPreference = 'Stop'
$exe = '{exe}'
$p = Start-Process -FilePath $exe -PassThru
$deadline = (Get-Date).AddSeconds({wt})
while ($true) {{
    if ($p.HasExited) {{
        Write-Output ("EARLY_EXIT " + $p.ExitCode)
        exit 0
    }}
    $p.Refresh()
    if ($p.MainWindowHandle -ne 0) {{ break }}
    if ((Get-Date) -gt $deadline) {{
        Write-Output 'NO_WINDOW'
        try {{ $p.Kill() }} catch {{}}
        exit 0
    }}
    Start-Sleep -Milliseconds 400
}}
$title = $p.MainWindowTitle
Write-Output ("WINDOW " + $title)

# The real close path: WM_CLOSE, exactly what clicking the X sends.
$sent = $p.CloseMainWindow()
Write-Output ("CLOSE_SENT " + $sent)

if ($p.WaitForExit({ct} * 1000)) {{
    Write-Output ("EXIT " + $p.ExitCode)
}} else {{
    Write-Output 'HUNG'
    try {{ $p.Kill() }} catch {{}}
}}
"""


def main() -> int:
    if not EXE.is_file():
        print("[FAIL] %s not found -- build first" % EXE)
        return 1

    script = PS.format(exe=str(EXE), wt=WINDOW_TIMEOUT_S, ct=CLOSE_TIMEOUT_S)
    r = subprocess.run(
        ["powershell", "-NoProfile", "-NonInteractive", "-Command", script],
        capture_output=True, text=True,
        timeout=WINDOW_TIMEOUT_S + CLOSE_TIMEOUT_S + 120,
        encoding="utf-8", errors="replace")
    out = (r.stdout or "").strip()
    err = (r.stderr or "").strip()
    print(out or "(no output)")
    if err:
        print("stderr: %s" % err[:400])

    fields = dict()
    for line in out.splitlines():
        parts = line.split(None, 1)
        if parts:
            fields[parts[0]] = parts[1] if len(parts) > 1 else ""

    if "NO_WINDOW" in fields:
        print("[FAIL] no main window within %ds" % WINDOW_TIMEOUT_S)
        return 1
    if "EARLY_EXIT" in fields:
        print("[FAIL] exited before the window appeared (code %s)"
              % fields["EARLY_EXIT"])
        return 1
    if "HUNG" in fields:
        print("[FAIL] did not exit within %ds of WM_CLOSE; had to be killed"
              % CLOSE_TIMEOUT_S)
        return 1
    if fields.get("CLOSE_SENT", "").strip().lower() != "true":
        print("[FAIL] CloseMainWindow() was not accepted")
        return 1
    if "EXIT" not in fields:
        print("[FAIL] no exit code reported")
        return 1

    code = fields["EXIT"].strip()
    title = fields.get("WINDOW", "").strip()
    if code != "0":
        print("[FAIL] window '%s' closed with exit code %s, expected 0"
              % (title, code))
        return 1

    print("[OK]   window '%s' closed on WM_CLOSE, exit code 0, no force kill"
          % title)
    return 0


if __name__ == "__main__":
    sys.exit(main())
