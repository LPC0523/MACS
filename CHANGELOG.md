# CHANGELOG

Hashes are recorded so a build in the field can be identified without trusting
what it says about itself: `sha256sum MACS.exe` against the table below.

---

## v1.5.6 — unreleased

### Fixed — the PFC array was making the PFC worse

Adding the 12-channel prefrontal array under the shipping whole-brain objective
degraded the region the array exists for. Measured on S28 (1st+2nd order,
std in Hz over the brain mask):

| | brain | PFC |
|---|---|---|
| no shim | 56.9 | 41.4 |
| SH only | 47.9 | 40.5 |
| SH + 12ch, whole brain | 45.9 | **43.8** |
| SH + 12ch, region-weighted λ=1.0 | 47.2 | **37.2** |

The objective counted every voxel equally and the PFC is 5.7% of the brain, so
the array's twelve degrees of freedom were spent on global field. The coils
ended up standing in for a harmonic rather than doing local work: their
low-order equivalent reached B21 **+299** against a model total of +283,
leaving the scanner −15.7 to apply. In second-order mode the coil field
averaged −0.8 Hz across the PFC while raising its std by 3.8 — spatial
structure, not compensation.

- **The system-shim page now weights a target region, on by default at
  λ_其余 = 1.0.** `solve_adaptive_joint` has taken `pfc_mask`/`lambda_rest`
  all along; the page passed `None`. λ=1.0 is not a preference for the target:
  `make_region_weights` normalises each region by its own voxel count, so 1.0
  is where the two regions carry equal weight rather than equal votes per
  voxel. Costs the brain 1.3 Hz, returns 6.6 to the PFC. Tightening further
  (0.2 → 35.2, 0.05 → 34.6) buys little and degrades the brain steeply
  (50.2, 53.1); at 0.01 the brain reaches 59.7, worse than no shim at all.

- **The region comes from the one-click run**, via `app_state["roi_mask"]` —
  the same registration, on the same grid, not a second one. Missing, empty or
  mismatched masks fall back to whole-brain equal weighting and say so in the
  log and under the checkbox, because silently solving a different problem from
  the one the checkbox promises is how this defect got here.

- **Target-region before/after is reported** (`roi_std_pre` / `roi_std_post`,
  in the RMSE card's tooltip and in `result.json`) whenever a region was used,
  so "did the target actually improve" is answerable from the result alone.

Investigated and rejected: the hypothesis that the cerebellum was pulling the
harmonics into a correction of the same polarity as the PFC's own offset. On
S28 the SH correction over the PFC is **opposite** in sign (−8.0 Hz onto a
+28.4 Hz field), and excluding the whole inferior-posterior brain from the fit
returns only 0.4 Hz to the PFC while costing that region 2.3. The competition
is with the whole brain by voxel count, not with one structure by geometry.

`tests/test_system_shim_plain_ls.py` covers the weighting and the fallbacks.

### Fixed — the centre-frequency offset had no stated baseline

v1.5.5 reported the centre frequency as a bare increment ("+20.3 (增量)") on
the grounds that the page never reads the console's f0. That leaves the
operator to supply the baseline, and the obvious choice is the wrong one.
`deltaB0` is local resonance minus the transmit frequency **the field map was
acquired at**; if a prescan runs between the field map and the shim entry, it
re-locks the console to the mean resonance of the *unshimmed* field, and the
console then reads `f0_map + mean(unshimmed b0)`. Adding the offset there
overshoots by the pre-shim mean — tens of Hz, routinely larger than the offset
itself. Nothing on screen distinguished the two baselines.

- **The field map's own transmit frequency is now read from DICOM
  (0018,0084)**, converted MHz → Hz, and carried through
  `PipelineResult.imaging_frequency_hz` → `app_state` → the system-shim page.
  Only slices `_is_b0_series` accepts contribute, so an EPI in the same study
  cannot supply its own re-locked f0.

- **It is resolved per acquisition, not per folder** — `sidecar_frequency_hz`
  reads the sidecar of the pair `_find_phase_magnitude` actually chose, and
  `probe_acquisition`'s folder-wide value is only a fallback for when the
  folder is unambiguous. Measured on a 5T phantom session (two phantoms, three
  table positions, 21 accepted B0 acquisitions in 105 min) the accepted series
  span 823 Hz. Most of that spread is legitimate — a different phantom at a
  different table position resonates elsewhere, and eight of those series are
  named `f0cal` because calibrating f0 *was* the experiment — which is the
  point: a folder-wide answer would be a different scan's zero. Verified on
  that data: a folder holding two acquisitions 671 Hz apart reports the f0 of
  the pair that was processed, where the folder-wide implementation reported
  nothing at all. Within the one table position that matters for the shim
  workflow, two field maps nine minutes apart with a vendor adjustment between
  them agree exactly, so this is not a claim that f0 always moves.

- **Phase and magnitude are cross-checked.** One acquisition has one f0; two
  different ones mean the pair was assembled across a recalibration, which
  makes the field map itself wrong rather than merely its zero, so neither
  value is used and the log says so. `files["phase"]` is now exposed for this.

- **The f0 row is absolute when that frequency is known** — `123.253430` →
  `123.253450` MHz instead of `+20.3` — so there is nothing left to attach to
  the wrong number. Six decimals in MHz is 1 Hz, which is the tag's own stored
  resolution. Without it the row stays an increment but now names its baseline.

- **The probe runs regardless of the auto-parameters checkbox.** That flag
  governs whether the DICOM's ΔTE and field strength *override* the page's
  values; the transmit frequency overrides nothing, it is provenance.

- **`imaging_frequency_hz` and `centre_frequency_target_hz` go into the
  exported `result.json`**, and `imaging_frequency_hz` into the run report's
  `acquisition` block — by the time either is read, the console has long since
  re-locked its own f0.

`tests/test_imaging_frequency.py` is new; `test_system_shim_plain_ls.py` gained
the absolute-row case.

### Fixed — a repeated protocol silently overwrote itself on conversion

Series are grouped by SeriesNumber but *named* by SeriesDescription, and a
session that runs one protocol repeatedly gives every repeat the same
description. `_run_pydicom` wrote each to that one filename, so only the last
survived — and because the three components are written independently, the
survivors need not have come from the same acquisition. The result is a field
map assembled from two different scans, with nothing on screen to show for it.

Measured on the same 5T session: **72 accepted B0 series collapsing onto 18
filenames**, one description used 11 times. Reproduced directly — six DICOM
series from two acquisitions converted to three NIfTIs, the log printing the
same filename twice with a tick each time.

- **Collisions are detected before writing**, and every series then gets its
  acquisition as a filename prefix (`a09_…`, `a14_…`). A prefix rather than a
  suffix because `_acquisition_key` strips role suffixes off the end, so only a
  prefix survives it — the three components still group, and phase still pairs
  with the magnitude from its own acquisition. Folders holding one acquisition,
  which is what the app is normally handed, keep the names they had.

- **The acquisition comes from `SeriesNumber // 100`** — the console numbers
  acquisition N as N01/N02/N03. Validated across the whole export: 50 groups,
  every one internally consistent in transmit frequency, none mixed. Numbering
  that does not fit (`SeriesNumber < 100`) falls back to the number whole, so
  the files stay distinct and pairing fails loudly rather than mixing scans.

- **`tools/sort_by_acquisition.py`** splits a flat export into one folder per
  acquisition by the same rule, with `--dry-run` to inspect the grouping first.
  Needed because the app, handed 24 complete field maps, takes the first by
  name — the earliest, rarely the one meant.

`tests/test_series_pairing.py` gained the repeated-protocol case.

---

## v1.5.5 — 2026-08-14

Built 2026-08-14T16:48 from `ee516f8+` · stamp `MACS v1.5.5 (build 2026-08-14T16:48, ee516f8+)`

**v1.5.4 is withdrawn** — its system-shim page solves a different problem from
the one described below and its coefficients are 12–27% short on the
second-order terms. `release/MACS_v1.5.4/` and its ZIP have been deleted; the
hashes stay recorded under v1.5.4 so a copy already in the field can be
identified as one to replace.

| artefact | value |
|---|---|
| `dist/MACS/MACS.exe` | 35,771,672 bytes (34.11 MB) |
| exe SHA-256 | `6eae0c88a0406fd9aa8da661b96fca7a1dc1902167c9588fca8cb8368148d271` |
| package | `dist/MACS/` — 1,873 files, 596.2 MB |
| `release/MACS_v1.5.5/` | 1,878 files, 596.3 MB |
| `release/MACS_v1.5.5.zip` | 267,056,071 bytes (254.7 MB) |
| zip SHA-256 | `3a9ff5931066edd1ba8f536d0e23477a9f22a941aa077a4ec60c576451fa5d86` |

### Changed — the system-shim solve is now plain least squares

- **Centring is off for the system-shim page.** The solve subtracted the
  in-mask weighted mean from both `b0` and every basis column before solving,
  and subtracted the mean from the residual afterwards. That is exactly
  equivalent to appending a free constant column (A00) to the design and
  projecting it out: a uniform offset costs the optimiser nothing, so the
  harmonics are never asked to correct it.

  That is the right problem only if something re-locks the centre frequency
  between the shim update and the acquisition. It does not happen here — the
  eight coefficients are typed into the console and the next scan runs with
  them. So the constant is real error, and the page now solves

      min || A w + b0 ||²   over the brain mask

  with no constant column, which is the formulation in
  `2nd_order_SH/SH_coefficient_conversion_memo.md` §6.

  Measured on S28 (208,771 voxels, second order only), the two solves differ by
  more than rounding:

  | term | plain (now) | centred (was) | shift |
  |---|---|---|---|
  | A20 | 107.34 | 77.88 | −27% |
  | A21 | 709.96 | 586.33 | −17% |
  | B21 | 283.37 | 230.47 | −19% |
  | A22 | −47.70 | −41.79 | −12% |
  | B22 | 22.05 | 16.89 | −23% |

  They optimise different things and each wins on its own metric: plain reaches
  55.19 Hz RMSE against centred's 55.56, centred reaches 50.94 Hz std against
  plain's 51.31. RMSE is what will be measured, so RMSE is what is minimised.

- **λ_SH now defaults to 0.** Eight well-separated harmonics over 200k voxels
  need no ridge for conditioning; it only biased the coefficients toward zero,
  and the coefficients are the deliverable. The spin box still goes to 1.0.
  λ_coil keeps its 0.5 default — that one limits current, which is a real
  amplifier constraint rather than a numerical convenience.

- **The residual is no longer shifted before reporting.** The headline card is
  now `全脑 RMSE` rather than `全脑 STD`, and the centre-frequency card reads
  `中心频率残余` — the offset the harmonics could not remove, already inside
  the reported RMSE, with nothing downstream that will take it out. Previously
  the page subtracted it and said so in a tooltip, which flattered the result
  by however far off-resonance the map was acquired (+20.3 Hz on S28).

- **The centre frequency is now offered as a correction.** With the constant
  no longer written off, the residual mean is a real, actionable quantity: the
  eight harmonics contain no zeroth order and cannot remove it, but f0 can. The
  SH table gained a last row, `中心频率 (f0, Hz)`, carrying `+offset` as a
  **change** — the console's f0 is an absolute frequency in the hundreds of MHz
  and this page never reads it, so unlike every other row there is no current
  value to add to, and the cell says so. Applying it subtracts the constant
  from the residual, which is exactly the gap between RMSE and std: on S28,
  55.19 → 51.31 Hz.

  Sign follows `deltaB0 = 局部共振频率 − 当前发射频率`, so a residual mean of
  +20.3 Hz means the tissue precesses 20.3 Hz above f0 and f0 goes up by that
  much. That convention is inherited from the phase unwrapping upstream rather
  than established here, so the row states it in its tooltip for the operator
  to check against the console instead of leaving a bare number to be trusted.
  If f0 is not adjustable on site, the row can be ignored — the reported RMSE
  already includes the offset either way. `hybrid_shim_result.json` carries it
  as `centre_frequency_delta_hz`.

- **Not changed: the coil/harmonic split.** `build_low_order_projection` exists
  to stop the coils and the harmonics competing for the same low-order field,
  which is a collinearity problem, not a frequency-offset one. It only runs
  when a coil array is loaded, and the SH-only path never touches it.

`solve_adaptive_joint` keeps `centre=True` as its default, so `tools/research/`
and the frozen pre-registration baselines in `results/preregistration/` — which
name it as their reference solution — still compute what they always did. Only
`hybrid_shim.joint_optimize`, reached solely from the system-shim page, passes
`centre=False`.

Verified against `np.linalg.lstsq` on S28 through the full `joint_optimize`
path: agreement to 2.9e-8 on both the five-term and eight-term solves, and the
reported residual mean now equals the true residual mean.
`tests/test_system_shim_plain_ls.py` holds the regression — the solve against
lstsq, the residual keeping its offset, `centre=True` still differing and still
being the default, and the f0 row's sign and labelling.

---

## v1.5.4 — 2026-08-14

Built 2026-08-14T15:20 from `ee516f8+` · stamp `MACS v1.5.4 (build 2026-08-14T15:20, ee516f8+)`

The `+` means the tree was dirty at package time: the fix below was not
committed when this package was built.

Repackaged at 15:20 after dropping two unused libraries — see *Packaging*. The
10:43 build (exe `441ea969…`, 724.1 MB) is superseded and should not be
distributed.

| artefact | value |
|---|---|
| `dist/MACS/MACS.exe` | 35,769,671 bytes (34.11 MB) |
| exe SHA-256 | `0314e5ac97083cf00854f84ea6871863b8da91819cd843a86d84ee6bf0284413` |
| package | `dist/MACS/` — 1,873 files, 596.2 MB |
| `release/MACS_v1.5.4/` | 1,878 files, 596.3 MB |
| `release/MACS_v1.5.4.zip` | 267,050,798 bytes (254.7 MB) |
| zip SHA-256 | `782743863663ccb31c2dfef5f2f0a6939b3c0fee659e0063a16052fbd2924579` |

### Packaging

- **Dropped `pyarrow` (76.8 MB) and `cvxopt` (50.2 MB) from the bundle.**
  Neither is imported anywhere in the app: pyarrow is an optional pandas
  backend and cvxopt an optional cvxpy solver, and cvxpy reaches its solvers
  through `try`/`except`, so a missing one is not an error. The build ships
  CLARABEL and SCS, which is what the linear-encoding SOCP actually solves
  with — `--selftest` still solves a real cone program at 1200.0 Hz/m on
  CLARABEL after the removal. 724.1 MB → 596.2 MB; the ZIP is 254.7 MB.

- **`scripts/make_release.py` now assembles the distributable.** It runs
  `MACS.exe --selftest` and refuses to package a build that does not pass,
  then writes `release/MACS_v{VERSION}/` with the launcher, the Chinese
  readme and the manuals, zips it, and prints the sizes and SHA-256s this
  table needs. The previous releases were assembled by hand, which is how
  `v1.5.0`'s readme came to describe a layout nobody re-verified.

- **Verified portable, not just built.** The ZIP was extracted to a path
  unrelated to the build tree and self-tested there: `app_base` follows the
  extraction, and ANTsPy, cpp_bet/ONNX Runtime and the CLARABEL SOCP all come
  up. The bundle carries its own `VCRUNTIME140.dll`, so a clean Windows 10/11
  machine needs nothing installed.

- **Longest path inside the bundle is 115 characters**
  (`_internal/sklearn/metrics/_pairwise_distances_reduction/…`). Against the
  260-character Windows default that leaves ~140 for the extraction root —
  enough for a desktop or `D:\MACS`, not for a deeply nested sync folder. The
  readme says so, because the failure mode is an opaque "找不到路径的一部分"
  from the unzip step rather than anything the app can report.

### Fixed

- **The dynamic-shim result read as rows of empty rectangles.** The current
  table and the per-slice bar chart labelled themselves in Chinese —
  「通道」, 「层 (slice)」, 「电流 (A)」 and the table's own title — and
  matplotlib draws with DejaVu Sans, which has no CJK glyphs. Qt renders the
  panel around the figures with a system font that does, so only the plots
  broke, which is why it looked like a text-encoding fault rather than a
  missing font. The figures speak English now, like the linear-encoding
  page's already did, and so does the caption above the slice card.

- **Chinese arriving as data would have hit the same wall.** A coil whose
  channel names are Chinese puts them on the tick labels, and no rewrite of
  the app's own strings covers that. `core.theme` now picks the first
  installed CJK face (Microsoft YaHei, SimHei, PingFang SC, Noto Sans SC …)
  and puts it in front of matplotlib's font stack at import time, falling
  back silently on a machine that has none. `axes.unicode_minus` is off, so
  a negative tick does not become the one box left on the figure.
  `tests/test_cjk_labels.py` holds both halves: no widget may pass a Chinese
  literal to matplotlib, and Chinese that arrives as data must render with
  no missing glyphs.

### Build chain

Packaging was blocked by the environment, not the code: `D:\python.exe` had
three `setuptools` dist-info directories (65.5.0, 82.0.1, 83.0.0) over one
mixed file tree, so `import setuptools` failed on a vendored `backports`
import and PyInstaller's `hook-setuptools` compared `None < (71, 0)`.
Reinstalled clean as 84.0.0. Note that setuptools ≥ 81 no longer ships
`pkg_resources`; nothing in the app or its dependencies imports it.

---

## v1.5.3 — 2026-08-10

Commit `5d466b4` · built 2026-08-10T15:29 · stamp `MACS v1.5.3 (build 2026-08-10T15:29, 5d466b4)`

| artefact | value |
|---|---|
| `dist/MACS/MACS.exe` | 34,279,116 bytes (32.69 MB) |
| exe SHA-256 | `89ee3de5837e35a1de6106b19e52b2031f4eed9cd5d510aea61e2db43bb5d0b3` |
| package | `dist/MACS/` — 2,502 files, 689.9 MB |
| package SHA-256 | `9890b3948d69e2b050278a2dd26c749bbeeb7f97e73a5b196125ff4224e1ea58` |

The package hash covers every file's path and contents, so it changes if any
asset is swapped even when the exe is untouched.

### Fixed

Five defects, all found by running v1.5.2 on infant CBCP data. The first
happened on every session.

- **The app crashed when closed.** `main_window.py` had no `closeEvent`, and Qt
  sends close events only to top-level windows — so `TeensyWidget.closeEvent`,
  which stops the serial reader and the port-refresh timer, never ran: the page
  is a child of a `QStackedWidget`, not a window. Closing the packaged build
  printed `QThread: Destroyed while thread '' is still running` and aborted
  with `0xC0000409`. `MainWindow` now asks each page to shut down, then stops
  any thread still running. `terminate()` is deliberately not used: it kills a
  thread wherever it happens to be, and these run Python, so terminating one
  holding the GIL hangs the process outright — which is what the first attempt
  at this fix did. The self-check worker honours `isInterruptionRequested()`
  instead.

- **A protocol named `B0fieldmap` lost all four of its magnitude echoes.** The
  series filter's left edge broke on digits, so the `0` in front of `field`
  stopped `field\s*map` from ever matching; only the phase series survived, on
  its own separate `_B0Map`. The run failed with "Could not identify
  phase/magnitude images" — and, not being an error, ΔTE could no longer be
  read from the two echoes and quietly fell back to whatever the page had typed
  in it. The edge now breaks on letters only; `PALB0Map_M` and `CalB0Map` are
  still rejected.

- **Phase and magnitude could come from different acquisitions.** An infant
  CBCP export carries `B0fieldmap_ref_*` and `B0fieldmap_shim_*`, before and
  after the scanner's own shim. `_find_phase_magnitude` assigned by overwriting
  as it walked the folder and could pair across the two. Files are grouped by
  acquisition first; with more than one complete set, both are named in the log
  and the first is used, because which of a before/after pair is correct is not
  something the code can decide.

- **The ROI box stuck on an anatomical region in dynamic mode.**
  `run_pipeline` already forces whole brain there, but the widget only greyed
  the box out, so it sat disabled reading 「前额叶」 — a region that was not
  going to be used and could not be changed. It is reset to whole brain now,
  and the previous choice is restored on the way back to static.

- **A console window opened behind the GUI.** The build stays `console=True` so
  `--selftest` can print and `check_package.py` can read it; the console is now
  hidden at GUI startup, but only when it belongs to this process, so launching
  from an existing terminal does not hide the user's own window.

### Fixed — build chain

- **Every package build left `version.py` dirty.** `restore()` wrote text, and
  that translates newlines, so a file checked out with LF came back as CRLF —
  same content, different bytes. git reported it modified after every build,
  and the *next* build stamped a `+` onto a tree whose only change was the
  previous build's own restore. Stamp and restore now read and write bytes.

- **That fix silently produced an unstamped build.** Keeping CRLF meant
  `^BUILD_DATE = ".*"$` no longer matched: under `re.M` the `$` sits before the
  newline, but the line still ends in a carriage return that the trailing quote
  cannot reach. The substitution did nothing, and the exe shipped reporting
  "build dev". `scripts/check_package.py` caught it. The end-of-line anchor is
  gone and the substitution count is now checked — a stamp that matches nothing
  raises instead of shipping.

### Added

- `tests/smoke_production.py` — nine checks run before packaging. Qt checks run
  in a subprocess ending in `os._exit(0)`: PySide6 crashes during interpreter
  teardown from source, after the window has been built, navigated and closed,
  and that would otherwise report failure for work that finished.
- `tests/test_packaged_shutdown.py` — launches the packaged exe, waits for the
  real window, sends `WM_CLOSE`, and requires exit code 0. Deliberately does
  *not* use `os._exit`, which is the point.
- `tests/test_series_pairing.py` — the series filter and acquisition pairing,
  against the real series names from the failing dataset.
- `scripts/check_package.py` — verifies a built package: no research-only
  module shipped, required assets present, packaged `--selftest` passes, and
  the version actually stamped.

### Verified

- 9/9 production smoke checks; `check_package.py` passed; packaged exe closes
  on `WM_CLOSE` with exit code 0.
- Real data `I01 (one infant CBCP session)`: all six B0 series convert,
  `ref` is selected and reported, ΔTE reads 4.92/7.38 ms from the headers,
  whole-brain RMSE 36.2 → 27.5 Hz over 36 channels at max 1.60 A.
- `ref` vs `shim` cross-check, one shared brain mask of 91,381 voxels:
  ref 36.23 Hz, shim 34.31 Hz. The scanner's shim is flatter, so the two groups
  are named as expected — but by only 5.3%, which is small for a scanner shim
  and worth confirming with the scanner side before `ref` is described as an
  unshimmed baseline in a paper.

---

## v1.5.2 — 2026-08-10

Commit `7e8de94` — the source baseline, first commit in the repository.

Tagged for the record rather than for release: this is the state whose exit
crash and `B0fieldmap` filter defects v1.5.3 fixes. Not recommended for use.

The repository root is the whole desktop toolbox, so `.gitignore` inverts the
usual rule — everything is ignored and the MACS source tree is let back in by
name. Subject DICOM/NIfTI, coil field maps, atlases, ONNX weights, PyInstaller
output, the build virtualenv, research results, logs and archives stay out.
