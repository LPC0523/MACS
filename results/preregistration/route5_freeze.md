# Route 5 pre-registration — frozen

CVaR_0.90 of |PFC residual| as the objective, rest held at the production level. Chosen on train and validation. Test fold **not run**.

> Pareto optimality applies only to regional MSE/RMS under the identical feasible set; it does not imply Pareto optimality for CVaR, EPI displacement, signal dropout, or measured EPI image quality.

> CVaR of the field is a property of the field, not an EPI image metric. Route 5 measures no displacement, no dropout and no image quality, and supports no claim about them.

> Not wired into the UI. Does not change any app default.

| item | value |
|---|---|
| beta | 0.90 (worst 10%) |
| primary candidate budget | 0% — rest RMS unchanged from production |
| secondary budget reported | 1%% |
| test_run | false |
| voxels | full PFC mask, no subsampling |
| solver | CLARABEL → SCS, exact Rockafellar-Uryasev LP |
| post-solve clipping | none; a violation raises |
| primary metric | PFC CVaR90 = pfc_worst10 |
| constraint metric | rest RMS at the production level |
| secondary metrics | PFC RMS, P95, P99, mean, current and scanner margin |
| coil | PFC 12ch, ±3.0 A |
| scanner window | ±2000 delta ∩ ±2000 absolute |
| split | seed 20260805, by participant, train 20 / validation 10 |
| provenance | no commit yet; SHA-256 per file, see JSON |

## Measured cost, train + validation (n = 30)

**PFC RMS median cost: 1.45 Hz.** That is what optimising the tail gives up on the mean square, at unchanged rest RMS.

| quantity | median (Hz) |
|---|---|
| PFC RMS given up | 1.4544 |
| PFC worst-decile gained | 2.2796 |
| PFC P99 gained | 4.6204 |

Tail improved on 30 of 30 subjects.

Full record: `route5_freeze.json`.
