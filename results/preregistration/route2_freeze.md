# Route 2 pre-registration — frozen

Candidate operating point: **rest_budget = 1%**, chosen on train and validation. Test fold **not run**.

> Pareto optimality applies only to regional MSE/RMS under the identical feasible set; it does not imply Pareto optimality for CVaR, EPI displacement, signal dropout, or measured EPI image quality.

> This is a research candidate, not an app default, and a 1% rest-RMS budget does not mean the rest of the brain is unaffected in EPI.

| item | value |
|---|---|
| budget | 0.01 |
| reference | whole-brain L-BFGS-B solve, uniform weights, lambda_pfc_low=0 |
| candidate solver | CLARABEL, SCS fallback, factored SOCP |
| guard tolerance | 1e-03 Hz, no post-solve clipping |
| coil | PFC 12ch, ±3.0 A |
| scanner window | ±2000 delta ∩ ±2000 absolute, per term |
| split seed | 20260805, grouped by participant |
| scans | 44 found, 4 excluded, 40 audited |
| folds | train 20, validation 10, test 10 |
| primary metric | PFC RMS under the shared f0 |
| constraint metric | rest RMS, capped at (1+budget)×reference |
| provenance | no commit yet; SHA-256 per file, see JSON |

## File hashes

| file | sha256 |
|---|---|
| `core/pfc_guarded_joint.py` | `684c7f2df44c42ff` |
| `core/controllability_audit.py` | `8c122337aa79d005` |
| `core/subject_split.py` | `8aa91b6279acd60c` |
| `core/shim_metrics.py` | `fb588bc8198db96c` |
| `core/shim_state.py` | `3cae87490ceaaef0` |
| `core/adaptive_joint.py` | `ed1c58416a4261f1` |
| `core/hybrid_shim.py` | `53c6ccbbf1c8976f` |
| `config/shim_delta_bounds.json` | `eddc16eeabcb064a` |
| `tests/run_pfc_guarded_pareto.py` | `29570230f8eb595e` |
| `tests/run_controllability_audit.py` | `03838afe96e545dc` |
| `tests/freeze_route2.py` | `09b2a95729570c4e` |

Full record: `route2_freeze.json`.
