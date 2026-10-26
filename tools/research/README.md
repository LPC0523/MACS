# tools/research

Batch analyses that produce study results. **Not tests** — nothing here runs in
CI, and a failure means an analysis did not finish, not that the app is broken.

    python tools/research/run_controllability_audit.py
    python tools/research/run_pfc_guarded_pareto.py
    python tools/research/freeze_route2.py

Shared helpers live in `common.py`. These scripts must not import from
`tests/`: a study depending on a test module means renaming a test can break a
result, which is the dependency the wrong way round.

Results are written to `results/` at the project root, unchanged by this move.

`run_*_pareto.py` and `run_cvar_shim.py` default to the train and validation
folds. The test fold stays sealed; opening it is a deliberate `--folds test`.
