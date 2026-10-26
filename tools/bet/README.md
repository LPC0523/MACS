# tools/bet

    python tools/bet/verify_bet_e2e.py

Scores the BET engine **inside a packaged build** against HD-BET references.
It deliberately loads `dist/MACS/_internal`, not the dev tree, because that is
what a user runs.
