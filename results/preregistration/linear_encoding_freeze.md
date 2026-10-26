# Linear-encoding core: frozen numerical definitions

> IDEALIZED OFFLINE SIMULATION. Research module, not in any
> production shim path. What is frozen is the mathematics and the
> code -- not a hardware capability claim.

## Objective

```
maximise   beta_actual = d^T G i

s.t.  d^T G i >= 0
      ||(I - d d^T) G i||        <= tan(theta_max) * d^T G i
      sqrt(mean_W[(R i)^2])      <= epsilon_nl * (d^T G i) * sigma
      lower <= i <= upper
```

No independent nominal beta, and none may be reintroduced.

| item | value |
|---|---|
| gamma | 42.577e6 Hz/T = 42577 Hz/mT |
| field maps | Hz/A, user-confirmed |
| coordinates | NIfTI RAS+, metres, from the affine only |
| scanner frame | NOT AVAILABLE |
| current limit | ±3 A |
| theta_max | required, UI default 5° |
| epsilon_nonlinear | required in every headline number |
| post-solve clipping | never |
| coverage threshold | 95% |
| solver | CLARABEL → SCS, SOCP |
| monotonicity tolerance | 1e-4 relative |

## Verified ranges, for the hashed field maps

| array | region | status |
|---|---|---|
| 12ch PFC | real PFC ROI | verified |
| 12ch PFC | DSV 4 cm, energy-centred | verified, 100% |
| 12ch PFC | DSV 8 cm, energy-centred | verified, 98% |
| 36ch | DSV 8 cm, isocentre | verified, 100% |
| 36ch | DSV 12 cm, isocentre | verified, 99.9% |
| 36ch | DSV 20 cm, isocentre | **NOT VERIFIED**, 46%, TRUNCATED |

No diameter is hard-coded. The UI recomputes coverage from the maps
that are loaded, so a larger calibration enlarges the usable DSV by
itself.

## Still unknown

R, L, mutual inductance, Vmax, amplifier bandwidth, H(f), scanner
ADC synchronisation, gradient coupling. Voltage, power and dynamic
realisability are reported as unknown, never as zero.

Full record: `linear_encoding_freeze.json`.
