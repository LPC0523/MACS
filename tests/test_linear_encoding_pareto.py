"""The SOCP frontier, the direction cone, and per-sample bipolar bounds.

    D:\\python.exe tests/test_linear_encoding_pareto.py

The feasibility certificate is the load-bearing test. The ray solution's own
NRMSE is an achievable relative error, so every epsilon at or above it must be
feasible and must reach at least the ray's amplitude -- if the SOCP ever comes
back with less, it is not solving the problem it claims to.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core.linear_encoding_metrics import (GAMMA_HZ_PER_MT,  # noqa: E402
                                          check_current_bounds, to_mt_per_m)
from core.linear_encoding_model import UnitStatus, build_model  # noqa: E402
from core.linear_encoding_pareto import (MAX_CONE_DEG,  # noqa: E402
                                         beta_ray_limited, dynamic_bounds,
                                         max_gradient_socp, pareto_frontier)
from core.spatial_grid import world_coordinates  # noqa: E402

FAILS = []


def ok(cond, msg):
    print("  [%s] %s" % ("OK" if cond else "FAIL", msg))
    if not cond:
        FAILS.append(msg)


def close(a, b, tol, msg):
    d = abs(float(a) - float(b))
    ok(d <= tol, "%s  (|%.6g - %.6g| = %.3g <= %.3g)" % (msg, a, b, d, tol))


SHAPE = (11, 12, 13)
AFF = np.diag([2.0, 2.0, 2.0, 1.0])
AFF[:3, 3] = [-11.0, -12.0, -13.0]
MASK = np.ones(SHAPE, dtype=bool)


def synthetic(fns):
    pts = world_coordinates(AFF, SHAPE)
    x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]
    return np.stack([f(x, y, z).reshape(SHAPE) for f in fns], -1).astype(float)


def model_of(fns):
    return build_model(synthetic(fns), MASK, AFF,
                       unit_status=UnitStatus.USER_CONFIRMED)


# ── 1. one conversion, everywhere ──────────────────────────────────
def t_conversion():
    print("1. Hz/m -> mT/m")
    close(GAMMA_HZ_PER_MT, 42577.0, 1e-9, "gamma = 42577 Hz/mT")
    close(float(to_mt_per_m(42577.0)), 1.0, 1e-12, "42577 Hz/m == 1 mT/m")
    import core.linear_encoding_pareto as _P
    ok(not hasattr(_P, "hz_per_m_to_mt_per_m"),
       "the duplicate conversion helper is gone; one entry point only")
    ok(np.allclose(to_mt_per_m([0.0, 42577.0, 85154.0]), [0.0, 1.0, 2.0]),
       "vectorised and linear")


# ── 2. a perfect linear coil: epsilon buys nothing ─────────────────
def t_exact():
    print("2. a perfectly linear array: relaxing epsilon adds no gradient")
    G0 = 400.0
    m = model_of([lambda x, y, z: G0 * x, lambda x, y, z: 300.0 * y,
                  lambda x, y, z: 250.0 * z])
    ray = beta_ray_limited(m, [1, 0, 0], 3.0)
    close(ray["beta_ray_limited_hz_per_m"], 3.0 * G0, 1e-6,
          "ray reaches Imax * G0")

    # The nonlinear residual is identically zero, so its constraint is
    # inactive at every epsilon and the box alone decides. A solver that
    # returned more at larger epsilon would be inventing gradient out of a
    # slack it cannot spend -- which is exactly what the old free-beta
    # objective did, reporting 1333 / 1714 / 2400 for the same 1200 Hz/m.
    for eps in (0.10, 0.30, 0.50):
        r = max_gradient_socp(m, [1, 0, 0], eps, 3.0)
        ok(r["feasible"], "eps=%.2f feasible" % eps)
        close(r["gradient_parallel_actual_hz_per_m"], 3.0 * G0, 1e-3,
              "eps=%.2f delivers Imax*G0 and no more" % eps)
        close(r["nonlinear_residual_ratio"], 0.0, 1e-6,
              "eps=%.2f: no nonlinear residual to spend" % eps)
        close(r["direction_error_deg"], 0.0, 1e-3, "eps=%.2f: on axis" % eps)
        close(r["box_violation_a"], 0.0, 1e-8, "no box violation")
        close(r["nonlinear_constraint_relative_violation"], 0.0, 1e-6,
              "no nonlinearity violation")
        ok(r["clipping_applied"] is False, "no clipping")


# ── 3. the objective must prefer the larger delivered gradient ─────
def t_adversarial():
    print("3. adversarial: delivered gradient wins over nominal amplitude")
    # Two channels, deliberately built so that the old objective and the new
    # one disagree. ch1 is a clean x gradient. ch2 is a weaker x gradient
    # carrying a large nonlinear term, so a solution leaning on ch2 has a
    # bigger *fit-relative* amplitude but delivers less real gradient.
    m = model_of([lambda x, y, z: 340.0 * x,
                  lambda x, y, z: 300.0 * x + 900.0 * (x ** 2 - y ** 2)])
    r = max_gradient_socp(m, [1, 0, 0], 0.50, 3.0)
    ok(r["feasible"], "feasible")
    iv = np.asarray(r["currents_a"])
    print("       currents %s -> delivered %.1f Hz/m, nl ratio %.3f"
          % (np.round(iv, 3), r["gradient_parallel_actual_hz_per_m"],
             r["nonlinear_residual_ratio"]))

    # Enumerate the corners of the box and confirm the solver found the best
    # *delivered* one among the feasible points, not merely a feasible one.
    best_ok, best_g = None, -np.inf
    for i1 in np.linspace(-3, 3, 25):
        for i2 in np.linspace(-3, 3, 25):
            cand = np.array([i1, i2])
            g = float((m.G @ cand)[0])
            if g <= 0:
                continue
            rr = float(np.linalg.norm(_L(m) @ cand))
            if rr <= 0.50 * _sigma(m) * g + 1e-9:
                if g > best_g:
                    best_g, best_ok = g, cand
    print("       grid search best feasible delivered = %.1f Hz/m at %s"
          % (best_g, np.round(best_ok, 3)))
    ok(r["gradient_parallel_actual_hz_per_m"] >= best_g - 1e-3,
       "the SOCP matches or beats an exhaustive feasible search "
       "(%.1f >= %.1f)" % (r["gradient_parallel_actual_hz_per_m"], best_g))

    # And the pure-ch1 solution, which the old objective would have passed
    # over, is not worse than what we returned.
    pure = np.array([3.0, 0.0])
    g_pure = float((m.G @ pure)[0])
    ok(r["gradient_parallel_actual_hz_per_m"] >= g_pure - 1e-3,
       "never below the clean single-channel solution (%.1f >= %.1f)"
       % (r["gradient_parallel_actual_hz_per_m"], g_pure))


def _L(m):
    # _residual_factor returns (L, info): the factor plus what its
    # eigenvalue clipping did, so callers can see the regularisation.
    from core.linear_encoding_pareto import _residual_factor
    return _residual_factor(m)[0]


def _sigma(m):
    from core.linear_encoding_pareto import position_sigma
    return position_sigma(m, [1, 0, 0])


# ── 3b. monotonicity in epsilon and in theta ───────────────────────
def t_monotone():
    print("3b. delivered gradient is monotone in epsilon and theta")
    TOL = 1e-4          # relative; matches the conic solvers' own accuracy
    m = model_of([lambda x, y, z: 200.0 * x + 400.0 * y,
                  lambda x, y, z: 150.0 * x + 90.0 * z,
                  lambda x, y, z: 300.0 * (x ** 2 - y ** 2) + 60.0 * x])

    prev, prev_e = -np.inf, None
    for eps in (0.10, 0.20, 0.30, 0.40, 0.50, 0.70):
        r = max_gradient_socp(m, [1, 0, 0], eps, 3.0)
        if not r["feasible"]:
            print("       eps=%.2f infeasible" % eps)
            continue
        g = r["gradient_parallel_actual_hz_per_m"]
        drop = (prev - g) / max(abs(prev), 1.0) if np.isfinite(prev) else 0.0
        ok(drop <= TOL,
           "eps %s->%.2f: delivered %.1f does not fall (rel drop %.2e)"
           % (prev_e, eps, g, max(drop, 0.0)))
        prev, prev_e = max(prev, g), "%.2f" % eps

    # 90 degrees is not a cone and tan() cannot express it; "no cone" is
    # theta_max_deg=None, tested separately below.
    for bad in (90.0, 0.0, -5.0, 60.0):
        try:
            max_gradient_socp(m, [1, 0, 0], 0.50, 3.0, theta_max_deg=bad)
            ok(False, "theta=%.0f should be rejected" % bad)
        except ValueError:
            ok(True, "theta=%.0f rejected before reaching tan()" % bad)
    ok(max_gradient_socp(m, [1, 0, 0], 0.50, 3.0,
                         theta_max_deg=None)["feasible"],
       "theta=None means no cone and is accepted")

    prev, prev_t = -np.inf, None
    for th in (1.0, 2.0, 5.0, 10.0, 30.0, MAX_CONE_DEG):
        r = max_gradient_socp(m, [1, 0, 0], 0.50, 3.0, theta_max_deg=th)
        if not r["feasible"]:
            print("       theta=%.0f infeasible" % th)
            continue
        g = r["gradient_parallel_actual_hz_per_m"]
        drop = (prev - g) / max(abs(prev), 1.0) if np.isfinite(prev) else 0.0
        ok(drop <= TOL,
           "theta %s->%.0f deg: delivered %.1f does not fall (rel drop %.2e)"
           % (prev_t, th, g, max(drop, 0.0)))
        prev, prev_t = max(prev, g), "%.0f" % th

    # A returned vector is only feasible if it satisfies its own constraints.
    r = max_gradient_socp(m, [1, 0, 0], 0.50, 3.0, theta_max_deg=5.0)
    if r["feasible"]:
        from core.linear_encoding_pareto import SOLVER_TOL
        ok(r["box_violation_a"] <= SOLVER_TOL["box_violation_a"],
           "accepted solutions respect the box within %.0e A"
           % SOLVER_TOL["box_violation_a"])
        ok(r["nonlinear_constraint_relative_violation"]
           <= SOLVER_TOL["nonlinear_constraint_relative_violation"],
           "and the nonlinearity budget")
        ok(r["direction_cone_relative_violation"]
           <= SOLVER_TOL["direction_cone_relative_violation"],
           "and the direction cone")
        ok("solver_attempts" in r and r["solver_attempts"],
           "every solver attempt is recorded")
        rf = r["residual_factor"]
        # Check the mechanism, not the prose: the string says "no jitter",
        # which a word search flags as containing "jitter".
        for k in ("min_eigenvalue", "max_eigenvalue",
                  "clipped_eigenvalue_count", "clipping_tolerance"):
            ok(k in rf, "residual factor reports %s" % k)
        ok(rf["clipped_eigenvalue_count"] >= 0
           and rf["min_eigenvalue"] >= -rf["clipping_tolerance"],
           "only numerically-negative eigenvalues were clipped (min %.2e, "
           "tol %.2e)" % (rf["min_eigenvalue"], rf["clipping_tolerance"]))
        ok(rf["suspicious"] is False,
           "no genuinely negative eigenvalue, so the decomposition is sound")


# ── 3c. the ray is a feasibility witness ───────────────────────────
def t_certificate():
    print("3c. the ray's nonlinearity ratio witnesses which epsilons work")
    m = model_of([lambda x, y, z: 200.0 * x + 400.0 * y,
                  lambda x, y, z: 150.0 * x + 90.0 * z,
                  lambda x, y, z: 300.0 * (x ** 2 - y ** 2) + 60.0 * x])
    ray = beta_ray_limited(m, [1, 0, 0], 3.0)
    par = ray["metrics"]["gradient_along_target_hz_per_m"]
    e0 = ray["metrics"]["nonlinear_rms_hz"] / max(par * _sigma(m), 1e-30)
    print("       ray delivers %.1f Hz/m at nonlinearity ratio %.4f"
          % (par, e0))
    ok(0.0 < e0 < 5.0, "the ray leaves a real, finite nonlinear residual")

    # The ray is one feasible point, not a bound. It minimises total fit error
    # -- orthogonal leakage included -- rather than the nonlinearity ratio, and
    # its amplitude is capped by whichever channel saturates first along that
    # single ray. So a *tighter* nonlinearity budget can still deliver more,
    # by choosing a different current direction. Only the upper relation is a
    # theorem, and it is asserted below.
    below = max_gradient_socp(m, [1, 0, 0], max(e0 * 0.5, 1e-4), 3.0)
    if below["feasible"]:
        ok(below["nonlinear_residual_ratio"] <= e0 * 0.5 * (1 + 1e-4),
           "a tighter budget is respected (ratio %.4f <= %.4f)"
           % (below["nonlinear_residual_ratio"], e0 * 0.5))
        print("       tighter budget still delivers %.1f Hz/m -- the ray is "
              "a witness, not a ceiling"
              % below["gradient_parallel_actual_hz_per_m"])
    else:
        ok(below.get("shape_reachable") is False,
           "a much tighter budget is reported unreachable, not as 0 Hz/m")

    above = max_gradient_socp(m, [1, 0, 0], e0 * 1.001, 3.0)
    ok(above["feasible"], "a budget just above the ray's ratio is feasible")
    ok(above["gradient_parallel_actual_hz_per_m"] >= par * (1 - 1e-4),
       "and delivers at least what the ray delivers (%.1f >= %.1f)"
       % (above["gradient_parallel_actual_hz_per_m"], par))

    # Homogeneity: doubling the current limit doubles the gradient and leaves
    # every ratio untouched, because both sides of each cone scale together.
    a1 = max_gradient_socp(m, [1, 0, 0], e0 * 2, 3.0)
    a2 = max_gradient_socp(m, [1, 0, 0], e0 * 2, 6.0)
    close(a2["gradient_parallel_actual_hz_per_m"]
          / a1["gradient_parallel_actual_hz_per_m"], 2.0, 1e-4,
          "doubling Imax doubles the delivered gradient")
    close(a2["nonlinear_residual_ratio"], a1["nonlinear_residual_ratio"], 1e-6,
          "and leaves the nonlinearity ratio unchanged")


# ── 4. the direction cone constrains what NRMSE cannot ─────────────
def t_cone():
    print("4. the direction cone separates tilt from nonlinearity")
    # A channel set whose best x fit is tilted towards y.
    m = model_of([lambda x, y, z: 300.0 * x + 260.0 * y,
                  lambda x, y, z: 120.0 * x - 40.0 * y,
                  lambda x, y, z: 100.0 * z])
    free = max_gradient_socp(m, [1, 0, 0], 0.50, 3.0)
    ok(free["feasible"], "feasible without a cone")
    print("       no cone: dir_err %.2f deg, leak ratio %.3f, delivered %.1f"
          % (free["direction_error_deg"], free["orthogonal_leakage_ratio"],
             free["gradient_parallel_actual_hz_per_m"]))

    prev_beta = free["gradient_parallel_actual_hz_per_m"]
    for th in (10.0, 5.0, 2.0):
        r = max_gradient_socp(m, [1, 0, 0], 0.50, 3.0, theta_max_deg=th)
        if not r["feasible"]:
            print("       theta<=%.0f deg: infeasible" % th)
            continue
        ok(r["direction_error_deg"] <= th + 1e-3,
           "theta<=%.0f deg is respected (%.3f deg)" % (th, r["direction_error_deg"]))
        ok(r["gradient_parallel_actual_hz_per_m"] <= prev_beta * (1 + 1e-4),
           "tightening the cone cannot increase delivered (%.1f <= %.1f)"
           % (r["gradient_parallel_actual_hz_per_m"], prev_beta))
        prev_beta = r["gradient_parallel_actual_hz_per_m"]

    # g_par and g_orth must add up to the full gradient vector.
    tot = np.linalg.norm(np.asarray(free["currents_a"]) @ m.G.T)
    close(np.hypot(free["gradient_parallel_actual_hz_per_m"],
                   free["orthogonal_leakage_hz_per_m"]), tot, 1e-6,
          "parallel and orthogonal components are a right-angle decomposition")


# ── 5. bounds with a static baseline ───────────────────────────────
def t_bounds():
    print("5. symmetric bipolar bounds around a static baseline")
    lo, hi = dynamic_bounds(4, 3.0, [2.5, -2.0, 0.0, 3.0], symmetric=True)
    ok(np.allclose(hi, [0.5, 1.0, 3.0, 0.0]), "upper = Imax - |i_static|")
    ok(np.allclose(lo, -np.asarray(hi)), "and the window is symmetric")
    ok(np.all(np.abs(np.asarray([2.5, -2.0, 0.0, 3.0]) + hi) <= 3.0 + 1e-12)
       and np.all(np.abs(np.asarray([2.5, -2.0, 0.0, 3.0]) + lo) <= 3.0 + 1e-12),
       "both i_static+i and i_static-i stay inside +-3 A")

    alo, ahi = dynamic_bounds(2, 3.0, [2.5, -2.0], symmetric=False)
    ok(np.allclose(ahi, [0.5, 5.0]) and np.allclose(alo, [-5.5, -1.0]),
       "the asymmetric window is the raw one, for a single-sign blip")

    try:
        dynamic_bounds(2, 3.0, [3.5, 0.0])
        ok(False, "an out-of-bounds baseline should be rejected")
    except ValueError:
        ok(True, "an out-of-bounds baseline is rejected")

    m = model_of([lambda x, y, z: 300.0 * x, lambda x, y, z: 200.0 * y])
    r0 = max_gradient_socp(m, [1, 0, 0], 0.30, 3.0)
    r1 = max_gradient_socp(m, [1, 0, 0], 0.30, 3.0, i_static=[1.5, 0.0])
    close(r1["gradient_parallel_actual_hz_per_m"]
          / r0["gradient_parallel_actual_hz_per_m"], 0.5, 1e-4,
          "a 1.5 A baseline halves the reachable beta on the binding channel")
    ok(np.all(np.abs(np.asarray(r1["currents_a"]) + np.array([1.5, 0.0]))
              <= 3.0 + 1e-9),
       "i_static + i is inside the limit")
    ok(np.all(np.abs(-np.asarray(r1["currents_a"]) + np.array([1.5, 0.0]))
              <= 3.0 + 1e-9),
       "and so is i_static - i")


# ── 6. per-sample check when channels carry different phases ───────
def t_per_sample():
    print("6. channels with different phases need the per-sample check")
    # The simplified bound assumes one shared scalar waveform. Give each
    # channel its own phase and the two disagree, so the per-sample form is
    # the only correct one.
    t = np.linspace(0.0, 0.01, 2001)
    f = 200.0
    a = np.array([1.0, 1.0])
    base = np.array([1.0, -1.0])
    phases = [0.0, np.pi / 2]
    dyn = np.stack([a[j] * np.sin(2 * np.pi * f * t + phases[j])
                    for j in range(2)], axis=1)

    # The shared-waveform shortcut would allow this scale.
    s_shortcut = float(np.min((3.0 - np.abs(base)) / np.abs(a)))
    total = base + s_shortcut * dyn
    rep = check_current_bounds(t, total, 3.0, base)
    ok(rep["ok"],
       "for these phases the shortcut happens to stay legal (peak %.3f A)"
       % rep["global_peak_abs_a"])

    # Now make the shortcut wrong: channels that share a limit but peak at
    # different times let a naive per-channel bound pass a combination the
    # per-sample check would catch if the limit were shared. Verify instead
    # that the per-sample check is what actually decides, by pushing past it.
    over = base + (s_shortcut * 1.05) * dyn
    rep2 = check_current_bounds(t, over, 3.0, base)
    ok(not rep2["ok"], "5%% beyond the shortcut is caught per sample")
    ok(rep2["first_violation_time_s"] is not None,
       "and the first offending sample is timed (t=%.5g s)"
       % rep2["first_violation_time_s"])
    ok(rep2["clipping_applied"] is False, "still no clipping")


# ── 7. the frontier helper reports every required field ────────────
def t_frontier_fields():
    print("7. every operating point carries the required fields")
    m = model_of([lambda x, y, z: 200.0 * x + 400.0 * y,
                  lambda x, y, z: 150.0 * x + 90.0 * z,
                  lambda x, y, z: 300.0 * (x ** 2 - y ** 2)])
    rows = pareto_frontier(m, [1, 0, 0], epsilons=(0.2, 0.5))
    need = ("gradient_parallel_actual_hz_per_m",
            "gradient_parallel_actual_mt_per_m",
            "gradient_magnitude_actual_hz_per_m", "direction_error_deg",
            "orthogonal_leakage_ratio", "nonlinear_residual_ratio",
            "nonlinear_residual_rms_hz", "target_fit_nrmse", "r_squared",
            "nonlinear_p95_hz", "currents_a", "current_peak_a", "status",
            "clipping_applied")
    for r in rows:
        if not r.get("feasible"):
            continue
        missing = [k for k in need if k not in r]
        ok(not missing, "%s: all primary fields present"
           % r.get("method", r.get("epsilon_nonlinear")))
        if missing:
            print("       missing: %s" % missing)
    ok(rows[0]["method"] == "beta_min_norm_ray_limit",
       "the ray baseline is the first row and is named for what it is")


if __name__ == "__main__":
    for fn in (t_conversion, t_exact, t_adversarial, t_monotone,
               t_certificate, t_cone, t_bounds, t_per_sample,
               t_frontier_fields):
        fn()
    print()
    if FAILS:
        print("%d FAILED" % len(FAILS))
        for f in FAILS:
            print("  -", f)
        sys.exit(1)
    print("all passed")
