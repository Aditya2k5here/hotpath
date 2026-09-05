"""Fit the cost model to the machine it is going to run on.

:class:`hp.policy.Amortised` reasons about nanoseconds: how much a backtracking
step costs, how much a byte costs at each tier, how much a subset construction
costs. Those are properties of a machine, not of an algorithm, and a policy
carrying someone else's constants is a policy making someone else's decisions.
The default constants shipped in :class:`hp.policy.CostModel` are deliberately a
little wrong, and the evaluation reports what that costs, so calibration is
not a setup step to be skipped, it is one of the variables under test.

Everything here is ordinary least squares on measurements taken seconds earlier,
solved by Gaussian elimination on the normal equations. No dependencies, and the
residuals are reported so a fit that does not describe the machine says so.
"""

import json
import math
import os
import time

from .. import policy as policy_mod
from ..jit import codegen, memory, program as prog
from ..re import backtrack, dfa as D, nfa as N, parser as P

CAL_PATTERNS = [
    r"[a-z]+@[a-z]+\.[a-z]+", r"\d+\.\d+\.\d+\.\d+", "error|warning|fatal",
    "(a|b)*abb", r"^GET /api/v\d+/\w+", r"[A-Z]{2}\d{2}[A-Z0-9]{4,20}",
    r"\d{4}-\d{2}-\d{2}", "(foo|bar|baz)+", r"password\s*[:=]\s*\S+",
    r"[0-9a-f]{8}-[0-9a-f]{4}", r"\w+;\w+;\w+", r"<[a-z]+ [a-z]+=\"[^\"]*\">",
]
# Build cost does not depend on the subject, so this list is free to include
# patterns that would match the calibration text, and it needs to, because the
# spread of counted-repetition spans is what the build-cost model turns on.
BUILD_PATTERNS = CAL_PATTERNS + [
    r"[a-z]{1,12}", r"(ab|cd|ef){2,8}", r"x{3}y{3}", r"\w{5,9}@\w{2,6}",
    r"a?b?c?d?e?f?", r"[0-9]{1,6}-[0-9]{1,6}", r"(foo)?(bar)?(baz)?",
    r"[a-f]{2,10}:[a-f]{2,10}", r"(\d\d){1,9}",
]

SIZES = [16, 64, 256, 1024, 4096, 16384]
# Calibration subjects must match *nothing*: a pattern that matches early
# returns early, and its cost stops depending on the subject length. Fitting a
# per-byte cost against a mixture of early exits and full scans produces a slope
# that describes neither. The conservative quantity a policy wants is the cost
# of a scan that runs to the end, so that is what is measured.
FILLER = b"qqq wwww qqq wwww qqq wwww "


def _solve(rows, targets, weights=None):
    """Weighted least squares by Gaussian elimination on the normal equations."""
    k = len(rows[0])
    w = weights if weights is not None else [1.0] * len(rows)
    ata = [[sum(wi * r[i] * r[j] for wi, r in zip(w, rows)) for j in range(k)]
           for i in range(k)]
    atb = [sum(wi * r[i] * t for wi, r, t in zip(w, rows, targets))
           for i in range(k)]
    for i in range(k):
        pivot = max(range(i, k), key=lambda p: abs(ata[p][i]))
        ata[i], ata[pivot] = ata[pivot], ata[i]
        atb[i], atb[pivot] = atb[pivot], atb[i]
        if abs(ata[i][i]) < 1e-12:                      # pragma: no cover
            continue
        for r in range(i + 1, k):
            f = ata[r][i] / ata[i][i]
            for c in range(i, k):
                ata[r][c] -= f * ata[i][c]
            atb[r] -= f * atb[i]
    out = [0.0] * k
    for i in range(k - 1, -1, -1):
        if abs(ata[i][i]) < 1e-12:                      # pragma: no cover
            continue
        out[i] = (atb[i] - sum(ata[i][j] * out[j]
                               for j in range(i + 1, k))) / ata[i][i]
    return out


def fit_relative(rows, targets):
    """Fit minimising *relative* error, not absolute.

    These quantities span four orders of magnitude, a backtracking call can
    cost 3 microseconds or 30 milliseconds, and ordinary least squares spends
    all its effort on the largest points, producing a model that is 1000% wrong
    everywhere the policy actually operates. Weighting each residual by
    ``1/y**2`` and iterating targets the error the policy cares about: being
    right to within a factor, at every scale.
    """
    active = list(range(len(rows[0])))
    while True:
        sub = [[r[i] for i in active] for r in rows]
        coef = _solve(sub, targets)
        for _ in range(6):
            pred = [max(sum(c * x for c, x in zip(coef, r)), 1.0) for r in sub]
            coef = _solve(sub, targets, [1.0 / (q * q) for q in pred])
        worst = min(range(len(coef)), key=lambda i: coef[i])
        # A negative coefficient means "this feature makes the prediction
        # smaller", which for a cost model is not a fit, it is an artefact.
# and it produces negative predicted costs on inputs outside the
        # calibration set. Drop the offender and refit rather than shipping it.
        if coef[worst] >= 0 or len(active) == 1:
            break
        active.pop(worst)
    out = [0.0] * len(rows[0])
    for i, a in enumerate(active):
        out[a] = coef[i]
    return out


def _rel_error(rows, targets, coef):
    errs = []
    for r, t in zip(rows, targets):
        pred = sum(c * x for c, x in zip(coef, r))
        if t:
            errs.append(abs(pred - t) / t)
    return sum(errs) / len(errs) if errs else 0.0


def _subject(n):
    return (FILLER * (n // len(FILLER) + 2))[:n]


def _best_of(fn, arg, tries=5):
    best = None
    for _ in range(tries):
        t0 = time.perf_counter_ns()
        fn(arg)
        took = time.perf_counter_ns() - t0
        best = took if best is None else min(best, took)
    return best


def calibrate(step_limit=50000, verbose=True):
    subjects = {n: _subject(n) for n in SIZES}
    dfas = {}
    programs = {}

    bt_rows, bt_y = [], []
    dfa_rows, dfa_y = [], []
    jit_rows, jit_y = [], []
    build_rows, build_y = [], []
    code_rows, code_y = [], []
    jbuild_rows, jbuild_y = [], []

    jit_ok = memory.available()

    for pat in BUILD_PATTERNS:
        node = P.unanchored(P.parse(pat))
        best = None
        for _ in range(3):
            t0 = time.perf_counter_ns()
            d = D.minimise(D.build(N.build(node)))
            took = time.perf_counter_ns() - t0
            best = took if best is None else min(best, took)
        dfas[pat] = d
        _, classes, span = P.shape(P.parse(pat))
        build_rows.append([1.0, math.log(N.size(node)),
                           math.log(classes + 1), math.log(span + 1)])
        build_y.append(float(best))

    for pat in CAL_PATTERNS:
        node = P.unanchored(P.parse(pat))
        d = dfas[pat]
        code = codegen.compile_to_bytes(d, 0)
        code_rows.append([1.0, float(len(d)), float(d.n_edges())])
        code_y.append(float(len(code)))

        if jit_ok:
            t0 = time.perf_counter_ns()
            p = prog.compile_dfa(d)
            jbuild_rows.append([1.0, float(len(code))])
            jbuild_y.append(float(time.perf_counter_ns() - t0))
            programs[pat] = p

        m = backtrack.Matcher(node, limit=step_limit)
        for n in SIZES:
            s = subjects[n]
            t0 = time.perf_counter_ns()
            try:
                m.fullmatch(s)
            except backtrack.Exhausted:
                pass
            took = time.perf_counter_ns() - t0
            bt_rows.append([float(m.steps)])
            bt_y.append(float(took))

            dfa_rows.append([1.0, float(n)])
            dfa_y.append(float(_best_of(d.fullmatch, s)))
            if jit_ok:
                jit_rows.append([1.0, float(n)])
                jit_y.append(float(_best_of(programs[pat], s)))

    for p in programs.values():
        p.free()

    bt = fit_relative(bt_rows, bt_y)
    dfa = fit_relative(dfa_rows, dfa_y)
    jit = fit_relative(jit_rows, jit_y) if jit_ok else [1800.0, 1.5]
    # Fitted in log space: the model for this one is a power law, so the fit is
    # ordinary least squares on the logarithms rather than a weighted fit on the
    # values. See CostModel.dfa_build_cost for why.
    bld = _solve(build_rows, [math.log(v) for v in build_y])
    cod = fit_relative(code_rows, code_y)
    jbl = fit_relative(jbuild_rows, jbuild_y) if jit_ok else [90000.0, 25.0]

    fit = {
        "bt_ns_per_step": bt[0],
        "dfa_ns_fixed": dfa[0], "dfa_ns_per_byte": dfa[1],
        "jit_ns_fixed": jit[0], "jit_ns_per_byte": jit[1],
        "dfa_build_log_const": bld[0], "dfa_build_exp_nfa": bld[1],
        "dfa_build_exp_class": bld[2], "dfa_build_exp_span": bld[3],
        "code_bytes_fixed": cod[0], "code_bytes_per_state": cod[1],
        "code_bytes_per_edge": cod[2],
        "jit_build_ns_fixed": jbl[0], "jit_build_ns_per_code_byte": jbl[1],
    }
    # Is the shape worth anything, or would a constant have done as well? The
    # answer is reported rather than assumed, because a feature that does not
    # beat a constant is a feature that should not be in the model.
    def log_error(coef):
        errs = []
        for r, t in zip(build_rows, build_y):
            pred = math.exp(sum(c * x for c, x in zip(coef, r)))
            errs.append(abs(pred - t) / t)
        return sum(errs) / len(errs)

    flat = [sum(math.log(v) for v in build_y) / len(build_y), 0.0, 0.0, 0.0]
    residuals = {
        # Would a constant have done as well? Reported rather than assumed: a
        # feature that does not beat a constant should not be in the model.
        "dfa_build_constant_only": log_error(flat),
        "backtrack_step": _rel_error(bt_rows, bt_y, bt),
        "dfa_call": _rel_error(dfa_rows, dfa_y, dfa),
        "jit_call": _rel_error(jit_rows, jit_y, jit) if jit_ok else None,
        "dfa_build": log_error(bld),
        "code_size": _rel_error(code_rows, code_y, cod),
        "jit_build": _rel_error(jbuild_rows, jbuild_y, jbl) if jit_ok else None,
    }
    if verbose:
        for k in sorted(fit):
            print("  %-28s %12.3f" % (k, fit[k]))
        print("  mean relative error:")
        for k in sorted(residuals):
            v = residuals[k]
            print("    %-24s %s" % (k, "n/a" if v is None else "%.1f%%"
                                    % (100 * v)))
    return fit, residuals


def model_from(fit):
    return policy_mod.CostModel(**fit)


def save(path, fit, residuals):
    with open(path, "w") as fh:
        json.dump({"fit": fit, "mean_relative_error": residuals}, fh, indent=1)


def load(path):
    """The calibrated model, or the shipped defaults if there is no file."""
    if not os.path.exists(path):
        return policy_mod.CostModel(), None
    with open(path) as fh:
        payload = json.load(fh)
    return model_from(payload["fit"]), payload


if __name__ == "__main__":                              # pragma: no cover
    here = os.path.join(os.path.dirname(__file__), "..", "..", "results")
    if not os.path.isdir(here):
        os.makedirs(here)
    fit, res = calibrate()
    save(os.path.join(here, "calibration.json"), fit, res)
