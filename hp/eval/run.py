"""The experiment: replay each workload under each policy, against the oracle.

Usage::

    python -m hp.eval.run measure  <workload>          # slow, cached
    python -m hp.eval.run policies <workload> [names]  # fast, needs the cache
    python -m hp.eval.run validate <workload> [names]  # replay vs real run
    python -m hp.eval.run report                       # collate everything

The measurement pass is separated from the policy pass and cached on disk
because the measurement is the expensive half -- it replays the whole trace on
the backtracking tier -- and because it must be done exactly once per workload.
Re-measuring between policies would let machine noise walk between the runs
being compared, and the comparison is the entire point.

Each policy is executed for real: the patterns are genuinely compiled, the
generated code genuinely runs. Reported numbers are wall-clock nanoseconds split
three ways -- matching, building, and the policy's own decision overhead -- so
that a policy cannot look good by moving its cost somewhere the total does not
see.
"""

import json
import os
import sys
import time

from .. import engine as E
from .. import policy as policy_mod
from . import (measure as measure_mod, oracle as oracle_mod,
               simulate as sim_mod, workload as wl)

RESULTS = os.path.join(os.path.dirname(__file__), "..", "..", "results")
STEP_LIMIT = 50000

POLICIES = [
    ("never", lambda: policy_mod.Never()),
    ("always-dfa", lambda: policy_mod.Fixed(E.TIER_DFA)),
    ("always-jit", lambda: policy_mod.Fixed(E.TIER_JIT)),
    ("calls(2,4)", lambda: policy_mod.CallCount(2, 4)),
    ("calls(10,50)", lambda: policy_mod.CallCount(10, 50)),
    ("calls(100,500)", lambda: policy_mod.CallCount(100, 500)),
    ("bytes(4k,64k)", lambda: policy_mod.ByteCount(4 << 10, 64 << 10)),
    ("bytes(64k,1M)", lambda: policy_mod.ByteCount(64 << 10, 1 << 20)),
]
K_VALUES = [0.25, 0.5, 1.0, 2.0, 4.0, 8.0]

CALIBRATION = "calibration.json"


def policy_entries():
    """Every policy under test, with the cost-based ones calibrated.

    The calibrated model is loaded once and shared, so ``amortised`` and
    ``adaptive`` differ only in where their build-cost estimate comes from --
    which is the comparison the experiment exists to make.
    """
    from . import calibrate as cal
    model, _ = cal.load(_path(CALIBRATION))
    entries = list(POLICIES)
    for k in K_VALUES:
        entries.append(("amortised(K=%g)" % k,
                        (lambda k: lambda: policy_mod.Amortised(k, model))(k)))
    for k in K_VALUES:
        entries.append(("adaptive(K=%g)" % k,
                        (lambda k: lambda: policy_mod.Adaptive(k, model))(k)))
    for k in (0.25, 0.5, 1.0):
        entries.append(("guarded/adaptive(K=%g)" % k,
                        (lambda k: lambda: policy_mod.Guarded(
                            policy_mod.Adaptive(k, model, trip=True),
                            model))(k)))
    for k in K_VALUES:
        entries.append(("adaptive+trip(K=%g)" % k,
                        (lambda k: lambda: policy_mod.Adaptive(
                            k, model, trip=True))(k)))
    return entries


def _path(name):
    if not os.path.isdir(RESULTS):
        os.makedirs(RESULTS)
    return os.path.join(RESULTS, name)


# -- pass 1: measurement -----------------------------------------------------
def do_measure(name, repeats=1):
    trace = wl.WORKLOADS[name]()
    t0 = time.time()
    costs = measure_mod.measure(trace, step_limit=STEP_LIMIT, repeats=repeats)
    elapsed = time.time() - t0

    payload = {
        "workload": name,
        "shape": wl.describe(trace),
        "step_limit": STEP_LIMIT,
        "measure_seconds": elapsed,
        "summary": measure_mod.summarise(costs),
        "patterns": {
            pat: {
                "lengths": c.lengths,
                "steps": c.steps,
                "ns": {E.TIER_NAMES[t]: c.ns[t] for t in c.ns},
                "available": [E.TIER_NAMES[t] for t in c.available_tiers],
                "build_ns": {E.TIER_NAMES[t]: v
                             for t, v in c.build_ns.items()},
                "censored_calls": c.censored_calls,
                "ast": [c.ast_nodes, c.ast_classes, c.nfa_states,
                        c.repeat_span, c.risky],
                "states": c.n_states,
                "edges": c.n_edges,
                "code_bytes": c.code_size,
                "note": c.note,
            } for pat, c in costs.items()},
    }
    with open(_path("measured-%s.json" % name), "w") as fh:
        json.dump(payload, fh)
    print("measured %s in %.1fs: %s" % (name, elapsed, payload["shape"]))
    print("  totals:", json.dumps(payload["summary"], indent=None))
    return payload


def load_measured(name):
    with open(_path("measured-%s.json" % name)) as fh:
        payload = json.load(fh)
    costs = {}
    for pat, d in payload["patterns"].items():
        c = measure_mod.TierCosts(pat)
        c.lengths = d["lengths"]
        c.steps = d["steps"]
        by_name = {v: k for k, v in E.TIER_NAMES.items()}
        c.ns = {by_name[k]: v for k, v in d["ns"].items()}
        c.available_tiers = [by_name[t] for t in d["available"]]
        c.build_ns = {by_name[k]: v for k, v in d["build_ns"].items()}
        c.censored_calls = d["censored_calls"]
        c.censored = sum(1 for x in c.censored_calls if x)
        (c.ast_nodes, c.ast_classes, c.nfa_states,
         c.repeat_span, c.risky) = d["ast"]
        c.n_states = d["states"]
        c.n_edges = d["edges"]
        c.code_size = d["code_bytes"]
        c.note = d["note"]
        costs[pat] = c
    return payload, costs


# -- pass 2: policies --------------------------------------------------------
def run_policy(trace, policy):
    eng = E.Engine(policy=policy, step_limit=STEP_LIMIT)
    for pat, subj in trace:
        eng.matches(pat, subj)
    totals = eng.totals()
    tiers = {}
    for p in eng.patterns.values():
        tiers[E.TIER_NAMES[p.tier]] = tiers.get(E.TIER_NAMES[p.tier], 0) + 1
    totals["final_tiers"] = tiers
    eng.free()
    return totals


def do_policies(name, only=None):
    payload, costs = load_measured(name)
    oracle = oracle_mod.evaluate(costs)
    trace = wl.WORKLOADS[name]()

    entries = policy_entries()
    if only:
        entries = [e for e in entries if e[0] in only]

    rows = []
    for label, make in entries:
        totals = sim_mod.run(trace, costs, make())
        rows.append({
            "policy": label,
            "engine_ns": totals["engine_ns"],
            "match_ns": totals["match_ns"],
            "build_ns": totals["build_ns"],
            "promotions": totals["promotions"],
            "forced": totals["forced"],
            "final_tiers": totals["final_tiers"],
            "recovered": oracle_mod.recovered(totals["engine_ns"],
                                              oracle["never_ns"],
                                              oracle["oracle_ns"]),
            # Always defined, and the number that travels between workloads:
            # how many times the optimum this policy cost.
            "vs_oracle": totals["engine_ns"] / oracle["oracle_ns"],
        })
        print("  %-20s %10.1f ms  %5.2fx oracle   recovered %s"
              % (label, totals["engine_ns"] / 1e6, rows[-1]["vs_oracle"],
                 "n/a" if rows[-1]["recovered"] is None
                 else "%6.1f%%" % (100 * rows[-1]["recovered"])))

    out = {
        "workload": name,
        "shape": payload["shape"],
        "oracle": {k: v for k, v in oracle.items() if k != "schedules"},
        "schedules": oracle["schedules"],
        "policies": rows,
    }
    path = _path("policies-%s.json" % name)
    if only and os.path.exists(path):
        with open(path) as fh:
            prev = json.load(fh)
        keep = [r for r in prev["policies"]
                if r["policy"] not in {r2["policy"] for r2 in rows}]
        out["policies"] = keep + rows
    with open(path, "w") as fh:
        json.dump(out, fh, indent=1)
    return out



# -- pass 2b: is the replay telling the truth? -------------------------------
def do_validate(name, only=None):
    """Run policies for real and compare with what the replay predicted.

    The replay is a shortcut, and a shortcut has to be audited. This executes
    the same policies against the same trace with the real engine -- real subset
    constructions, real mmap, real generated code -- and reports the error. If
    the two disagree by more than measurement noise, the replay is wrong and so
    is every number derived from it.
    """
    payload, costs = load_measured(name)
    trace = wl.WORKLOADS[name]()
    entries = policy_entries()
    if only:
        entries = [e for e in entries if e[0] in only]

    rows = []
    for label, make in entries:
        predicted = sim_mod.run(trace, costs, make())["engine_ns"]
        actual = run_policy(trace, make())["engine_ns"]
        err = (actual - predicted) / predicted if predicted else 0.0
        rows.append({"policy": label, "predicted_ns": predicted,
                     "actual_ns": actual, "error": err})
        print("  %-20s predicted %8.1f ms   actual %8.1f ms   %+6.1f%%"
              % (label, predicted / 1e6, actual / 1e6, 100 * err))
    worst = max(abs(r["error"]) for r in rows)
    print("  worst absolute error: %.1f%%" % (100 * worst))
    out = {"workload": name, "rows": rows, "worst_abs_error": worst}
    with open(_path("validation-%s.json" % name), "w") as fh:
        json.dump(out, fh, indent=1)
    return out


# -- pass 3: the table -------------------------------------------------------
def do_report():
    names = [n for n in wl.WORKLOADS
             if os.path.exists(_path("policies-%s.json" % n))]
    tables = {}
    for n in names:
        with open(_path("policies-%s.json" % n)) as fh:
            tables[n] = json.load(fh)

    labels = []
    for t in tables.values():
        for r in t["policies"]:
            if r["policy"] not in labels:
                labels.append(r["policy"])

    def table(title, cell):
        out = ["### %s" % title, "",
               "| policy | " + " | ".join(names) + " |",
               "|---|" + "---|" * len(names)]
        for label in labels:
            cells = []
            for n in names:
                row = next((r for r in tables[n]["policies"]
                            if r["policy"] == label), None)
                cells.append("-" if row is None else cell(row))
            out.append("| `%s` | %s |" % (label, " | ".join(cells)))
        return "\n".join(out)

    text = "\n\n".join([
        table("Cost as a multiple of the hindsight optimum (lower is better)",
              lambda r: "%.2f" % r["vs_oracle"]),
        table("Share of the available saving recovered (higher is better)",
              lambda r: ("n/a" if r["recovered"] is None
                         else "%.0f%%" % (100 * r["recovered"]))),
    ])
    with open(_path("recovery.md"), "w") as fh:
        fh.write(text + "\n")
    print(text)
    return text


def main(argv):
    if len(argv) < 2:
        print(__doc__)
        return 2
    cmd = argv[1]
    if cmd == "measure":
        do_measure(argv[2], repeats=int(argv[3]) if len(argv) > 3 else 1)
    elif cmd == "policies":
        do_policies(argv[2], only=argv[3:] or None)
    elif cmd == "validate":
        do_validate(argv[2], only=argv[3:] or None)
    elif cmd == "report":
        do_report()
    else:
        print(__doc__)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
