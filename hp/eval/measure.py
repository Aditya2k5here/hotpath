"""Measure what every call in a trace would cost at every tier.

The point of this module is to make the oracle in :mod:`hp.eval.oracle` real
rather than modelled. Instead of assuming a cost function and asking which
policy wins under it, this replays the trace three times -- once pinned to each
tier -- and records the wall-clock nanoseconds each individual call actually
took. The oracle then optimises over *those* numbers, so "the best possible
policy would have taken 4.1 ms" is a statement about this machine, not about a
spreadsheet.

Two honesty notes, both of which push the reported results in the conservative
direction.

**Tier 0 is censored.** A backtracking engine on a large subject does not
finish, so it runs under a step budget and a call that exhausts the budget is
recorded at the budget's cost. Its true cost is higher -- often unboundedly so.
That understates how much promotion was worth, which understates the savings
available, which makes every policy's recovered fraction look *worse* than it
is. A censored count is reported alongside so the reader can see how much of the
trace this affects.

**Build costs are measured on the same machine, in the same process.** The DFA
construction and the code generation are timed for real, including the mmap and
the mprotect. Nothing here is a constant someone typed in.
"""

import time

from .. import engine as E
from ..re import backtrack
from ..jit import memory


class TierCosts(object):
    """Per-call nanoseconds at each tier, for one pattern, in trace order."""

    def __init__(self, source):
        self.source = source
        self.lengths = []
        self.ns = {E.TIER_BACKTRACK: [], E.TIER_DFA: [], E.TIER_JIT: []}
        self.steps = []               # tier-0 work, for the policies
        self.censored = 0
        self.censored_calls = []      # per call, did tier 0 run out of budget
        self.build_ns = {}
        self.available_tiers = [E.TIER_BACKTRACK]
        self.ast_nodes = 0
        self.ast_classes = 0
        self.nfa_states = 0
        self.repeat_span = 0
        self.risky = False
        self.n_states = None
        self.n_edges = None
        self.code_size = None
        self.note = None

    def calls(self):
        return len(self.lengths)


def _time_call(fn, subject):
    t0 = time.perf_counter_ns()
    fn(subject)
    return time.perf_counter_ns() - t0


def measure(trace, step_limit=50000, jit=True, repeats=1):
    """Replay ``trace`` once per tier and return ``{pattern: TierCosts}``.

    ``repeats`` runs each call more than once and keeps the minimum, which
    suppresses scheduler noise on the fast tiers. The default of 1 is right for
    large traces, where the noise averages out across thousands of calls and the
    extra passes cost more than they buy.
    """
    by_pattern = {}
    order = {}
    for pat, subj in trace:
        if pat not in by_pattern:
            by_pattern[pat] = TierCosts(pat)
            order[pat] = len(order)
        by_pattern[pat].lengths.append(len(subj))

    jit_ok = jit and memory.available()

    for pat in sorted(by_pattern, key=lambda p: order[p]):
        costs = by_pattern[pat]
        subjects = [s for p, s in trace if p == pat]

        # -- tier 0 ---------------------------------------------------------
        p0 = E.Pattern(pat, step_limit=step_limit, jit_enabled=False)
        costs.ast_nodes = p0.ast_nodes
        costs.ast_classes = p0.ast_classes
        costs.repeat_span = p0.repeat_span
        costs.risky = p0.risky
        costs.nfa_states = p0.nfa_states
        for s in subjects:
            best = None
            hit_limit = False
            for _ in range(repeats):
                t0 = time.perf_counter_ns()
                try:
                    p0.bt.fullmatch(s)
                except backtrack.Exhausted:
                    hit_limit = True
                took = time.perf_counter_ns() - t0
                best = took if best is None else min(best, took)
            costs.censored_calls.append(hit_limit)
            costs.censored += 1 if hit_limit else 0
            costs.ns[E.TIER_BACKTRACK].append(best)
            costs.steps.append(p0.bt.steps)

        # -- tier 1 ---------------------------------------------------------
        p1 = E.Pattern(pat, jit_enabled=False)
        t0 = time.perf_counter_ns()
        p1.promote(E.TIER_DFA)
        build_dfa = time.perf_counter_ns() - t0
        if p1.dfa is None:
            costs.note = p1.reason
            continue
        costs.build_ns[E.TIER_DFA] = build_dfa
        costs.n_states = len(p1.dfa)
        costs.n_edges = p1.dfa.n_edges()
        costs.available_tiers.append(E.TIER_DFA)
        for s in subjects:
            best = None
            for _ in range(repeats):
                took = _time_call(p1.dfa.fullmatch, s)
                best = took if best is None else min(best, took)
            costs.ns[E.TIER_DFA].append(best)

        # -- tier 2 ---------------------------------------------------------
        if not jit_ok:
            continue
        p2 = E.Pattern(pat)
        p2.dfa = p1.dfa
        p2.tier = E.TIER_DFA
        t0 = time.perf_counter_ns()
        p2.promote(E.TIER_JIT)
        build_jit = time.perf_counter_ns() - t0
        if p2.program is None:
            costs.note = p2.reason
            continue
        costs.build_ns[E.TIER_JIT] = build_jit
        costs.code_size = len(p2.program.code)
        costs.available_tiers.append(E.TIER_JIT)
        for s in subjects:
            best = None
            for _ in range(repeats):
                took = _time_call(p2.program, s)
                best = took if best is None else min(best, took)
            costs.ns[E.TIER_JIT].append(best)
        p2.free()

    return by_pattern


def summarise(costs_by_pattern):
    """Aggregate per-tier totals, for a sanity look before any policy runs."""
    out = {}
    for tier in (E.TIER_BACKTRACK, E.TIER_DFA, E.TIER_JIT):
        total = 0
        calls = 0
        for c in costs_by_pattern.values():
            if tier in c.available_tiers:
                total += sum(c.ns[tier])
                calls += len(c.ns[tier])
        out[E.TIER_NAMES[tier]] = {"total_ns": total, "calls": calls}
    out["censored_calls"] = sum(c.censored for c in costs_by_pattern.values())
    out["build_ns"] = {
        E.TIER_NAMES[t]: sum(c.build_ns.get(t, 0)
                             for c in costs_by_pattern.values())
        for t in (E.TIER_DFA, E.TIER_JIT)}
    return out
