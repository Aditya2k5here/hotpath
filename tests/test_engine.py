"""Tiers, promotion, and the properties the experiment depends on.

The central invariant is interchangeability: the engine may change tier between
any two calls, and no caller may be able to tell. Everything else here defends a
specific claim made in the results -- that the ceiling drops instead of raising,
that the step budget cannot hang the engine, that the oracle really is optimal,
and that replaying a policy over recorded costs gives the same answer as running
it for real.
"""

import random
import re

import pytest

from hp import engine as E, policy as PL
from hp.eval import oracle as O, simulate as S
from hp.jit import memory

PATTERNS = ["abc", "a+b", "[a-z]+@[a-z]+", r"\d{3}-\d{4}", "(foo|bar)+",
            "^GET ", "error$", r"\w+\s\w+"]


@pytest.fixture(scope="module")
def subjects():
    rnd = random.Random(99)
    out = [b"", b"abc", b"xabcx", b"aab", b"me@you", b"555-1234", b"GET /x",
           b"an error", b"foo bar", b"hello world"]
    for _ in range(200):
        out.append(bytes(ord(rnd.choice("abcdefgo 0123@-"))
                         for _ in range(rnd.randrange(0, 12))))
    return out


# -- interchangeability ------------------------------------------------------
@pytest.mark.parametrize("pattern", PATTERNS)
def test_all_tiers_answer_identically(pattern, subjects):
    ref = re.compile(pattern.replace("$", r"\Z").encode())
    pats = []
    for tier in (E.TIER_BACKTRACK, E.TIER_DFA, E.TIER_JIT):
        p = E.Pattern(pattern, step_limit=1_000_000)
        p.promote(tier)
        pats.append(p)
    try:
        for s in subjects:
            want = ref.search(s) is not None
            for p in pats:
                assert p.matches(s) is want, (pattern, s, p.tier)
    finally:
        for p in pats:
            p.free()


@pytest.mark.parametrize("pattern", PATTERNS)
def test_promotion_mid_stream_changes_nothing(pattern, subjects):
    """Promote between calls and the answers must not move."""
    ref = re.compile(pattern.replace("$", r"\Z").encode())
    p = E.Pattern(pattern, step_limit=1_000_000)
    try:
        for i, s in enumerate(subjects):
            if i == 5:
                p.promote(E.TIER_DFA)
            if i == 20:
                p.promote(E.TIER_JIT)
            assert p.matches(s) is (ref.search(s) is not None), (pattern, s)
    finally:
        p.free()


# -- promotion mechanics -----------------------------------------------------
def test_promotion_is_monotone_and_idempotent():
    p = E.Pattern("a+b")
    try:
        assert p.tier == E.TIER_BACKTRACK
        assert p.promote(E.TIER_DFA) is not None
        assert p.tier == E.TIER_DFA
        assert p.promote(E.TIER_DFA) is None       # nothing left to do
        assert p.promote(E.TIER_BACKTRACK) is None  # never demotes
        assert p.tier == E.TIER_DFA
    finally:
        p.free()


def test_an_exponential_pattern_lowers_the_ceiling_instead_of_raising():
    p = E.Pattern(r"[ab]*a[ab]{14}", max_states=64)
    assert p.promote(E.TIER_JIT) is None
    assert p.ceiling == E.TIER_BACKTRACK
    assert p.dfa is None
    assert "exceeded" in p.reason
    assert p.matches(b"abab") is False             # still answers


def test_the_step_budget_forces_a_promotion_rather_than_hanging():
    p = E.Pattern("^(a+)+$", step_limit=5000)
    try:
        assert p.matches(b"a" * 40 + b"!") is False
        assert p.forced == 1
        assert p.tier >= E.TIER_DFA
    finally:
        p.free()


def test_shape_and_risk_are_available_before_any_build():
    p = E.Pattern("^(a+)+$")
    assert p.risky is True
    assert p.nfa_states > 0
    assert p.n_states is None                      # nothing built yet
    p = E.Pattern(r"\d+\.\d+")
    assert p.risky is False


@pytest.mark.skipif(not memory.available(), reason="no executable memory")
def test_engine_frees_every_program():
    eng = E.Engine(policy=PL.Fixed(E.TIER_JIT))
    for pat in PATTERNS:
        eng.matches(pat, b"abc")
    programs = [p.program for p in eng.patterns.values() if p.program]
    assert programs
    eng.free()
    assert all(p.block.freed for p in programs)


# -- policies ----------------------------------------------------------------
def test_call_count_promotes_on_schedule():
    eng = E.Engine(policy=PL.CallCount(3, 6))
    tiers = []
    for _ in range(8):
        eng.matches("a+b", b"aab")
        tiers.append(eng.pattern("a+b").tier)
    eng.free()
    assert tiers[0] == E.TIER_BACKTRACK
    assert tiers[3] == E.TIER_DFA
    assert tiers[-1] == E.TIER_JIT


def test_never_stays_put_when_nothing_forces_it():
    eng = E.Engine(policy=PL.Never())
    for _ in range(50):
        eng.matches("abc", b"abc")
    assert eng.pattern("abc").tier == E.TIER_BACKTRACK
    eng.free()


def test_guarded_refuses_tier_zero_for_a_risky_pattern():
    guard = PL.Guarded(PL.Never(), PL.CostModel())
    risky = E.Pattern("^(a+)+$")
    safe = E.Pattern(r"\d+\.\d+")
    assert guard.target(risky, b"aaaa") >= E.TIER_DFA
    assert guard.target(safe, b"1.2") == E.TIER_BACKTRACK


def test_guarded_skips_tier_zero_for_a_subject_too_large_to_learn_from():
    guard = PL.Guarded(PL.Never(), PL.CostModel())
    p = E.Pattern(r"\d+\.\d+")
    assert guard.target(p, b"1.2") == E.TIER_BACKTRACK
    assert guard.target(p, b"x" * 4_000_000) >= E.TIER_DFA


def test_adaptive_learns_the_build_cost_it_is_shown():
    model = PL.CostModel()
    a = PL.Adaptive(1.0, model)
    p = E.Pattern("a+b")
    before = a._build_cost(p)
    for _ in range(20):
        a.observe_build(p, E.TIER_DFA, 3_000_000.0)
    after = a._build_cost(p)
    assert after != before
    assert abs(after - 3_000_000.0) < 100_000.0


def test_cost_model_is_monotone_in_the_things_it_charges_for():
    m = PL.CostModel()
    assert m.match_cost(E.TIER_DFA, 1000, 1) > m.match_cost(E.TIER_DFA, 10, 1)
    assert m.match_cost(E.TIER_JIT, 1000, 1) < m.match_cost(E.TIER_DFA, 1000, 1)
    small = E.Pattern("ab")
    big = E.Pattern(r"[A-Z]{2}\d{2}[A-Z0-9]{4,20}")
    assert m.dfa_build_cost(big) > m.dfa_build_cost(small)


# -- the oracle --------------------------------------------------------------
def _brute_force(c0, c1, c2, bd, bj):
    """Every (i, j) schedule, checked exhaustively."""
    n = len(c0)
    best = None
    for i in range(n + 1):
        for j in range(i, n + 1):
            cost = sum(c0[:i]) + sum(c1[i:j]) + sum(c2[j:])
            if i < n or j < n:
                cost += bd
            if j < n:
                cost += bj
            best = cost if best is None else min(best, cost)
    return best


@pytest.mark.parametrize("seed", range(40))
def test_the_oracle_is_actually_optimal(seed):
    """The linear-time optimum must equal exhaustive search, or it is not one."""
    rnd = random.Random(seed)
    n = rnd.randrange(1, 14)
    c0 = [rnd.uniform(1, 5000) for _ in range(n)]
    c1 = [rnd.uniform(1, 500) for _ in range(n)]
    c2 = [rnd.uniform(1, 50) for _ in range(n)]
    bd = rnd.uniform(0, 20000)
    bj = rnd.uniform(0, 20000)
    got, _, _ = O.best_schedule(c0, c1, c2, bd, bj)
    assert abs(got - _brute_force(c0, c1, c2, bd, bj)) < 1e-6


def test_no_policy_can_beat_the_oracle():
    """The property that makes the recovered fraction meaningful."""
    trace, costs = _tiny_workload()
    best = O.evaluate(costs)["oracle_ns"]
    model = PL.CostModel()
    for policy in (PL.Never(), PL.Fixed(E.TIER_DFA), PL.Fixed(E.TIER_JIT),
                   PL.CallCount(2, 4), PL.ByteCount(100, 1000),
                   PL.Amortised(0.5, model), PL.Adaptive(0.5, model),
                   PL.Guarded(PL.Adaptive(0.5, model), model)):
        got = S.run(trace, costs, policy)["engine_ns"]
        assert got >= best - 1e-6, policy.name


def test_recovered_is_undefined_when_nothing_was_available():
    assert O.recovered(10.0, 10.0, 10.0) is None
    assert O.recovered(5.0, 10.0, 0.0) == 0.5


def _tiny_workload():
    """A hand-built measurement set, so the test needs no timing."""
    from hp.eval import measure as M
    trace = []
    costs = {}
    rnd = random.Random(3)
    for pat, n in (("a+b", 12), ("[a-z]+", 5), ("abc", 1)):
        c = M.TierCosts(pat)
        c.available_tiers = [E.TIER_BACKTRACK, E.TIER_DFA, E.TIER_JIT]
        c.build_ns = {E.TIER_DFA: 500000, E.TIER_JIT: 200000}
        c.n_states, c.n_edges = 6, 20
        c.ast_nodes, c.ast_classes, c.nfa_states, c.repeat_span = 5, 2, 30, 0
        c.risky = False
        for _ in range(n):
            length = rnd.randrange(10, 200)
            c.lengths.append(length)
            c.steps.append(length * 4)
            c.censored_calls.append(False)
            c.ns[E.TIER_BACKTRACK].append(length * 3000.0)
            c.ns[E.TIER_DFA].append(200.0 + length * 30.0)
            c.ns[E.TIER_JIT].append(600.0 + length * 0.8)
            trace.append((pat, b"x" * length))
        costs[pat] = c
    rnd.shuffle(trace)
    return trace, costs


# -- the replay ---------------------------------------------------------------
@pytest.mark.skipif(not memory.available(), reason="no executable memory")
@pytest.mark.parametrize("policy_name", ["never", "always-jit", "calls-2"])
def test_replay_and_real_run_take_the_same_decisions(policy_name):
    """Timings differ between passes; the *decisions* must not.

    This is the check that licenses evaluating policies by replay. It compares
    tiers and promotion counts rather than nanoseconds, because nanoseconds are
    exactly the thing that legitimately varies.
    """
    from hp.eval import measure as M
    trace = [(pat, b"x" * (7 * i % 90 + 3))
             for i in range(300) for pat in ("a+b", "[a-z]+@[a-z]+")]
    costs = M.measure(trace, step_limit=200000)

    replayed = S.run(trace, costs, PL.ALL[policy_name]())
    eng = E.Engine(policy=PL.ALL[policy_name](), step_limit=200000)
    for pat, subj in trace:
        eng.matches(pat, subj)
    real = eng.totals()
    real_tiers = {}
    for p in eng.patterns.values():
        key = E.TIER_NAMES[p.tier]
        real_tiers[key] = real_tiers.get(key, 0) + 1
    eng.free()

    assert replayed["final_tiers"] == real_tiers
    assert replayed["promotions"] == real["promotions"]
