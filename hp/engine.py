"""Three tiers behind one ``matches`` call, and the bookkeeping to choose.

A pattern starts interpreted and may be promoted twice:

===== ============================= ======================= ==================
Tier  Machinery                     Cost to enter           Cost to run
===== ============================= ======================= ==================
0     backtracking over the AST     parse only              unbounded
1     subset-constructed DFA        NFA + powerset + Moore  linear, in Python
2     that DFA as x86-64            code generation + mmap  linear, native
===== ============================= ======================= ==================

Every tier answers exactly one question -- *does this pattern occur in this
subject* -- and the test suite requires all three to agree with Python's ``re``
on every input it is given. Interchangeability is what makes promotion safe: the
engine can change tiers between two calls and no caller can tell, except by
timing.

Promotion is not free and not always right. Tier 1 costs a subset construction
that can blow up exponentially; tier 2 costs code generation and a syscall. A
pattern used twice on twenty-byte inputs should stay where it is. Deciding is
the job of :mod:`hp.policy`; this module's job is to make the decision
*executable* and to record what each one cost.
"""

import time

from .re import backtrack, dfa as D, nfa as N, parser as P

TIER_BACKTRACK = 0
TIER_DFA = 1
TIER_JIT = 2

TIER_NAMES = {TIER_BACKTRACK: "backtrack", TIER_DFA: "dfa", TIER_JIT: "jit"}


class Unavailable(Exception):
    """A tier could not be built for this pattern, permanently.

    Distinct from "not worth building": the engine records it so the policy
    stops proposing a promotion that will never succeed.
    """


class Pattern(object):
    """One pattern, its tiers, and the accounting for both.

    ``ceiling`` is the highest tier this pattern can reach. It drops when a
    promotion turns out to be impossible -- an exponential subset construction,
    or an alphabet too wide for the code generator -- so the engine degrades to
    a slower tier that works rather than failing.
    """

    def __init__(self, source, max_states=5000, step_limit=None,
                 jit_enabled=True):
        self.source = source
        self.max_states = max_states
        self.ast = P.parse(source)
        self.search_ast = P.unanchored(self.ast)
        self.ast_nodes, self.ast_classes, self.repeat_span = P.shape(self.ast)
        self.nfa_states = N.size(self.search_ast)
        self.risky = P.risky(self.ast)
        self.tier = TIER_BACKTRACK
        self.ceiling = TIER_JIT if jit_enabled else TIER_DFA

        self.bt = backtrack.Matcher(self.search_ast, limit=step_limit)
        self.dfa = None
        self.program = None

        # accounting
        self.calls = 0
        self.bytes_seen = 0
        self.steps = 0                 # tier-0 work actually done
        self.forced = 0                # promotions the step budget compelled
        self.build_ns = {}             # tier -> nanoseconds spent entering it
        self.n_states = None
        self.n_edges = None
        self.code_size = None
        self.reason = None             # why the ceiling dropped, if it did

    # -- matching ------------------------------------------------------------
    def matches(self, subject):
        self.calls += 1
        self.bytes_seen += len(subject)
        if self.tier == TIER_JIT:
            return self.program(subject)
        if self.tier == TIER_DFA:
            return self.dfa.fullmatch(subject)
        try:
            got = self.bt.fullmatch(subject)
        except backtrack.Exhausted:
            # The safety net. A backtracking engine that has already spent its
            # budget is not going to get cheaper, so this is the one promotion
            # no policy gets a say in: build the DFA and answer the question.
            # Without it the engine can hang, and an engine that hangs is not a
            # baseline, it is an outage.
            self.forced += 1
            if self.promote(TIER_DFA) is None or self.dfa is None:
                raise
            return self.matches(subject)
        self.steps += self.bt.steps
        return got

    # -- promotion -----------------------------------------------------------
    def promote(self, target):
        """Move up to ``target``, building whatever that needs.

        Returns the nanoseconds spent, or ``None`` if nothing happened. Failure
        to build lowers the ceiling instead of raising, because from the
        caller's point of view a pattern that cannot be compiled is not an
        error -- it is a pattern that stays interpreted.
        """
        target = min(target, self.ceiling)
        if target <= self.tier:
            return None
        spent = 0
        if self.dfa is None and target >= TIER_DFA:
            t0 = time.perf_counter_ns()
            try:
                automaton = N.build(self.search_ast)
                self.dfa = D.minimise(D.build(automaton,
                                              max_states=self.max_states))
            except D.TooManyStates as exc:
                self.ceiling = TIER_BACKTRACK
                self.reason = str(exc)
                return None
            spent += time.perf_counter_ns() - t0
            self.n_states = len(self.dfa)
            self.n_edges = self.dfa.n_edges()
            self.build_ns[TIER_DFA] = spent
            self.tier = TIER_DFA
        if target >= TIER_JIT and self.program is None:
            from .jit import codegen, memory, program as prog
            if not memory.available():
                self.ceiling = TIER_DFA
                self.reason = "no executable memory on this platform"
                return spent or None
            t0 = time.perf_counter_ns()
            try:
                self.program = prog.compile_dfa(self.dfa)
            except codegen.NotCompilable as exc:
                self.ceiling = TIER_DFA
                self.reason = str(exc)
                return spent or None
            jit_ns = time.perf_counter_ns() - t0
            spent += jit_ns
            self.build_ns[TIER_JIT] = jit_ns
            self.code_size = len(self.program.code)
            self.tier = TIER_JIT
        return spent

    def listing(self):
        """The generated machine code, or an explanation of its absence."""
        from .jit import program as prog
        if self.dfa is None:
            self.promote(TIER_DFA)
        if self.dfa is None:
            return "not compilable: %s" % self.reason
        return prog.listing_for(self.dfa)

    def free(self):
        if self.program is not None:
            self.program.free()
            self.program = None

    def __repr__(self):
        return "<Pattern %r tier=%s calls=%d>" % (
            self.source, TIER_NAMES[self.tier], self.calls)


class Engine(object):
    """A cache of patterns plus a policy that decides when to promote them.

    The policy sees the pattern's history and returns a target tier; the engine
    applies it. Keeping the two apart is what makes the experiment possible --
    the same recorded workload can be replayed against every policy, including
    the hindsight oracle, with no other change.
    """

    def __init__(self, policy=None, max_states=5000, step_limit=None,
                 jit_enabled=True):
        from . import policy as policy_mod
        self.policy = policy or policy_mod.Never()
        self.max_states = max_states
        self.step_limit = step_limit
        self.jit_enabled = jit_enabled
        self.patterns = {}
        self.promotions = []           # (source, from_tier, to_tier, calls)
        # Split so a policy's own overhead can be told apart from the work it
        # causes. In a Python host the decision can cost more than a compiled
        # match, which is worth knowing and worth not hiding inside a total.
        self.match_ns = 0
        self.build_ns = 0
        self.decide_ns = 0

    def pattern(self, source):
        p = self.patterns.get(source)
        if p is None:
            p = Pattern(source, max_states=self.max_states,
                        step_limit=self.step_limit,
                        jit_enabled=self.jit_enabled)
            self.patterns[source] = p
        return p

    def matches(self, source, subject):
        p = self.pattern(source)
        t0 = time.perf_counter_ns()
        target = self.policy.target(p, subject)
        t1 = time.perf_counter_ns()
        self.decide_ns += t1 - t0
        if target > p.tier:
            was = p.tier
            spent = p.promote(target)
            self.build_ns += time.perf_counter_ns() - t1
            if spent is not None and p.tier != was:
                for tier in range(was + 1, p.tier + 1):
                    self.policy.observe_build(p, tier, p.build_ns.get(tier))
                self.promotions.append((source, was, p.tier, p.calls))
        t2 = time.perf_counter_ns()
        got = p.matches(subject)
        self.match_ns += time.perf_counter_ns() - t2
        return got

    def totals(self):
        """``match + build`` is the number the oracle is comparable to."""
        return {
            "match_ns": self.match_ns,
            "build_ns": self.build_ns,
            "decide_ns": self.decide_ns,
            "engine_ns": self.match_ns + self.build_ns,
            "wall_ns": self.match_ns + self.build_ns + self.decide_ns,
            "promotions": len(self.promotions),
            "forced": sum(p.forced for p in self.patterns.values()),
        }

    def free(self):
        for p in self.patterns.values():
            p.free()
