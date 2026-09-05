"""When is a pattern worth compiling?

Every tiering policy is a bet on the future made from the past, and the bet has
an asymmetric payoff. Promote too eagerly and you pay a subset construction and
a syscall for a pattern used twice. Promote too late and you have already burned
the time compiling would have saved, and in the pathological case, rather more
than that.

The policies here range from the two trivial answers, through the counter-based
heuristic every real JIT actually ships, to one that reasons about cost
explicitly. They share an interface: given a pattern's history, name a tier.
None of them may look at the subject beyond its length, and none of them may
look at the future. That privilege belongs to the oracle in
:mod:`hp.eval.oracle`, which exists to say how much was on the table.
"""

import math

from . import engine as E


class Policy(object):
    name = "policy"

    def target(self, pattern, subject):
        raise NotImplementedError                       # pragma: no cover

    def observe_build(self, pattern, tier, ns):
        """Told what a promotion actually cost, once it has happened.

        Most policies ignore it. It is here because the alternative. Letting
        one policy reach into the engine for feedback the others cannot see.
Would make the comparison meaningless.
        """

    def __repr__(self):
        return "<%s>" % self.name


class Never(Policy):
    """Stay interpreted forever. The floor, and the status quo for one-shot use."""
    name = "never"

    def target(self, pattern, subject):
        return E.TIER_BACKTRACK


class Fixed(Policy):
    """Promote to a tier on first sight and never think again.

    ``Fixed(TIER_JIT)`` is the "compile everything" strawman. It is not a
    strawman for long-running scanners, which is exactly why it needs measuring
    rather than dismissing.
    """

    def __init__(self, tier):
        self.tier = tier
        self.name = "always-%s" % E.TIER_NAMES[tier]

    def target(self, pattern, subject):
        return self.tier


class CallCount(Policy):
    """Promote after a fixed number of calls: the counter every JIT ships.

    Simple, robust, and blind to size. It treats a pattern called ten times on
    ten bytes exactly like one called ten times on ten megabytes. Whether that
    blindness costs anything is an empirical question, and it is one of the
    questions the evaluation answers.
    """

    def __init__(self, dfa_after, jit_after):
        self.dfa_after = dfa_after
        self.jit_after = jit_after
        self.name = "calls(%d,%d)" % (dfa_after, jit_after)

    def target(self, pattern, subject):
        if pattern.calls >= self.jit_after:
            return E.TIER_JIT
        if pattern.calls >= self.dfa_after:
            return E.TIER_DFA
        return E.TIER_BACKTRACK


class ByteCount(Policy):
    """Promote after a volume of input, not a number of calls.

    The obvious repair to :class:`CallCount`: one call over a megabyte is worth
    compiling for, a thousand calls over eight bytes each may not be.
    """

    def __init__(self, dfa_after, jit_after):
        self.dfa_after = dfa_after
        self.jit_after = jit_after
        self.name = "bytes(%d,%d)" % (dfa_after, jit_after)

    def target(self, pattern, subject):
        seen = pattern.bytes_seen + len(subject)
        if seen >= self.jit_after:
            return E.TIER_JIT
        if seen >= self.dfa_after:
            return E.TIER_DFA
        return E.TIER_BACKTRACK


class CostModel(object):
    """Machine constants, measured rather than assumed.

    :mod:`hp.eval.calibrate` fills these in by timing the real tiers on the
    machine the experiment runs on. The defaults are only there so a
    :class:`Amortised` built without calibration still behaves sensibly; every
    reported number comes from measured ones.
    """

    def __init__(self, bt_ns_per_step=90.0,
                 dfa_ns_per_byte=95.0, dfa_ns_fixed=700.0,
                 jit_ns_per_byte=1.5, jit_ns_fixed=1800.0,
                 bt_ns_fixed=0.0,
                 dfa_build_log_const=7.0, dfa_build_exp_nfa=1.68,
                 dfa_build_exp_class=0.38, dfa_build_exp_span=0.22,
                 jit_build_ns_fixed=90000.0, jit_build_ns_per_code_byte=25.0,
                 code_bytes_fixed=20.0, code_bytes_per_state=28.0,
                 code_bytes_per_edge=9.0):
        self.bt_ns_per_step = bt_ns_per_step
        self.bt_ns_fixed = bt_ns_fixed
        self.dfa_ns_per_byte = dfa_ns_per_byte
        self.dfa_ns_fixed = dfa_ns_fixed
        self.jit_ns_per_byte = jit_ns_per_byte
        self.jit_ns_fixed = jit_ns_fixed
        self.dfa_build_log_const = dfa_build_log_const
        self.dfa_build_exp_nfa = dfa_build_exp_nfa
        self.dfa_build_exp_class = dfa_build_exp_class
        self.dfa_build_exp_span = dfa_build_exp_span
        self.jit_build_ns_fixed = jit_build_ns_fixed
        self.jit_build_ns_per_code_byte = jit_build_ns_per_code_byte
        self.code_bytes_fixed = code_bytes_fixed
        self.code_bytes_per_state = code_bytes_per_state
        self.code_bytes_per_edge = code_bytes_per_edge

    # -- what a promotion will cost -----------------------------------------
    def dfa_build_cost(self, pattern):
        """Estimated, from the shape of the NFA the pattern will produce.

        This is the weak link in the whole model, and it is weak for a reason
        that is not fixable by choosing better features. Subset construction
        cost tracks the number of DFA states *before* minimisation, and that
        number is not a function of anything observable beforehand:
        ``[A-Z]{2}\d{2}[A-Z0-9]{4,20}`` has 59 NFA states and 9 DFA states, and
        passes through 3312 on the way, a hundred times the cost of patterns
        that look just like it. Fitting against pattern length was useless;
        against NFA size and class count it is merely bad, and the calibration
        output says by how much.

        The form is multiplicative rather than additive, a power law in the
        NFA size, the class count and the counted-repetition span, because the
        quantity spans two orders of magnitude and an additive fit spends all
        its accuracy on the largest case. That change alone took the mean
        relative error from 352% to 55%; it is still the worst-fitting part of
        the model by a wide margin.

        :class:`Adaptive` is the response: stop predicting and start measuring.
        """
        return math.exp(
            self.dfa_build_log_const
            + self.dfa_build_exp_nfa * math.log(max(pattern.nfa_states, 1))
            + self.dfa_build_exp_class * math.log(pattern.ast_classes + 1)
            + self.dfa_build_exp_span * math.log(pattern.repeat_span + 1))

    def estimated_code_bytes(self, n_states, n_edges):
        return (self.code_bytes_fixed
                + self.code_bytes_per_state * n_states
                + self.code_bytes_per_edge * n_edges)

    def jit_build_cost(self, pattern):
        """Estimated, but by this point the DFA exists, so it is a good one.

        Deliberately phrased in terms of the state and edge counts rather than
        the DFA object, so that a policy can be driven by a replay of recorded
        measurements exactly as it is driven by the live engine. If the policy
        could only run against live objects, the simulation and the real run
        would be two different policies and comparing them would prove nothing.
        """
        if pattern.n_states is None:
            return self.jit_build_ns_fixed + self.jit_build_ns_per_code_byte * 400
        return (self.jit_build_ns_fixed + self.jit_build_ns_per_code_byte
                * self.estimated_code_bytes(pattern.n_states,
                                            pattern.n_edges or 0))

    # -- what a call costs at each tier -------------------------------------
    def match_cost(self, tier, n_bytes, steps_per_byte):
        if tier == E.TIER_JIT:
            return self.jit_ns_fixed + self.jit_ns_per_byte * n_bytes
        if tier == E.TIER_DFA:
            return self.dfa_ns_fixed + self.dfa_ns_per_byte * n_bytes
        return (self.bt_ns_fixed
                + self.bt_ns_per_step * steps_per_byte * max(n_bytes, 1))


class Amortised(Policy):
    """Promote once the work already wasted would have paid for the build.

    The rule is one line: keep a running total of what this pattern has cost at
    its current tier *in excess of* what it would have cost one tier up, and
    promote when that excess reaches ``K`` times the estimated build cost.

    ``K = 1`` promotes as soon as the promotion has notionally broken even on
    past traffic alone. Optimistic, since past traffic is spent and only
    future traffic can actually be recovered. ``K = 4`` demands the pattern
    prove itself four times over. Sweeping ``K`` is the experiment: it turns
    "when should you compile" from an opinion into a curve.

    The excess is measured from observed behaviour, including the backtracking
    step count, which is the only signal that distinguishes a pattern that is
    merely popular from one that is quietly quadratic.
    """

    def __init__(self, k=1.0, model=None, trip=False):
        self.k = k
        self.model = model or CostModel()
        self.trip = trip
        self.name = "amortised%s(K=%g)" % ("+trip" if trip else "", k)
        self._excess = {}

    def target(self, pattern, subject):
        if pattern.tier >= pattern.ceiling:
            return pattern.tier
        m = self.model
        n = len(subject)
        steps_per_byte = (pattern.steps / max(pattern.bytes_seen, 1)
                          if pattern.calls else 1.0)

        here = m.match_cost(pattern.tier, n, steps_per_byte)
        up = m.match_cost(pattern.tier + 1, n, steps_per_byte)
        key = (id(pattern), pattern.tier)
        excess = self._excess.get(key, 0.0) + max(here - up, 0.0)
        self._excess[key] = excess

        build = self._build_cost(pattern)
        if excess >= self.k * build:
            return pattern.tier + 1
        # The trip. Accumulating excess is the right rule for a pattern that is
        # merely popular, and the wrong one for a pattern that is pathological:
        # a single catastrophic call can cost more than the build, and waiting
        # for K of them to accumulate means paying that cost K times over. So a
        # single call whose excess alone exceeds the build promotes on the spot.
        if self.trip and max(here - up, 0.0) >= build:
            return pattern.tier + 1
        return pattern.tier

    def _build_cost(self, pattern):
        m = self.model
        return (m.dfa_build_cost(pattern) if pattern.tier == E.TIER_BACKTRACK
                else m.jit_build_cost(pattern))


class Adaptive(Amortised):
    """:class:`Amortised`, but the build cost is learned instead of predicted.

    The estimate a cost-based policy most needs is the one it can least compute:
    what the next subset construction will cost. So this one stops trying. It
    starts from the model's guess and, every time a promotion actually happens,
    folds the measured cost into a running average, one number per tier, shared
    across every pattern in the process.

    The bet is that build cost is far more predictable *across patterns in one
    workload* than it is *from one pattern's text*, because a workload's
    patterns tend to resemble each other. When that holds this should track the
    truth within a few promotions. When it does not, a workload of one
    expensive pattern among many cheap ones, the average is dragged the wrong
    way and this should do worse than the static estimate. Both cases appear in
    the evaluation.

    ``weight`` is the exponential decay: 0.3 means each new observation moves
    the estimate 30% of the way.
    """

    def __init__(self, k=1.0, model=None, weight=0.3, trip=False):
        Amortised.__init__(self, k=k, model=model, trip=trip)
        self.weight = weight
        self.name = "adaptive%s(K=%g)" % ("+trip" if trip else "", k)
        self.learned = {}
        self.seen = {}

    def observe_build(self, pattern, tier, ns):
        if ns is None or ns <= 0:
            return
        prior = self.learned.get(tier)
        self.learned[tier] = (ns if prior is None
                              else prior + self.weight * (ns - prior))
        self.seen[tier] = self.seen.get(tier, 0) + 1

    def _build_cost(self, pattern):
        target = pattern.tier + 1
        learned = self.learned.get(target)
        if learned is None:
            return Amortised._build_cost(self, pattern)
        return learned


class Guarded(Policy):
    """Any policy, plus a static veto on patterns that can detonate.

    The evaluation produced one result no amount of tuning improved: on a
    workload of catastrophic-backtracking patterns, every reactive policy lost
    to simply always building the DFA, by one to two orders of magnitude. That
    is not a tuning failure, it is structural. A reactive policy decides using
    evidence from calls it has already made, and for these patterns the first
    bad call is the disaster.

    So this wrapper stops being reactive about them. Patterns whose *shape*
    admits exponential backtracking (:func:`hp.re.parser.risky`) skip tier 0
    entirely; everything else is handed to the wrapped policy unchanged. The
    check is a walk of the AST, it runs once per pattern, and it is the only
    place in this project where a decision is made without measurement,
because measurement is exactly what is unaffordable here.
    """

    def __init__(self, inner, model=None):
        self.inner = inner
        self.model = model or inner.model if hasattr(inner, "model") \
            else CostModel()
        self.name = "guarded/" + inner.name

    def target(self, pattern, subject):
        base = self.inner.target(pattern, subject)
        if pattern.tier > E.TIER_BACKTRACK:
            return base
        if getattr(pattern, "risky", False):
            return max(E.TIER_DFA, base)
        # The second veto, from the same principle. Tier 0 is never faster per
        # byte than the DFA. It is the same scan with interpretive overhead on
        # top, so if a single subject is long enough that even the DFA's
        # per-byte cost would exceed the build, running it on the backtracker
        # cannot pay off no matter what happens later. There is nothing to
        # learn from that call except how long it takes.
        return max(self._single_call_tier(pattern, len(subject)), base)

    def _single_call_tier(self, pattern, n):
        """The cheapest tier for *this one call*, builds included.

        Tier 0 is never faster per byte than the DFA. It is the same left-to-
        right scan with interpretive overhead on top, so its cost is at least
        the DFA's. That gives a lower bound good enough to compare against, and
        the comparison settles the question without waiting: if paying for the
        machine code is cheaper than scanning this single subject one tier down,
        the promotion has already paid for itself before the call runs.

        This is where the ``bulk`` workload was being lost. A pattern seen six
        times on quarter-megabyte blobs never accumulates enough *calls* to trip
        a counter, and by the time an amortising policy has watched two of them
        it has spent more than the compiler would have cost.
        """
        m = self.model
        dfa_build = (self.inner._build_cost(pattern)
                     if hasattr(self.inner, "_build_cost")
                     else m.dfa_build_cost(pattern))
        scan_dfa = m.dfa_ns_fixed + m.dfa_ns_per_byte * n
        scan_jit = m.jit_ns_fixed + m.jit_ns_per_byte * n
        jit_build = m.jit_build_cost(pattern)
        if dfa_build + jit_build + scan_jit < scan_dfa:
            return E.TIER_JIT
        if scan_dfa < dfa_build:          # tier 0 costs at least this much
            return E.TIER_BACKTRACK
        return E.TIER_DFA

    def observe_build(self, pattern, tier, ns):
        self.inner.observe_build(pattern, tier, ns)


ALL = {
    "never": lambda: Never(),
    "always-dfa": lambda: Fixed(E.TIER_DFA),
    "always-jit": lambda: Fixed(E.TIER_JIT),
    "calls-2": lambda: CallCount(2, 4),
    "calls-10": lambda: CallCount(10, 50),
    "calls-100": lambda: CallCount(100, 500),
    "bytes-4k": lambda: ByteCount(4 << 10, 64 << 10),
    "bytes-64k": lambda: ByteCount(64 << 10, 1 << 20),
}
