"""Replay a policy against recorded per-call measurements.

Policies are compared here rather than by re-running them, for one reason: the
oracle optimises over the numbers :mod:`hp.eval.measure` recorded, so a policy
scored by a fresh timing run is being compared against a different set of
measurements than the one it is being judged by. Machine noise between the two
passes then shows up as a policy beating the optimum, which happened, and
which is nonsense.

Driving both from the same recorded costs removes that entirely: a policy cannot
score better than the oracle, and the difference between two policies is
attributable to their decisions rather than to what else the machine was doing.

The policies themselves are the real objects from :mod:`hp.policy`, unmodified.
They see a stand-in pattern exposing exactly the attributes the live engine
exposes. ``calls``, ``bytes_seen``, ``steps``, ``tier``, ``ceiling``,
``n_states``, ``n_edges``: so what is simulated is the decision, never the
policy. :func:`hp.eval.run.validate` then re-runs the same policies for real and
reports how far the simulation was off, which is the check that this shortcut is
allowed.
"""

from .. import engine as E


class SimPattern(object):
    """What a policy sees during replay. Same attributes as the live pattern."""

    def __init__(self, source, costs):
        self.source = source
        self.costs = costs
        self.ast_nodes = costs.ast_nodes
        self.ast_classes = costs.ast_classes
        self.nfa_states = costs.nfa_states
        self.repeat_span = costs.repeat_span
        self.risky = costs.risky
        self.tier = E.TIER_BACKTRACK
        self.ceiling = max(costs.available_tiers)
        self.calls = 0
        self.bytes_seen = 0
        self.steps = 0
        self.forced = 0
        self.n_states = None
        self.n_edges = None
        self.index = 0                 # position in this pattern's own history

    def build_to(self, target, policy=None):
        """Enter every tier up to ``target``; returns the build cost paid.

        Feedback to the policy goes through the same hook the live engine uses,
        with the same numbers, so a policy that learns from build costs learns
        the same thing in both.
        """
        target = min(target, self.ceiling)
        paid = 0
        if target >= E.TIER_DFA and self.tier < E.TIER_DFA:
            ns = self.costs.build_ns.get(E.TIER_DFA, 0)
            paid += ns
            self.n_states = self.costs.n_states
            self.n_edges = self.costs.n_edges
            self.tier = E.TIER_DFA
            if policy is not None:
                policy.observe_build(self, E.TIER_DFA, ns)
        if target >= E.TIER_JIT and self.tier < E.TIER_JIT:
            ns = self.costs.build_ns.get(E.TIER_JIT, 0)
            paid += ns
            self.tier = E.TIER_JIT
            if policy is not None:
                policy.observe_build(self, E.TIER_JIT, ns)
        return paid


def run(trace, costs_by_pattern, policy):
    """Total nanoseconds this policy would have spent on this trace."""
    sims = {pat: SimPattern(pat, c) for pat, c in costs_by_pattern.items()}
    match_ns = 0
    build_ns = 0
    promotions = 0

    for source, subject in trace:
        p = sims[source]
        was = p.tier
        target = policy.target(p, subject)
        if target > p.tier:
            build_ns += p.build_to(target, policy)
        i = p.index

        # The safety net, exactly as the live engine applies it: a tier-0 call
        # that exhausts its step budget forces a promotion whatever the policy
        # thinks, and the budget is spent before the DFA runs.
        if p.tier == E.TIER_BACKTRACK and p.costs.censored_calls[i]:
            match_ns += p.costs.ns[E.TIER_BACKTRACK][i]
            build_ns += p.build_to(E.TIER_DFA, policy)
            p.forced += 1

        if p.tier != was:
            promotions += 1
        match_ns += p.costs.ns[p.tier][i]
        if p.tier == E.TIER_BACKTRACK:
            p.steps += p.costs.steps[i]
        p.calls += 1
        p.bytes_seen += len(subject)
        p.index += 1

    tiers = {}
    for p in sims.values():
        key = E.TIER_NAMES[p.tier]
        tiers[key] = tiers.get(key, 0) + 1
    return {
        "match_ns": match_ns,
        "build_ns": build_ns,
        "engine_ns": match_ns + build_ns,
        "promotions": promotions,
        "forced": sum(p.forced for p in sims.values()),
        "final_tiers": tiers,
    }
