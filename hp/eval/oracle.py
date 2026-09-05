"""What the best possible policy would have cost, in hindsight.

Without this number a policy comparison is unmoored. "Policy A is 12% faster
than policy B" says nothing about whether either is any good; both might be
leaving 90% of the available saving on the table, or A might be within a
rounding error of perfect. So every result in this project is reported as a
*fraction of what was achievable*, and this module computes the denominator.

The oracle is given the entire trace in advance and asked, for each pattern,
where to build the DFA and where to compile it. Tiers are monotone. Once
promoted a pattern stays promoted, so a schedule is just two indices, and the
exact optimum over all of them falls out of prefix sums in linear time:

    total(i, j) = S0[i] + (S1[j] - S1[i]) + (S2[n] - S2[j])
                  + build_dfa if i < n
                  + build_jit if j < n

with ``i`` the first call served by the DFA, ``j`` the first served by machine
code, and ``Sk`` the prefix sums of the measured per-call costs at tier ``k``.
Minimising over ``0 <= i <= j <= n`` needs one pass and a running minimum. No
search, no approximation: this is the optimum, not a good schedule.

Two weaker oracles are computed alongside, because the gap between them is
itself a result:

* **best fixed tier**: the best single tier for the whole trace, chosen in
  hindsight. The difference between this and the full oracle is exactly the
  value of being able to change tier at all.
* **free promotion**: the full oracle with build costs set to zero. The
  difference between this and the full oracle is exactly what compilation costs.
"""

from .. import engine as E

INF = float("inf")


def _prefix(values):
    out = [0.0]
    acc = 0.0
    for v in values:
        acc += v
        out.append(acc)
    return out


def best_schedule(c0, c1, c2, build_dfa, build_jit,
                  has_dfa=True, has_jit=True):
    """Optimal ``(cost, i, j)`` for one pattern.

    ``i`` is the index of the first call served by the DFA and ``j`` the first
    served by machine code; ``n`` for either means "never".
    """
    n = len(c0)
    S0 = _prefix(c0)
    best = (S0[n], n, n)                      # never promote

    if not has_dfa:
        return best

    S1 = _prefix(c1)
    for i in range(n):                        # promote to the DFA at i, stop
        cost = S0[i] + (S1[n] - S1[i]) + build_dfa
        if cost < best[0]:
            best = (cost, i, n)

    if not has_jit:
        return best

    S2 = _prefix(c2)
    # f(i) = S0[i] - S1[i]; running minimum over i <= j.
    best_f, best_i = INF, 0
    for j in range(n):
        f = S0[j] - S1[j]
        if f < best_f:
            best_f, best_i = f, j
        cost = best_f + (S1[j] - S2[j]) + S2[n] + build_dfa + build_jit
        if cost < best[0]:
            best = (cost, best_i, j)
    return best


def best_fixed_tier(c0, c1, c2, build_dfa, build_jit, has_dfa, has_jit):
    """Cheapest single tier for the whole trace, build cost included."""
    options = [(sum(c0), E.TIER_BACKTRACK)]
    if has_dfa:
        options.append((sum(c1) + build_dfa, E.TIER_DFA))
    if has_jit:
        options.append((sum(c2) + build_dfa + build_jit, E.TIER_JIT))
    return min(options)


def evaluate(costs_by_pattern):
    """Oracle totals over every pattern in a measured trace."""
    total, fixed_total, free_total, never_total = 0.0, 0.0, 0.0, 0.0
    schedules = {}
    for pat, c in costs_by_pattern.items():
        has_dfa = E.TIER_DFA in c.available_tiers
        has_jit = E.TIER_JIT in c.available_tiers
        c0 = c.ns[E.TIER_BACKTRACK]
        c1 = c.ns[E.TIER_DFA] if has_dfa else []
        c2 = c.ns[E.TIER_JIT] if has_jit else []
        bd = c.build_ns.get(E.TIER_DFA, 0)
        bj = c.build_ns.get(E.TIER_JIT, 0)

        cost, i, j = best_schedule(c0, c1, c2, bd, bj, has_dfa, has_jit)
        total += cost
        never_total += sum(c0)
        fixed_total += best_fixed_tier(c0, c1, c2, bd, bj, has_dfa, has_jit)[0]
        free_total += best_schedule(c0, c1, c2, 0, 0, has_dfa, has_jit)[0]
        schedules[pat] = {"cost_ns": cost, "dfa_at": i, "jit_at": j,
                          "calls": len(c0), "states": c.n_states,
                          "code_bytes": c.code_size}
    return {
        "oracle_ns": total,
        "best_fixed_tier_ns": fixed_total,
        "free_promotion_ns": free_total,
        "never_ns": never_total,
        "schedules": schedules,
    }


def recovered(policy_ns, never_ns, oracle_ns):
    """Fraction of the achievable saving a policy actually got.

    ``1.0`` is the oracle; ``0.0`` is doing nothing; negative means the policy
    made things worse than not tiering at all, which several of them do on at
    least one workload. That is the interesting part.
    """
    available = never_ns - oracle_ns
    if available <= 0:
        return None
    return (never_ns - policy_ns) / available
