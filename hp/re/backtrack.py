"""Tier 0: a backtracking matcher over the AST.

This is the engine every mainstream language ships. It is here for three
reasons, and only the first is obvious.

It is the **cheapest tier to enter**: matching can begin the instant the pattern
parses, with no automaton to construct. For a pattern used twice on short
inputs that is the right answer, and a tiering policy that cannot express "do
nothing" is not a tiering policy.

It is the **reference for the reference**. The DFA is checked against Python's
``re``, but Python's ``re`` cannot be asked "how much work did that take", and
it certainly cannot be asked from inside generated machine code. This one can.

And it is the **fault being measured**. ``(a+)+b`` against thirty ``a``s and no
``b`` takes exponentially many steps here and thirty in the DFA. Every
catastrophic-backtracking incident anyone has ever paged about is that gap. The
step counter below turns the gap into a number, which is what lets the tiering
policy be evaluated rather than asserted.

Matching is greedy and leftmost, with a continuation passed down so that ``a*a``
can give back characters the way a real backtracking engine does.
"""

from . import parser as P


class Exhausted(Exception):
    """The backtracking matcher gave up. Two ways that happens, both real."""


class StepLimit(Exhausted):
    """The step budget ran out.

    Not a bug in the pattern and not a bug in the input. It is the
    *combination* that is pathological, which is exactly the thing no amount of
    static analysis of the pattern can predict and the tiering policy therefore
    has to survive.
    """

    def __init__(self, steps):
        Exhausted.__init__(self, "backtracking exceeded %d steps" % steps)
        self.steps = steps


class StackLimit(Exhausted):
    """The interpreter stack ran out.

    A backtracking matcher recurses once per input byte it is holding a choice
    point for, so on a long enough subject it dies of depth rather than of time,
which is the same failure that made long inputs crash real regex engines
    for years. It is caught rather than propagated because from the engine's
    point of view it means exactly what a step-limit does: this tier cannot
    answer, promote.
    """


class Matcher(object):
    """Greedy leftmost backtracking over the AST.

    ``steps`` counts every node visit, including the ones that get undone. It is
    the honest cost of this tier: unlike wall-clock time it does not depend on
    what else the machine is doing, so policy experiments over it reproduce
    exactly.
    """

    def __init__(self, node, limit=None):
        self.node = node
        self.limit = limit
        self.steps = 0
        self.data = b""
        self.n = 0

    # -- entry points --------------------------------------------------------
    def fullmatch(self, data):
        self._reset(data)
        try:
            return self._m(self.node, 0, lambda p: p == self.n)
        except RecursionError:
            raise StackLimit("backtracking ran out of stack on %d bytes"
                             % self.n)

    def match_end(self, data, start):
        """Longest match beginning at ``start``; ``-1`` if none.

        Backtracking is greedy, not longest, ``a|ab`` stops at ``a``. To keep
        the three tiers interchangeable this asks for each candidate end
        explicitly, longest first, instead of reporting whatever greedy happened
        to reach. It costs a factor of the subject length, which is a fair price
        for the tiers agreeing on every input.
        """
        self._reset(data)
        try:
            for end in range(self.n, start - 1, -1):
                if self._m(self.node, start,
                           (lambda e: lambda p: p == e)(end)):
                    return end
        except RecursionError:
            raise StackLimit("backtracking ran out of stack on %d bytes"
                             % self.n)
        return -1

    def search(self, data):
        for start in range(len(data) + 1):
            end = self.match_end(data, start)
            if end >= 0:
                return start, end
        return None

    def _reset(self, data):
        self.data = data
        self.n = len(data)
        self.steps = 0

    # -- the matcher ---------------------------------------------------------
    def _m(self, node, pos, k):
        """True if ``node`` matches at ``pos`` and ``k`` accepts what follows."""
        self.steps += 1
        if self.limit is not None and self.steps > self.limit:
            raise StepLimit(self.limit)

        if isinstance(node, P.Empty):
            return k(pos)

        if isinstance(node, P.Class):
            return (pos < self.n and self.data[pos] in node.bytes_
                    and k(pos + 1))

        if isinstance(node, P.Anchor):
            # ``pos == 0`` is the whole test for ``^``: positions only increase,
            # so reaching offset zero means the match began there.
            ok = (pos == 0) if node.kind == "^" else (pos == self.n)
            return ok and k(pos)

        if isinstance(node, P.Concat):
            return self._concat(node.parts, 0, pos, k)

        if isinstance(node, P.Alt):
            for option in node.options:
                if self._m(option, pos, k):
                    return True
            return False

        if isinstance(node, P.Repeat):
            return self._repeat(node, pos, k, 0)

        raise AssertionError("unknown AST node %r" % (node,))  # pragma: no cover

    def _concat(self, parts, i, pos, k):
        if i == len(parts):
            return k(pos)
        return self._m(parts[i], pos, lambda p: self._concat(parts, i + 1, p, k))

    def _repeat(self, node, pos, k, done):
        """Greedy: take another copy if allowed, else stop here and continue.

        The ``p == pos`` guard is what stops ``(a*)*`` from looping forever on a
        body that can match the empty string. It is not an optimisation.
Without it the engine does not terminate, which is a fact about
        backtracking engines that people tend to discover the hard way.
        """
        high = node.max
        if high is None or done < high:
            def more(p):
                if p == pos and done >= node.min:
                    return False                  # zero-width repetition
                return self._repeat(node, p, k, done + 1)
            if self._m(node.node, pos, more):
                return True
        return done >= node.min and k(pos)


def compile_pattern(pattern, limit=None):
    return Matcher(P.parse(pattern), limit=limit)
