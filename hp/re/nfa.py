"""Thompson construction: AST -> nondeterministic finite automaton.

Each AST node becomes a small NFA fragment with one entry and one exit, wired
together with epsilon transitions. The construction is linear in the size of the
pattern, which is the property that makes the whole pipeline safe: there is no
way to write a pattern whose *automaton* blows up, only one whose subset
construction does, and that is bounded and detectable.

Anchors are the one subtlety. ``^`` and ``$`` do not consume input, so they are
epsilon edges -- but conditional ones, valid only at the start or the end of the
subject. They are carried through as guarded edges and resolved when the epsilon
closure is taken, because that is the only place the position is known.
"""

from . import parser as P

EPSILON = None
START_ANCHOR = "^"
END_ANCHOR = "$"


class NFA(object):
    """States are integers. ``trans[s]`` is a list of ``(guard, target)``.

    ``guard`` is a frozenset of bytes for a consuming edge, or one of
    ``EPSILON`` / ``START_ANCHOR`` / ``END_ANCHOR`` for a non-consuming one.
    """

    def __init__(self):
        self.trans = []
        self.start = None
        self.accept = None

    def new_state(self):
        self.trans.append([])
        return len(self.trans) - 1

    def edge(self, src, guard, dst):
        self.trans[src].append((guard, dst))

    def __len__(self):
        return len(self.trans)

    def byte_sets(self):
        """Every distinct consuming guard, for building the alphabet.

        Returned in a canonical order: the alphabet builder enumerates these,
        and a set's iteration order varies with the hash seed, which would make
        the generated machine code differ between runs of the same program.
        """
        out = set()
        for edges in self.trans:
            for guard, _ in edges:
                if isinstance(guard, frozenset):
                    out.add(guard)
        return sorted(out, key=lambda s: (len(s), sorted(s)))

    def closure(self, states, at_start, at_end):
        """Epsilon closure, honouring anchors for the given position."""
        stack = list(states)
        seen = set(states)
        while stack:
            s = stack.pop()
            for guard, dst in self.trans[s]:
                if dst in seen:
                    continue
                if guard is EPSILON:
                    pass
                elif guard is START_ANCHOR:
                    if not at_start:
                        continue
                elif guard is END_ANCHOR:
                    if not at_end:
                        continue
                else:
                    continue                      # consuming edge
                seen.add(dst)
                stack.append(dst)
        return frozenset(seen)

    def step(self, states, byte):
        """States reachable by consuming ``byte`` (before taking the closure)."""
        out = set()
        for s in states:
            for guard, dst in self.trans[s]:
                if isinstance(guard, frozenset) and byte in guard:
                    out.add(dst)
        return out


def size(node):
    """How many states :func:`build` will create, without creating them.

    Mirrors ``_emit`` case for case. It exists because counted repetition is the
    one construct whose cost is invisible in the source text: ``[A-Z0-9]{4,20}``
    is seven characters and seventeen copies of an automaton. A policy deciding
    whether to pay for a subset construction can afford this -- it is a walk of
    the AST -- and cannot afford to be surprised by it.
    """
    if isinstance(node, (P.Empty, P.Class, P.Anchor)):
        return 2
    if isinstance(node, P.Concat):
        return sum(size(part) for part in node.parts)
    if isinstance(node, P.Alt):
        return 2 + sum(size(opt) for opt in node.options)
    if isinstance(node, P.Repeat):
        body = size(node.node)
        if node.max is None:
            return node.min * body + body + 2
        copies = node.max
        # ``{0,0}`` emits no copy of the body at all: one state for the shared
        # exit and one for the entry that epsilon-jumps straight to it.
        return (copies * body + 1) if copies else 2
    raise AssertionError("unknown AST node %r" % (node,))   # pragma: no cover


def build(node):
    """Compile an AST to an NFA with a single start and a single accept state."""
    nfa = NFA()
    start, accept = _emit(nfa, node)
    nfa.start = start
    nfa.accept = accept
    return nfa


def _emit(nfa, node):
    if isinstance(node, P.Empty):
        s = nfa.new_state()
        e = nfa.new_state()
        nfa.edge(s, EPSILON, e)
        return s, e

    if isinstance(node, P.Class):
        s = nfa.new_state()
        e = nfa.new_state()
        nfa.edge(s, node.bytes_, e)
        return s, e

    if isinstance(node, P.Anchor):
        s = nfa.new_state()
        e = nfa.new_state()
        nfa.edge(s, START_ANCHOR if node.kind == "^" else END_ANCHOR, e)
        return s, e

    if isinstance(node, P.Concat):
        first_start = None
        prev_end = None
        for part in node.parts:
            s, e = _emit(nfa, part)
            if first_start is None:
                first_start = s
            else:
                nfa.edge(prev_end, EPSILON, s)
            prev_end = e
        return first_start, prev_end

    if isinstance(node, P.Alt):
        s = nfa.new_state()
        e = nfa.new_state()
        for option in node.options:
            os_, oe = _emit(nfa, option)
            nfa.edge(s, EPSILON, os_)
            nfa.edge(oe, EPSILON, e)
        return s, e

    if isinstance(node, P.Repeat):
        return _emit_repeat(nfa, node)

    raise AssertionError("unknown AST node %r" % (node,))     # pragma: no cover


def _emit_repeat(nfa, node):
    low, high = node.min, node.max

    if high is None:
        # {n,} : n copies, then a star on the last one
        start = end = None
        for _ in range(low):
            s, e = _emit(nfa, node.node)
            if start is None:
                start = s
            else:
                nfa.edge(end, EPSILON, s)
            end = e
        s, e = _emit(nfa, node.node)
        loop_in = nfa.new_state()
        loop_out = nfa.new_state()
        nfa.edge(loop_in, EPSILON, s)
        nfa.edge(loop_in, EPSILON, loop_out)      # skip: zero more repetitions
        nfa.edge(e, EPSILON, loop_in)             # go round again
        nfa.edge(e, EPSILON, loop_out)
        if start is None:
            return loop_in, loop_out
        nfa.edge(end, EPSILON, loop_in)
        return start, loop_out

    # {m,n} : m mandatory copies followed by n-m optional ones
    start = end = None
    for _ in range(low):
        s, e = _emit(nfa, node.node)
        if start is None:
            start = s
        else:
            nfa.edge(end, EPSILON, s)
        end = e
    final = nfa.new_state()
    if end is not None:
        nfa.edge(end, EPSILON, final)
    for _ in range(high - low):
        s, e = _emit(nfa, node.node)
        if start is None:
            start = s
            end = e
            nfa.edge(s, EPSILON, final)           # this copy may be skipped
        else:
            nfa.edge(end, EPSILON, s)
            end = e
        nfa.edge(e, EPSILON, final)
    if start is None:                              # {0,0}: matches only empty
        s = nfa.new_state()
        nfa.edge(s, EPSILON, final)
        return s, final
    return start, final
