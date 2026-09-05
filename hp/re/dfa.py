"""Subset construction, alphabet compression and minimisation.

Three passes, each of which earns its place:

**Alphabet compression.** A DFA over bytes nominally has 256 transitions per
state. Almost all of them are identical: for the pattern ``[a-z]+@[a-z]+``,
every byte in ``a..z`` behaves the same way everywhere, as does every byte that
appears in no class at all. So the bytes are partitioned into *equivalence
classes* -- typically three or four of them -- and the DFA is built over classes
instead. This is the difference between emitting a 2 KB jump table per state and
emitting three compare instructions, and it is why the generated code is small
enough to read.

**Subset construction.** The usual powerset algorithm. Anchors are the reason it
is not quite textbook. ``^`` and ``$`` are conditions on *position*, not on
input, so rather than let position leak into the transition function each state
carries two accept bits -- accepts here, and accepts here if the subject ends
here -- and the automaton has two start states, one for "at the beginning of the
subject" and one for "somewhere in the middle". Both are decided once, at
construction time; the running matcher never looks at a position again. That
matters more than it sounds, because the generated machine code inherits the
property: no bounds arithmetic beyond the loop counter.

**Minimisation.** Moore's partition refinement. Fewer states means less
generated code, which means the JIT pays for itself sooner -- and since the
tiering policy is the thing being measured, making compilation cheaper is not a
cosmetic improvement, it moves the results.
"""

from . import nfa as N


class DFA(object):
    """A DFA over compressed byte classes.

    ``byte_class[b]`` gives the class of byte ``b``; ``trans[s][c]`` gives the
    state reached from ``s`` on class ``c``, or ``-1`` for the dead state.

    Two accept bits per state:

    * ``accept_here[s]`` -- this state is a match, wherever the subject ends.
    * ``accept_at_end[s]`` -- this state is a match *if the subject ends here*,
      which is what ``$`` means.

    Two start states, likewise: ``start`` is entered at offset 0 (where ``^``
    holds) and ``start_mid`` anywhere else. For an unanchored pattern they are
    the same state, and minimisation merges them away.
    """

    def __init__(self, n_classes, byte_class):
        self.n_classes = n_classes
        self.byte_class = byte_class          # list of 256 ints
        self.trans = []                       # list of lists, len == n_classes
        self.accept_here = []                 # bool per state
        self.accept_at_end = []               # bool per state
        self.start = 0
        self.start_mid = 0

    def add_state(self, accept_here, accept_at_end):
        self.trans.append([-1] * self.n_classes)
        self.accept_here.append(accept_here)
        self.accept_at_end.append(accept_at_end)
        return len(self.trans) - 1

    def __len__(self):
        return len(self.trans)

    def n_edges(self):
        return sum(1 for row in self.trans for t in row if t >= 0)

    def anchored(self):
        """True when starting at offset 0 differs from starting anywhere else."""
        return self.start != self.start_mid

    # -- reference execution: tier 1, and the check the JIT is held to -------
    def fullmatch(self, data):
        s = self.start
        bc = self.byte_class
        trans = self.trans
        for b in data:
            s = trans[s][bc[b]]
            if s < 0:
                return False
        return self.accept_at_end[s]

    def match_end(self, data, start):
        """Longest match beginning exactly at ``start``; ``-1`` if none.

        A state is a match at offset ``i`` if it accepts unconditionally, or if
        it accepts at end of subject and ``i`` happens to be the end.
        """
        n = len(data)
        s = self.start if start == 0 else self.start_mid
        bc = self.byte_class
        trans = self.trans
        best = -1
        if self.accept_here[s] or (start == n and self.accept_at_end[s]):
            best = start
        i = start
        while i < n:
            s = trans[s][bc[data[i]]]
            if s < 0:
                break
            i += 1
            if self.accept_here[s] or (i == n and self.accept_at_end[s]):
                best = i
        return best

    def search(self, data):
        """Leftmost-longest match as ``(start, end)``, or ``None``."""
        end = self.match_end(data, 0)
        if end >= 0:
            return 0, end
        if self._mid_is_dead():
            # ``^abc``: the middle start state has nowhere to go, so no offset
            # after the first can begin a match. Saying so here saves the outer
            # loop, and saying it *structurally* rather than by looking for a
            # leading ``^`` in the source means it also fires for patterns that
            # are anchored only after simplification.
            return None
        for start in range(1, len(data) + 1):
            end = self.match_end(data, start)
            if end >= 0:
                return start, end
        return None

    def _mid_is_dead(self):
        s = self.start_mid
        if self.accept_here[s] or self.accept_at_end[s]:
            return False
        return all(t < 0 for t in self.trans[s])

    def describe(self):
        out = ["DFA: %d states, %d classes, %d edges, start=%d, start_mid=%d"
               % (len(self), self.n_classes, self.n_edges(),
                  self.start, self.start_mid)]
        for s, row in enumerate(self.trans):
            marks = ""
            if self.accept_here[s]:
                marks += " (accept)"
            elif self.accept_at_end[s]:
                marks += " (accept at end)"
            targets = ", ".join("c%d->%d" % (c, t)
                                for c, t in enumerate(row) if t >= 0)
            out.append("  s%d%s: %s" % (s, marks, targets or "dead end"))
        return "\n".join(out)


def build_alphabet(automaton):
    """Partition the 256 bytes into classes that no guard can tell apart.

    Two bytes are equivalent iff they belong to exactly the same set of guards,
    so the whole partition falls out of one signature per byte. Classes are
    numbered by their smallest member, which keeps the numbering stable across
    runs -- generated code that changes between runs is impossible to review.
    """
    sets = automaton.byte_sets()
    signature = {}
    for b in range(256):
        sig = frozenset(i for i, s in enumerate(sets) if b in s)
        signature.setdefault(sig, []).append(b)

    byte_class = [0] * 256
    for cls, (_, members) in enumerate(sorted(signature.items(),
                                              key=lambda kv: kv[1][0])):
        for b in members:
            byte_class[b] = cls
    return len(signature), byte_class


class TooManyStates(Exception):
    """The subset construction exceeded its bound.

    Reaching this is not a bug: some patterns genuinely have exponentially many
    DFA states. It is reported so the engine can fall back to a tier that does
    not need a DFA, rather than allocating until the machine dies.
    """


def build(automaton, max_states=5000):
    n_classes, byte_class = build_alphabet(automaton)

    # One representative byte per class, for taking transitions.
    rep = [None] * n_classes
    for b in range(256):
        if rep[byte_class[b]] is None:
            rep[byte_class[b]] = b

    dfa = DFA(n_classes, byte_class)
    accept = automaton.accept

    def add(states):
        here = accept in states
        at_end = accept in automaton.closure(states, False, True)
        return dfa.add_state(here, at_end)

    bol = automaton.closure({automaton.start}, at_start=True, at_end=False)
    mid = automaton.closure({automaton.start}, at_start=False, at_end=False)

    index = {bol: add(bol)}
    queue = [bol]
    if mid not in index:
        index[mid] = add(mid)
        queue.append(mid)
    dfa.start = index[bol]
    dfa.start_mid = index[mid]

    while queue:
        current = queue.pop()
        s = index[current]
        for c in range(n_classes):
            moved = automaton.step(current, rep[c])
            if not moved:
                continue
            target = automaton.closure(moved, at_start=False, at_end=False)
            if target not in index:
                if len(dfa) >= max_states:
                    raise TooManyStates(
                        "subset construction exceeded %d states" % max_states)
                index[target] = add(target)
                queue.append(target)
            dfa.trans[s][c] = index[target]
    return dfa


def minimise(dfa):
    """Moore's algorithm: refine a partition until nothing splits.

    The dead state is not represented, so ``-1`` acts as a block of its own in
    every signature; since real block ids are non-negative it can never collide
    with one.
    """
    n = len(dfa)
    if n <= 1:
        return dfa

    initial = {}
    block = [0] * n
    for s in range(n):
        key = (dfa.accept_here[s], dfa.accept_at_end[s])
        if key not in initial:
            initial[key] = len(initial)
        block[s] = initial[key]
    n_blocks = len(initial)

    while True:
        signatures = {}
        new_block = [0] * n
        for s in range(n):
            sig = (block[s],) + tuple(
                block[t] if t >= 0 else -1 for t in dfa.trans[s])
            if sig not in signatures:
                signatures[sig] = len(signatures)
            new_block[s] = signatures[sig]
        if len(signatures) == n_blocks:
            break
        block, n_blocks = new_block, len(signatures)

    # Renumber in first-seen order so the state numbering is deterministic.
    order = []
    remap = {}
    for s in range(n):
        b = block[s]
        if b not in remap:
            remap[b] = len(order)
            order.append(s)

    out = DFA(dfa.n_classes, dfa.byte_class)
    for s in order:
        out.add_state(dfa.accept_here[s], dfa.accept_at_end[s])
    for s in order:
        i = remap[block[s]]
        for c in range(dfa.n_classes):
            t = dfa.trans[s][c]
            out.trans[i][c] = remap[block[t]] if t >= 0 else -1
    out.start = remap[block[dfa.start]]
    out.start_mid = remap[block[dfa.start_mid]]
    return out


def compile_pattern(pattern, max_states=5000, minimize=True):
    from . import parser
    automaton = N.build(parser.parse(pattern))
    dfa = build(automaton, max_states=max_states)
    return minimise(dfa) if minimize else dfa
