"""Regular expression syntax -> AST.

The supported language is deliberately the *regular* one: literals, character
classes, concatenation, alternation, repetition, anchors and grouping. No
backreferences, no lookaround.

That is not a shortcut, it is the premise. Backreferences make a pattern
language strictly more powerful than a finite automaton, so an engine that
supports them cannot compile to a DFA and cannot give a linear-time guarantee.
Every catastrophic-backtracking outage you have read about is that trade being
made silently. Leaving them out is what makes the rest of this project possible.

Character classes are represented as 256-bit sets over bytes, computed at parse
time, so everything downstream. NFA, DFA, code generator. Deals with one
uniform notion of "which input bytes take this edge" and never re-interprets
syntax.
"""


class ParseError(Exception):
    pass


# --- AST -------------------------------------------------------------------
class Node(object):
    __slots__ = ()


class Empty(Node):
    """Matches the empty string."""
    __slots__ = ()

    def __repr__(self):
        return "Empty()"


class Class(Node):
    """A set of acceptable bytes, held as a frozenset of ints 0..255."""
    __slots__ = ("bytes_", "label")

    def __init__(self, bytes_, label=None):
        self.bytes_ = frozenset(bytes_)
        self.label = label

    def __repr__(self):
        if self.label:
            return "Class(%s)" % self.label
        if len(self.bytes_) == 1:
            return "Lit(%r)" % chr(next(iter(self.bytes_)))
        return "Class(%d bytes)" % len(self.bytes_)


class Concat(Node):
    __slots__ = ("parts",)

    def __init__(self, parts):
        self.parts = list(parts)

    def __repr__(self):
        return "Concat(%r)" % (self.parts,)


class Alt(Node):
    __slots__ = ("options",)

    def __init__(self, options):
        self.options = list(options)

    def __repr__(self):
        return "Alt(%r)" % (self.options,)


class Repeat(Node):
    """``min`` to ``max`` copies; ``max=None`` means unbounded."""
    __slots__ = ("node", "min", "max")

    def __init__(self, node, min_, max_):
        self.node = node
        self.min = min_
        self.max = max_

    def __repr__(self):
        return "Repeat(%r, %s, %s)" % (self.node, self.min, self.max)


class Anchor(Node):
    """``^`` or ``$``."""
    __slots__ = ("kind",)

    def __init__(self, kind):
        self.kind = kind

    def __repr__(self):
        return "Anchor(%r)" % self.kind


# --- byte sets --------------------------------------------------------------
ALL_BYTES = frozenset(range(256))
DIGITS = frozenset(range(0x30, 0x3A))
WORD = frozenset(list(range(0x30, 0x3A)) + list(range(0x41, 0x5B)) +
                 list(range(0x61, 0x7B)) + [0x5F])
SPACE = frozenset([0x20, 0x09, 0x0A, 0x0B, 0x0C, 0x0D])
DOT = ALL_BYTES - frozenset([0x0A])           # `.` excludes newline, as usual

ESCAPES = {
    "d": (DIGITS, r"\d"), "D": (ALL_BYTES - DIGITS, r"\D"),
    "w": (WORD, r"\w"), "W": (ALL_BYTES - WORD, r"\W"),
    "s": (SPACE, r"\s"), "S": (ALL_BYTES - SPACE, r"\S"),
}
LITERAL_ESCAPES = {"n": 0x0A, "t": 0x09, "r": 0x0D, "f": 0x0C, "v": 0x0B,
                   "0": 0x00, "a": 0x07, "b": 0x08}

MAX_REPEAT = 1000


class Parser(object):
    def __init__(self, pattern):
        self.src = pattern
        self.i = 0

    # -- helpers ------------------------------------------------------------
    def eof(self):
        return self.i >= len(self.src)

    def peek(self):
        return None if self.eof() else self.src[self.i]

    def take(self):
        c = self.src[self.i]
        self.i += 1
        return c

    def expect(self, c):
        if self.eof() or self.src[self.i] != c:
            raise ParseError("expected %r at position %d in %r"
                             % (c, self.i, self.src))
        self.i += 1

    # -- grammar ------------------------------------------------------------
    def parse(self):
        node = self.alternation()
        if not self.eof():
            # A leftover ')' always means an unmatched one: concatenation stops
            # at ')' expecting a group to close around it, so reaching here with
            # one left is the diagnosis, not a surprise.
            if self.peek() == ")":
                raise ParseError("unbalanced ) at position %d in %r"
                                 % (self.i, self.src))
            raise ParseError("unexpected %r at position %d in %r"
                             % (self.peek(), self.i, self.src))
        return node

    def alternation(self):
        options = [self.concatenation()]
        while self.peek() == "|":
            self.take()
            options.append(self.concatenation())
        return options[0] if len(options) == 1 else Alt(options)

    def concatenation(self):
        parts = []
        while not self.eof() and self.peek() not in ("|", ")"):
            parts.append(self.repetition())
        if not parts:
            return Empty()
        return parts[0] if len(parts) == 1 else Concat(parts)

    def repetition(self):
        node = self.atom()
        while True:
            c = self.peek()
            if c == "*":
                self.take(); node = Repeat(node, 0, None)
            elif c == "+":
                self.take(); node = Repeat(node, 1, None)
            elif c == "?":
                self.take(); node = Repeat(node, 0, 1)
            elif c == "{":
                save = self.i
                bounds = self.try_counted()
                if bounds is None:
                    self.i = save
                    return node
                node = Repeat(node, bounds[0], bounds[1])
            else:
                return node

    def try_counted(self):
        """``{m}``, ``{m,}`` or ``{m,n}``; returns None if it is a literal brace."""
        self.expect("{")
        digits = ""
        while not self.eof() and self.peek().isdigit():
            digits += self.take()
        if not digits:
            return None
        low = int(digits)
        high = low
        if self.peek() == ",":
            self.take()
            digits2 = ""
            while not self.eof() and self.peek().isdigit():
                digits2 += self.take()
            high = int(digits2) if digits2 else None
        if self.peek() != "}":
            return None
        self.take()
        if high is not None and high < low:
            raise ParseError("{%d,%d} is empty in %r" % (low, high, self.src))
        if low > MAX_REPEAT or (high is not None and high > MAX_REPEAT):
            raise ParseError("repetition bound above %d in %r"
                             % (MAX_REPEAT, self.src))
        return low, high

    def atom(self):
        if self.eof():
            raise ParseError("unexpected end of pattern in %r" % self.src)
        c = self.take()
        if c == "(":
            # Groups exist for precedence only. There is no capture: a DFA has
            # no way to report where a submatch started, and pretending
            # otherwise would be a lie the code generator could not keep.
            if self.src.startswith("?:", self.i):
                self.i += 2
            node = self.alternation()
            self.expect(")")
            return node
        if c == "[":
            return self.char_class()
        if c == ".":
            return Class(DOT, ".")
        if c == "^":
            return Anchor("^")
        if c == "$":
            return Anchor("$")
        if c == "\\":
            return self.escape()
        if c in "*+?":
            raise ParseError("nothing to repeat at position %d in %r"
                             % (self.i - 1, self.src))
        if c == ")":
            raise ParseError("unbalanced ) at position %d in %r"
                             % (self.i - 1, self.src))
        return Class([ord(c)], repr(c))

    def escape(self):
        if self.eof():
            raise ParseError("pattern ends with a backslash: %r" % self.src)
        c = self.take()
        if c in ("A", "Z"):
            # ``\A`` and ``\Z`` are what ``^`` and ``$`` mean here anyway: this
            # engine anchors to the subject, never to a line.
            return Anchor("^" if c == "A" else "$")
        if c in ("b", "B", "G"):
            raise ParseError(
                "\\%s in %r: word boundaries and similar zero-width assertions "
                "depend on the two bytes either side of a position, which a "
                "byte-at-a-time automaton does not have. Use an explicit "
                "character class instead." % (c, self.src))
        if c in ESCAPES:
            bytes_, label = ESCAPES[c]
            return Class(bytes_, label)
        if c in LITERAL_ESCAPES:
            return Class([LITERAL_ESCAPES[c]], "\\" + c)
        if c.isdigit():
            raise ParseError(
                "backreference \\%s in %r: this engine compiles to a finite "
                "automaton, which cannot express backreferences" % (c, self.src))
        return Class([ord(c)], repr(c))

    def char_class(self):
        negated = False
        if self.peek() == "^":
            self.take()
            negated = True
        members = set()
        first = True
        while True:
            if self.eof():
                raise ParseError("unterminated [ in %r" % self.src)
            c = self.take()
            if c == "]" and not first:
                break
            first = False
            if c == "\\":
                if self.eof():
                    raise ParseError("pattern ends with a backslash: %r" % self.src)
                e = self.take()
                if e in ESCAPES:
                    members |= ESCAPES[e][0]
                    continue
                lo = LITERAL_ESCAPES.get(e, ord(e))
            else:
                lo = ord(c)
            # a range, but only when the '-' is not the last character
            if self.peek() == "-" and self.i + 1 < len(self.src) \
                    and self.src[self.i + 1] != "]":
                self.take()
                hc = self.take()
                if hc == "\\":
                    he = self.take()
                    hi = LITERAL_ESCAPES.get(he, ord(he))
                else:
                    hi = ord(hc)
                if hi < lo:
                    raise ParseError("bad character range in %r" % self.src)
                members |= set(range(lo, hi + 1))
            else:
                members.add(lo)
        if negated:
            members = set(ALL_BYTES) - members
        if not members:
            raise ParseError("empty character class in %r" % self.src)
        return Class(members)


def parse(pattern):
    return Parser(pattern).parse()


def shape(node):
    """``(nodes, distinct byte classes, counted-repetition span)`` for an AST.

    A tiering policy has to guess what a subset construction will cost *before*
    paying for it. Pattern length is the obvious guess and a bad one. ``.*``
    and ``[^\n]{500}`` are the same length and nothing like the same cost. The
    parsed shape is nearly as cheap to obtain (tier 0 has already parsed) and
    describes the automaton far better: node count bounds the NFA size, and the
    number of distinct classes bounds the alphabet the powerset runs over.
    """
    seen = set()
    count = [0]
    span = [0]

    def walk(n):
        count[0] += 1
        if isinstance(n, Class):
            seen.add(n.bytes_)
        elif isinstance(n, Concat):
            for part in n.parts:
                walk(part)
        elif isinstance(n, Alt):
            for opt in n.options:
                walk(opt)
        elif isinstance(n, Repeat):
            # The span of a counted repetition, the number of *optional*
            # copies. Is the one syntactic feature that predicts a subset
            # construction blowing up, because each optional copy is a bit in
            # the state set the powerset has to track separately.
            if n.max is not None:
                span[0] += n.max - n.min
            walk(n.node)

    walk(node)
    return count[0], len(seen), span[0]


def min_len(node):
    """Shortest string this node can match. Used only by :func:`risky`."""
    if isinstance(node, Empty) or isinstance(node, Anchor):
        return 0
    if isinstance(node, Class):
        return 1
    if isinstance(node, Concat):
        return sum(min_len(p) for p in node.parts)
    if isinstance(node, Alt):
        return min(min_len(o) for o in node.options)
    if isinstance(node, Repeat):
        return node.min * min_len(node.node)
    return 0                                            # pragma: no cover


def _has_unbounded(node):
    if isinstance(node, Repeat):
        return node.max is None or _has_unbounded(node.node)
    if isinstance(node, Concat):
        return any(_has_unbounded(p) for p in node.parts)
    if isinstance(node, Alt):
        return any(_has_unbounded(o) for o in node.options)
    return False


def risky(node):
    """Does this pattern have a shape that makes backtracking explode?

    Two rules, both syntactic and both cheap:

    * **Nested unbounded repetition**: ``(a+)+``. The outer loop can split a
      run of ``a``s among its iterations in exponentially many ways, and a
      backtracking engine tries all of them before concluding there is no match.
    * **Unbounded repetition over an ambiguous alternation**: ``(x|xx)+``.
      Branches of differing minimum length under a star give the same splitting
      freedom without any nesting to see.

    This is a heuristic and it is deliberately one-sided: it may call a safe
    pattern risky, and the cost of that is one subset construction. It exists
    because the evaluation showed something no amount of policy tuning fixed,
a pattern that can detonate must not be run on the backtracker even once,
    and by the time a reactive policy has evidence, the detonation has happened.
    """
    if isinstance(node, Repeat):
        if node.max is None:
            if _has_unbounded(node.node):
                return True
            inner = node.node
            if isinstance(inner, Alt):
                lengths = {min_len(o) for o in inner.options}
                if len(lengths) > 1:
                    return True
        return risky(node.node)
    if isinstance(node, Concat):
        return any(risky(p) for p in node.parts)
    if isinstance(node, Alt):
        return any(risky(o) for o in node.options)
    return False


def unanchored(node):
    """``P`` -> ``.*P.*``, so "matches anywhere" becomes "matches entirely".

    This is how the engine answers *search* questions with a single left-to-
    right pass and no restarting. Restarting at every offset is what makes naive
    DFA search quadratic; wrapping the pattern instead pushes the work into the
    automaton, where it costs states rather than time.

    Anchors survive the transformation, which is the part worth checking. ``^``
    is an edge that the epsilon closure only admits at offset zero, and the
    leading ``.*`` has consumed nothing there, so ``^abc`` still cannot match at
    offset 5. ``$`` is admitted only in the end-of-subject closure, which is
    consulted for the accept bit and never for a transition, so the trailing
    ``.*`` cannot smuggle bytes in after it.

    The wildcard used here is *every* byte, newline included. Unlike ``.`` in
    the pattern language. A search is over the whole subject, not over lines.
    """
    any_star = Repeat(Class(ALL_BYTES, "any"), 0, None)
    return Concat([any_star, node, any_star])
