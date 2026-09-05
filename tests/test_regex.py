"""The front end, held against Python's own ``re``.

Every claim this project makes rests on the three tiers agreeing. If the DFA and
the backtracker disagree, the tiering experiment is measuring two different
programs and none of the numbers mean anything. So the front end is tested
differentially against a reference implementation on a large corpus rather than
against hand-written expectations, which only ever cover what the author
already thought of.

One deliberate divergence is encoded here rather than papered over: ``$`` means
end of subject, always. Python's ``$`` also matches before a trailing newline,
which is a line-oriented convenience an engine that scans bytes should not be
guessing at. The reference patterns are translated accordingly, and
:func:`test_dollar_is_end_of_subject` pins the difference so it stays a decision
rather than a bug.
"""

import random
import re

import pytest

from hp.re import backtrack as B, dfa as D, nfa as N, parser as P

PATTERNS = [
    "a", "abc", "a*", "a+", "a?", "a|b", "(ab)+", "a{2,4}", "a{3}", "a{2,}",
    "[a-z]+", "[^a-z]+", "[a-z0-9_]{1,3}", r"\d+", r"\w+", r"\s*",
    "a.c", ".*", ".+", "^abc", "abc$", "^abc$", "^a*$", "()", "(|a)",
    "(a|b)*abb", "(ab|cd)+", "a(b|c)d", "[abc]|[def]", "x*y*z*", "a*b*",
    r"\d{1,3}\.\d{1,3}", "(a+)+b", "colou?r", "(foo|bar|baz)+", "[a-c]{0,2}",
    r"[a-z0-9_.]+@[a-z0-9]+\.[a-z]{2,4}", r"^\+?\d{3,6}$", "(x|xx)+y",
    r"\.css|\.js", r"[^ ]+@[^ ]+", "a{0,0}", r"\d{4}-\d{2}-\d{2}",
]

ALPHABET = "abcdefxyz0123_. @-"


def _reference(pattern):
    """Compile the Python equivalent, mapping ``$`` to ``\\Z``.

    Only a top-level ``$`` is rewritten: inside a character class it is an
    ordinary byte, and rewriting it there would change the pattern's meaning.
    """
    out = []
    in_class = False
    i = 0
    while i < len(pattern):
        c = pattern[i]
        if c == "\\" and i + 1 < len(pattern):
            out.append(pattern[i:i + 2])
            i += 2
            continue
        if c == "[":
            in_class = True
        elif c == "]":
            in_class = False
        out.append(r"\Z" if (c == "$" and not in_class) else c)
        i += 1
    return re.compile("".join(out).encode())


@pytest.fixture(scope="module")
def subjects():
    fixed = [b"", b"a", b"ab", b"abc", b"abb", b"aabb", b"xyz", b"a.c", b"axc",
             b"123", b"12.34", b"foo", b"foobar", b"colour", b"color",
             b"a1_b2", b"   ", b"@", b"cd", b"abcd", b"aaaa", b"abababb",
             b"me@you.com", b"2026-09-02", b"style.css", b"+441234"]
    rnd = random.Random(20260905)
    for _ in range(500):
        n = rnd.randrange(0, 9)
        fixed.append(bytes(ord(rnd.choice(ALPHABET)) for _ in range(n)))
    return fixed


@pytest.mark.parametrize("pattern", PATTERNS)
def test_dfa_agrees_with_python(pattern, subjects):
    dfa = D.compile_pattern(pattern)
    ref = _reference(pattern)
    for s in subjects:
        assert dfa.fullmatch(s) is (ref.fullmatch(s) is not None), (pattern, s)


@pytest.mark.parametrize("pattern", PATTERNS)
def test_minimisation_preserves_the_language(pattern, subjects):
    raw = D.compile_pattern(pattern, minimize=False)
    small = D.compile_pattern(pattern, minimize=True)
    assert len(small) <= len(raw)
    for s in subjects:
        assert raw.fullmatch(s) == small.fullmatch(s), (pattern, s)


@pytest.mark.parametrize("pattern", PATTERNS)
def test_backtracker_agrees_with_the_dfa(pattern, subjects):
    m = B.Matcher(P.parse(pattern), limit=2_000_000)
    dfa = D.compile_pattern(pattern)
    for s in subjects:
        assert m.fullmatch(s) == dfa.fullmatch(s), (pattern, s)


@pytest.mark.parametrize("pattern", PATTERNS)
def test_search_start_agrees_with_python(pattern, subjects):
    dfa = D.compile_pattern(pattern)
    ref = _reference(pattern)
    for s in subjects:
        want = ref.search(s)
        got = dfa.search(s)
        if want is None:
            assert got is None, (pattern, s, got)
        else:
            assert got is not None and got[0] == want.start(), (pattern, s)


@pytest.mark.parametrize("pattern", PATTERNS)
def test_unanchored_wrapping_equals_search(pattern, subjects):
    """``.*P.*`` matched entirely must mean the same as ``P`` found anywhere."""
    wrapped = D.minimise(D.build(N.build(P.unanchored(P.parse(pattern)))))
    ref = _reference(pattern)
    for s in subjects:
        assert wrapped.fullmatch(s) is (ref.search(s) is not None), (pattern, s)


def test_dollar_is_end_of_subject():
    dfa = D.compile_pattern("^$")
    assert dfa.fullmatch(b"") is True
    # Python's ``$`` also matches just before a trailing newline, so it finds a
    # match in b"\n" where this engine finds none. That is the decision.
    assert dfa.fullmatch(b"\n") is False
    assert re.search(b"^$", b"\n") is not None
    assert re.search(rb"^\Z", b"\n") is None


# -- static analysis ---------------------------------------------------------
@pytest.mark.parametrize("pattern", PATTERNS)
def test_nfa_size_is_exact(pattern):
    """The static estimator must equal the construction, not approximate it.

    The tiering policy uses it to predict build cost; an estimator that drifts
    would make the policy wrong in a way that looks like the policy's fault.
    """
    node = P.unanchored(P.parse(pattern))
    assert N.size(node) == len(N.build(node)), pattern


@pytest.mark.parametrize("pattern", ["(a+)+", "^(a+)+$", r"^(\w+\s?)+$",
                                     "(x|xx)+y", "(a*)*b", "((ab)+)+"])
def test_risky_patterns_are_flagged(pattern):
    assert P.risky(P.parse(pattern)) is True


@pytest.mark.parametrize("pattern", [
    r"\d+\.\d+\.\d+\.\d+", "(foo|bar|baz)+", r"[a-z0-9._]+@[a-z0-9.]+",
    r"^[A-Z]{2}\d{2}[A-Z0-9]{4,20}$", "(ab)+c", r"\w+\s\w+", "[A-Z][a-z]+",
    "TODO|FIXME", r"password\s*[:=]\s*\S+", "a{3,6}b"])
def test_ordinary_patterns_are_not_flagged(pattern):
    """False positives cost a subset construction, so they are worth counting."""
    assert P.risky(P.parse(pattern)) is False


def test_shape_reports_counted_repetition_span():
    _, _, span = P.shape(P.parse(r"[A-Z]{2}\d{2}[A-Z0-9]{4,20}"))
    assert span == 16
    _, _, span = P.shape(P.parse(r"[a-z]+@[a-z]+"))
    assert span == 0


# -- parse errors ------------------------------------------------------------
@pytest.mark.parametrize("bad,fragment", [
    (r"a\1", "backreference"),
    (r"\bword\b", "zero-width"),
    ("a{1001}", "repetition bound"),
    ("[a", "unterminated"),
    ("(a", "expected"),
    ("*a", "nothing to repeat"),
    ("a)", "unbalanced"),
    ("a\\", "backslash"),
    ("[z-a]", "bad character range"),
    ("a{3,1}", "empty"),
])
def test_rejected_with_an_explanation(bad, fragment):
    with pytest.raises(P.ParseError) as excinfo:
        P.parse(bad)
    assert fragment in str(excinfo.value)


def test_backreference_error_says_why_not_just_what():
    with pytest.raises(P.ParseError) as excinfo:
        P.parse(r"(a)\1")
    assert "finite automaton" in str(excinfo.value)


# -- bounds ------------------------------------------------------------------
def test_subset_construction_is_bounded():
    """Exponential blowup is reported, not suffered."""
    pattern = r"[ab]*a[ab]{12}"
    with pytest.raises(D.TooManyStates):
        D.compile_pattern(pattern, max_states=64)
    assert len(D.compile_pattern(pattern, max_states=100000)) > 64


def test_backtracking_step_budget_is_enforced():
    m = B.Matcher(P.parse("(a+)+b"), limit=10000)
    with pytest.raises(B.StepLimit):
        m.fullmatch(b"a" * 40)


def test_deep_subject_exhausts_the_stack_not_the_process():
    """A backtracker dies of depth on long inputs. That is caught, not fatal."""
    m = B.Matcher(P.unanchored(P.parse("abc")))
    with pytest.raises(B.Exhausted):
        m.fullmatch(b"x" * 40000)


def test_zero_width_repetition_terminates():
    dfa = D.compile_pattern("(a*)*")
    assert dfa.fullmatch(b"") is True
    assert dfa.fullmatch(b"aaa") is True
    m = B.Matcher(P.parse("(a*)*"), limit=100000)
    assert m.fullmatch(b"aaaa") is True


# -- alphabet compression ----------------------------------------------------
def test_byte_classes_collapse_indistinguishable_bytes():
    dfa = D.compile_pattern("[a-z]+")
    # Two classes: the letters, and everything else.
    assert dfa.n_classes == 2
    assert dfa.byte_class[ord("a")] == dfa.byte_class[ord("z")]
    assert dfa.byte_class[ord("a")] != dfa.byte_class[ord("0")]


def test_byte_classes_are_stable_across_runs():
    """Class numbering feeds the generated code; it must not depend on hashing."""
    first = D.compile_pattern(r"[a-z]+@[a-z]+\.[a-z]+").byte_class
    for _ in range(5):
        assert D.compile_pattern(r"[a-z]+@[a-z]+\.[a-z]+").byte_class == first
