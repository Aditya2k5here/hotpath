"""Traces to replay: sequences of ``(pattern, subject)`` calls.

A tiering policy can only be judged against the traffic it will see, and the
right answer changes completely between one shape of traffic and another. These
six are chosen so that no single policy wins all of them -- if one did, the
question would not be worth asking.

* ``one-shot`` -- four hundred patterns, one call each. Every promotion is pure
  loss. The workload where the correct policy is to do nothing.
* ``log-lines`` -- a handful of patterns over many short lines. The classic
  ``grep``-shaped load, and the one where compiling obviously pays.
* ``validate`` -- form-validation patterns over very short strings. High call
  count, negligible bytes: the case where a call-counting policy promotes and a
  byte-counting policy does not, and one of them is wrong.
* ``bulk`` -- three patterns over quarter-megabyte blobs. Few calls, enormous
  volume: the exact reverse of ``validate``.
* ``adversarial`` -- patterns whose backtracking cost explodes on inputs that
  look ordinary. Promotion here is not an optimisation, it is the difference
  between answering and not.
* ``zipf`` -- sixty patterns with realistically skewed popularity and mixed
  subject sizes. The mixture, and the one whose result generalises.

Everything is seeded, so a trace is a value: two runs of the same workload are
the same list of calls, and a policy comparison is not a comparison of luck.
"""

import random

LOG_TEMPLATES = [
    b'%s - - [02/Sep/2026:%02d:%02d:%02d] "GET /api/v1/%s?id=%d HTTP/1.1" 200 %d',
    b'%s - - [02/Sep/2026:%02d:%02d:%02d] "POST /api/v1/%s HTTP/1.1" 500 %d',
    b'%s - - [02/Sep/2026:%02d:%02d:%02d] "GET /static/%s.css HTTP/1.1" 304 %d',
]
PATHS = [b"users", b"orders", b"sessions", b"health", b"metrics", b"assets"]

LOG_PATTERNS = [
    r"\d+\.\d+\.\d+\.\d+",
    r'"(GET|POST|PUT|DELETE) /api/v\d+/\w+',
    r" (4|5)\d\d ",
    r"\.css|\.js|\.png",
    r"id=\d+",
]

VALIDATE_PATTERNS = [
    r"^[a-z0-9_.]+@[a-z0-9]+\.[a-z]{2,4}$",
    r"^\d{4}-\d{2}-\d{2}$",
    r"^\+?\d{7,15}$",
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$",
    r"^#[0-9a-fA-F]{6}$",
    r"^[A-Z]{2}\d{2}[A-Z0-9]{4,20}$",
]

BULK_PATTERNS = [
    r"BEGIN [A-Z ]+ KEY",
    r"password\s*[:=]\s*\S+",
    r"[a-z0-9._]+@[a-z0-9.]+\.[a-z]{2,6}",
]

# Patterns whose backtracking cost is exponential in the length of a run of
# characters that *almost* matches. Every one of these is a shape that has
# taken down a production service somewhere.
ADVERSARIAL_PATTERNS = [
    r"^(a+)+$",
    r"^(\w+\s?)+$",
    r"^(x|xx)+y$",
]

TAIL_PATTERNS = [
    r"TODO|FIXME", r"[A-Z]{3,}", r"\d{4,}", r"foo|bar|baz", r"a{3,6}b",
    r"[aeiou]{2,}", r"(ab)+c", r"x[0-9]*y", r"[^ ]+@[^ ]+", r"\s{2,}",
    r"v\d+\.\d+\.\d+", r"0x[0-9a-f]+", r"[A-Z][a-z]+ [A-Z][a-z]+",
    r"(https?)://", r"\.\.\/", r"NULL|nil|None", r"[<>]{2}", r"%[sd]",
    r"__\w+__", r"[0-9]{3}-[0-9]{2}-[0-9]{4}",
]

WORDS = (b"the quick brown fox jumps over lazy dog alpha beta gamma delta "
         b"epsilon zeta eta theta iota kappa lambda mu nu xi omicron pi rho "
         b"sigma tau upsilon phi chi psi omega TODO FIXME NULL None 0xdeadbeef "
         b"v1.20.3 https://example.com user@example.com").split()


def _text(rnd, n_bytes):
    out = []
    total = 0
    while total < n_bytes:
        w = rnd.choice(WORDS)
        out.append(w)
        total += len(w) + 1
    return b" ".join(out)[:n_bytes]


def _log_line(rnd):
    tmpl = rnd.choice(LOG_TEMPLATES)
    ip = b"%d.%d.%d.%d" % tuple(rnd.randrange(1, 255) for _ in range(4))
    args = [ip, rnd.randrange(24), rnd.randrange(60), rnd.randrange(60),
            rnd.choice(PATHS)]
    if b"id=%d" in tmpl:
        args.append(rnd.randrange(100000))
    args.append(rnd.randrange(10000))
    return tmpl % tuple(args)


def one_shot(seed=1, n=400):
    rnd = random.Random(seed)
    trace = []
    for i in range(n):
        # Distinct but syntactically ordinary patterns: a pattern cache would
        # not help either, which is the point.
        pat = r"%s\d{%d}[a-z]{1,%d}" % (
            "".join(rnd.choice("abcdefgh") for _ in range(3)),
            rnd.randrange(1, 4), rnd.randrange(2, 6))
        trace.append((pat, _text(rnd, 40)))
    return trace


def log_lines(seed=2, lines=2500):
    rnd = random.Random(seed)
    subjects = [_log_line(rnd) for _ in range(lines)]
    trace = []
    for s in subjects:
        for pat in LOG_PATTERNS:
            trace.append((pat, s))
    rnd.shuffle(trace)
    return trace


def validate(seed=3, per_pattern=1800):
    rnd = random.Random(seed)
    samples = {
        VALIDATE_PATTERNS[0]: [b"ada@example.com", b"not-an-email", b"a@b.co"],
        VALIDATE_PATTERNS[1]: [b"2026-09-02", b"2026-9-2", b"20260902"],
        VALIDATE_PATTERNS[2]: [b"+441234567890", b"12345", b"07700900123"],
        VALIDATE_PATTERNS[3]: [b"3f2504e0-4f89-11d3-9a0c-0305e82c3301", b"nope"],
        VALIDATE_PATTERNS[4]: [b"#ff8800", b"#gg0000", b"#fff"],
        VALIDATE_PATTERNS[5]: [b"GB29NWBK60161331926819", b"xx"],
    }
    trace = []
    for pat in VALIDATE_PATTERNS:
        pool = samples[pat]
        for _ in range(per_pattern):
            trace.append((pat, rnd.choice(pool)))
    rnd.shuffle(trace)
    return trace


def bulk(seed=4, blobs=6, size=192 * 1024):
    rnd = random.Random(seed)
    docs = [_text(rnd, size) for _ in range(blobs)]
    trace = []
    for d in docs:
        for pat in BULK_PATTERNS:
            trace.append((pat, d))
    return trace


def adversarial(seed=5, calls=90):
    rnd = random.Random(seed)
    trace = []
    for i in range(calls):
        pat = ADVERSARIAL_PATTERNS[i % len(ADVERSARIAL_PATTERNS)]
        # Mostly benign lengths; occasionally the length that detonates.
        n = rnd.choice([6, 8, 10, 12, 14, 18, 22, 26])
        if pat == r"^(a+)+$":
            s = b"a" * n + (b"!" if rnd.random() < 0.5 else b"")
        elif pat == r"^(x|xx)+y$":
            s = b"x" * n + (b"z" if rnd.random() < 0.5 else b"y")
        else:
            s = b" ".join([b"word"] * (n // 2)) + (b"!" if rnd.random() < 0.5
                                                  else b"")
        trace.append((pat, s))
    rnd.shuffle(trace)
    return trace


def zipf(seed=6, calls=6000, n_patterns=40, alpha=1.1):
    """Skewed popularity and mixed sizes -- the shape real traffic has.

    Pattern *i* is chosen with probability proportional to ``1/(i+1)**alpha``,
    so a few patterns carry most of the calls and a long tail is seen once or
    twice. A policy that promotes on first sight pays for the whole tail; one
    that waits too long never promotes the head.
    """
    rnd = random.Random(seed)
    pool = (TAIL_PATTERNS + LOG_PATTERNS + BULK_PATTERNS)[:n_patterns]
    weights = [1.0 / (i + 1) ** alpha for i in range(len(pool))]
    total = sum(weights)
    cum, acc = [], 0.0
    for w in weights:
        acc += w / total
        cum.append(acc)

    sizes = [24, 64, 200, 800, 4096, 16384]
    size_w = [0.35, 0.25, 0.2, 0.12, 0.06, 0.02]
    size_cum, acc = [], 0.0
    for w in size_w:
        acc += w
        size_cum.append(acc)

    texts = {n: [_text(rnd, n) for _ in range(4)] for n in sizes}
    trace = []
    for _ in range(calls):
        u = rnd.random()
        pat = pool[next(i for i, c in enumerate(cum) if u <= c)]
        u = rnd.random()
        size = sizes[next(i for i, c in enumerate(size_cum) if u <= c)]
        trace.append((pat, rnd.choice(texts[size])))
    return trace


def mixture(seed=7, hot_calls=1500, n_cold=200):
    """A fixed hot core plus ``n_cold`` patterns used exactly once.

    The single axis this project turns on. Compiling a pattern is worth it in
    proportion to how often the pattern is reused, so the question "is a tiering
    policy worth having" is really the question "what fraction of your patterns
    are used once". This workload makes that fraction a dial: the hot core never
    changes, and ``n_cold`` cold patterns are added around it. Sweeping it finds
    the point where compiling everything on sight stops being the right answer,
    which is a number rather than an opinion.
    """
    rnd = random.Random(seed)
    lines = [_log_line(rnd) for _ in range(hot_calls)]
    trace = []
    for s_ in lines:
        for pat in LOG_PATTERNS:
            trace.append((pat, s_))
    for i in range(n_cold):
        pat = r"%s\d{%d}[a-z]{1,%d}" % (
            "".join(rnd.choice("abcdefgh") for _ in range(3)),
            rnd.randrange(1, 4), rnd.randrange(2, 6))
        trace.append((pat, _text(rnd, 40)))
    rnd.shuffle(trace)
    return trace


COLD_COUNTS = [0, 50, 200, 800, 2000, 5000]

WORKLOADS = {
    "one-shot": one_shot,
    "log-lines": log_lines,
    "validate": validate,
    "bulk": bulk,
    "adversarial": adversarial,
    "zipf": zipf,
}
for _n in COLD_COUNTS:
    WORKLOADS["mix-%d" % _n] = (lambda n: lambda: mixture(n_cold=n))(_n)


def describe(trace):
    pats = {}
    for pat, subj in trace:
        c, b = pats.get(pat, (0, 0))
        pats[pat] = (c + 1, b + len(subj))
    return {
        "calls": len(trace),
        "patterns": len(pats),
        "bytes": sum(b for _, b in pats.values()),
        "max_calls_per_pattern": max(c for c, _ in pats.values()),
        "mean_subject_bytes": sum(len(s) for _, s in trace) / len(trace),
    }
