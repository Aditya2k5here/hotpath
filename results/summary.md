# Results

Everything below was produced on one machine by `make experiment`, and every
number is wall-clock nanoseconds measured on that machine rather than a model.
The raw JSON each table is built from is in this directory; `tools/summarise.py`
only formats it.

## 1. Does compiling to machine code actually help?

Generated x86-64 against CPython's own regex engine, on subjects where neither
finds a match (an early match makes both engines return early and measures
nothing).

| pattern | subject | CPython `re` | hotpath JIT | speedup |
|---|---:|---:|---:|---:|
| `[a-z0-9._]+@[a-z0-9.]+\.[a-z]{2,6}` | 64 B | 17.77 ns/B | 6.20 ns/B | **2.9×** |
| `[a-z0-9._]+@[a-z0-9.]+\.[a-z]{2,6}` | 1 KB | 17.42 ns/B | 0.96 ns/B | **18.2×** |
| `[a-z0-9._]+@[a-z0-9.]+\.[a-z]{2,6}` | 64 KB | 17.52 ns/B | 0.54 ns/B | **32.7×** |
| `[a-z0-9._]+@[a-z0-9.]+\.[a-z]{2,6}` | 1 MB | 17.51 ns/B | 0.56 ns/B | **31.2×** |
| `\d+\.\d+\.\d+\.\d+` | 64 B | 10.99 ns/B | 6.57 ns/B | **1.7×** |
| `\d+\.\d+\.\d+\.\d+` | 1 KB | 10.83 ns/B | 0.72 ns/B | **15.1×** |
| `\d+\.\d+\.\d+\.\d+` | 64 KB | 10.84 ns/B | 0.32 ns/B | **33.7×** |
| `\d+\.\d+\.\d+\.\d+` | 1 MB | 10.66 ns/B | 0.31 ns/B | **34.4×** |
| `error\|warning\|fatal` | 64 B | 8.91 ns/B | 6.52 ns/B | **1.4×** |
| `error\|warning\|fatal` | 1 KB | 7.02 ns/B | 1.10 ns/B | **6.4×** |
| `error\|warning\|fatal` | 64 KB | 7.15 ns/B | 0.75 ns/B | **9.5×** |
| `error\|warning\|fatal` | 1 MB | 7.38 ns/B | 0.73 ns/B | **10.1×** |
| `(a\|b)*abb` | 64 B | 29.16 ns/B | 6.48 ns/B | **4.5×** |
| `(a\|b)*abb` | 1 KB | 30.08 ns/B | 1.03 ns/B | **29.2×** |
| `(a\|b)*abb` | 64 KB | 29.59 ns/B | 0.64 ns/B | **46.4×** |
| `(a\|b)*abb` | 1 MB | 30.24 ns/B | 0.64 ns/B | **47.6×** |
| `^GET /api/v\d+/\w+` | 64 B | 1.60 ns/B | 5.95 ns/B | **0.3×** |
| `^GET /api/v\d+/\w+` | 1 KB | 0.09 ns/B | 0.37 ns/B | **0.2×** |
| `^GET /api/v\d+/\w+` | 64 KB | 0.00 ns/B | 0.01 ns/B | **0.3×** |
| `^GET /api/v\d+/\w+` | 1 MB | 0.00 ns/B | 0.00 ns/B | **0.4×** |

The JIT's advantage grows with subject length because its per-call overhead is
fixed (a `ctypes` call, ~0.6 µs) while its per-byte cost is ~0.6 ns. Below a
kilobyte the overhead dominates and the win is small; above 64 KB it is the
whole story. This is exactly the shape that makes a tiering policy necessary
rather than decorative.

The last pattern is the one that loses, and it is worth reading rather than
skipping. `^GET /api/v\d+/\w+` is anchored, so both engines answer after a
byte or two; what is being compared is no longer scanning speed but the cost of
getting into and out of the matcher, and a `ctypes` call is far more expensive
than CPython dispatching to its own C. Compiling a pattern like this to machine
code is a waste no matter how good the code is — which is the argument for the
tiering policy stated as a benchmark row rather than as a claim.

## 2. What did it cost to build, and what did that buy?

| tier | enter | cost per byte | notes |
|---|---:|---:|---|
| backtracking interpreter | free (parse only) | ~1.2 µs per step | unbounded; dies of stack depth past ~30 KB |
| interpreted DFA | ~0.3–2 ms | 29.6 ns | linear, no matter the pattern |
| generated x86-64 | + ~0.2–1.3 ms | 0.63 ns | linear, ~47× the interpreted DFA |

Build costs are measured per pattern, including `mmap` and `mprotect`.


## 3. The workloads

Twelve traces. The first six are shapes real traffic takes; the `mix-N` family
is one shape with a dial on it — a fixed hot core of five patterns over log
lines, plus *N* patterns used exactly once — which is the axis the whole
question turns on.

| workload | calls | patterns | bytes | mean subject | tier-0 calls censored |
|---|---:|---:|---:|---:|---:|
| `one-shot` | 400 | 390 | 15 KB | 39 B | 0 (0%) |
| `validate` | 10800 | 6 | 118 KB | 11 B | 0 (0%) |
| `log-lines` | 12500 | 5 | 1 MB | 84 B | 0 (0%) |
| `bulk` | 18 | 3 | 3 MB | 196608 B | 18 (100%) |
| `adversarial` | 90 | 3 | 1 KB | 21 B | 24 (27%) |
| `zipf` | 6000 | 28 | 4 MB | 726 B | 1178 (20%) |
| `mix-0` | 7500 | 5 | 616 KB | 84 B | 0 (0%) |
| `mix-50` | 7550 | 54 | 618 KB | 83 B | 0 (0%) |
| `mix-200` | 7700 | 199 | 624 KB | 83 B | 0 (0%) |
| `mix-800` | 8300 | 760 | 647 KB | 79 B | 0 (0%) |
| `mix-2000` | 9500 | 1704 | 694 KB | 74 B | 0 (0%) |
| `mix-5000` | 12500 | 3412 | 811 KB | 66 B | 0 (0%) |

**Censoring, and which way it biases the result.** The backtracking tier runs
under a 50,000-step budget; a call that exhausts it is recorded at the budget's
cost, which is a *lower bound* on what it would really have cost. That
understates how much promotion was worth, which shrinks the denominator of every
"recovered" figure. Where censoring is heavy — `bulk` is 100%, `zipf` about 20% —
the multiple-of-optimum column is the honest one and the recovered column should
be ignored.

## 4. Every policy against the hindsight optimum

The oracle sees the whole trace in advance and picks, per pattern, the exact two
indices at which to build the DFA and the machine code. It is the true optimum,
not a good schedule: tiers are monotone, so a schedule is two indices, and the
minimum over all of them falls out of prefix sums in linear time
(`hp/eval/oracle.py`, checked against exhaustive search in the test suite).

Cost as a multiple of that optimum — 1.00 is perfect, lower is better:


| policy | one-shot | validate | log-lines | bulk | adversarial | zipf | mix-0 | mix-50 | mix-200 | mix-800 | mix-2000 | mix-5000 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `never` | 1.00 | 28.51 | 132.73 | 13.20 | 82.76 | 5.07 | 92.66 | 72.36 | 43.99 | 18.44 | 8.32 | 4.05 |
| `always-dfa` | 9.96 | 1.00 | 2.23 | 12.84 | 1.00 | 3.71 | 1.79 | 3.70 | 6.04 | 7.82 | 7.53 | 6.03 |
| `always-jit` | 12.80 | 1.49 | 1.00 | 1.21 | 2.22 | 1.00 | 1.00 | 3.71 | 7.21 | 9.89 | 9.72 | 7.90 |
| `calls(2,4)` | 1.00 | 1.57 | 1.15 | 9.50 | 45.06 | 1.47 | 1.16 | 1.14 | 1.08 | 1.05 | 1.14 | 1.52 |
| `calls(10,50)` | 1.00 | 1.69 | 1.58 | 13.20 | 69.87 | 2.77 | 1.70 | 1.59 | 1.34 | 1.14 | 1.08 | 1.04 |
| `calls(100,500)` | 1.00 | 3.26 | 6.39 | 13.20 | 82.76 | 4.41 | 7.36 | 6.43 | 4.08 | 2.21 | 1.54 | 1.22 |
| `bytes(4k,64k)` | 1.00 | 6.52 | 3.95 | 1.21 | 82.76 | 3.04 | 4.50 | 3.87 | 2.66 | 1.67 | 1.30 | 1.13 |
| `bytes(64k,1M)` | 1.00 | 28.51 | 42.65 | 11.15 | 82.76 | 4.99 | 49.16 | 38.38 | 23.94 | 10.02 | 4.90 | 2.63 |
| `amortised(K=0.25)` | 1.12 | 1.23 | 1.12 | 3.20 | 43.83 | 1.11 | 1.14 | 1.17 | 1.11 | 1.28 | 1.69 | 2.31 |
| `amortised(K=0.5)` | 1.00 | 1.25 | 1.21 | 3.20 | 43.83 | 1.19 | 1.23 | 1.20 | 1.11 | 1.06 | 1.07 | 1.27 |
| `amortised(K=1)` | 1.00 | 1.46 | 1.33 | 3.20 | 66.98 | 1.38 | 1.40 | 1.32 | 1.19 | 1.08 | 1.05 | 1.03 |
| `amortised(K=2)` | 1.00 | 1.86 | 1.66 | 3.20 | 66.98 | 1.55 | 1.79 | 1.61 | 1.36 | 1.14 | 1.08 | 1.04 |
| `amortised(K=4)` | 1.00 | 2.65 | 2.24 | 3.20 | 66.98 | 2.00 | 2.34 | 2.08 | 1.63 | 1.26 | 1.12 | 1.06 |
| `amortised(K=8)` | 1.00 | 4.32 | 3.20 | 3.20 | 66.98 | 2.49 | 3.48 | 2.94 | 2.14 | 1.47 | 1.21 | 1.09 |
| `adaptive(K=0.25)` | 1.21 | 1.21 | 1.14 | 3.20 | 43.83 | 1.16 | 1.13 | 1.10 | 1.07 | 1.04 | 1.31 | 1.69 |
| `adaptive(K=0.5)` | 1.00 | 1.15 | 1.15 | 3.20 | 43.83 | 1.38 | 1.25 | 1.13 | 1.09 | 1.04 | 1.04 | 1.13 |
| `adaptive(K=1)` | 1.00 | 1.26 | 1.26 | 3.20 | 66.98 | 1.63 | 1.27 | 1.25 | 1.16 | 1.06 | 1.05 | 1.02 |
| `adaptive(K=2)` | 1.00 | 1.52 | 1.50 | 3.20 | 66.98 | 2.13 | 1.52 | 1.42 | 1.31 | 1.10 | 1.07 | 1.03 |
| `adaptive(K=4)` | 1.00 | 1.98 | 1.87 | 3.20 | 66.98 | 2.78 | 1.84 | 1.77 | 1.59 | 1.20 | 1.12 | 1.05 |
| `adaptive(K=8)` | 1.00 | 2.91 | 3.01 | 4.50 | 66.98 | 3.77 | 2.51 | 2.37 | 2.02 | 1.37 | 1.18 | 1.07 |
| `guarded/adaptive(K=0.25)` | 1.21 | 1.21 | 1.14 | 1.21 | 1.00 | 1.16 | 1.13 | 1.10 | 1.07 | 1.04 | 1.31 | 1.69 |
| `guarded/adaptive(K=0.5)` | 1.00 | 1.15 | 1.15 | 1.21 | 1.00 | 1.38 | 1.25 | 1.13 | 1.09 | 1.04 | 1.04 | 1.13 |
| `guarded/adaptive(K=1)` | 1.00 | 1.26 | 1.26 | 1.21 | 1.00 | 1.63 | 1.27 | 1.25 | 1.16 | 1.06 | 1.05 | 1.02 |
| `adaptive+trip(K=0.25)` | 1.21 | 1.21 | 1.14 | 3.20 | 43.83 | 1.16 | 1.13 | 1.10 | 1.07 | 1.04 | 1.31 | 1.69 |
| `adaptive+trip(K=0.5)` | 1.00 | 1.15 | 1.15 | 3.20 | 43.83 | 1.38 | 1.25 | 1.13 | 1.09 | 1.04 | 1.04 | 1.13 |
| `adaptive+trip(K=1)` | 1.00 | 1.26 | 1.26 | 3.20 | 66.98 | 1.63 | 1.27 | 1.25 | 1.16 | 1.06 | 1.05 | 1.02 |
| `adaptive+trip(K=2)` | 1.00 | 1.52 | 1.46 | 3.20 | 66.98 | 2.01 | 1.50 | 1.40 | 1.29 | 1.11 | 1.07 | 1.03 |
| `adaptive+trip(K=4)` | 1.00 | 1.98 | 1.93 | 3.20 | 66.98 | 2.41 | 1.87 | 1.72 | 1.53 | 1.18 | 1.11 | 1.05 |
| `adaptive+trip(K=8)` | 1.00 | 2.91 | 2.75 | 3.20 | 66.98 | 3.19 | 2.58 | 2.28 | 1.89 | 1.32 | 1.17 | 1.06 |

### The headline

| policy | worst case across all twelve workloads |
|---|---:|
| `guarded/adaptive(K=0.5)` | **1.38×** |
| `guarded/adaptive(K=1)` | **1.63×** |
| `guarded/adaptive(K=0.25)` | **1.69×** |
| `always-jit` | **12.80×** |
| `always-dfa` | **12.84×** |
| `amortised(K=0.25)` | **43.83×** |
| `amortised(K=0.5)` | **43.83×** |
| `adaptive(K=0.25)` | **43.83×** |

Every *fixed* policy is catastrophically wrong somewhere:
`always-jit` is 12.8× the optimum on `one-shot`, `always-dfa` is 12.8× on
`bulk`, and doing nothing is 133× on `log-lines`. The best plain call-counter
is 45× on `adversarial`. `guarded/adaptive(K=0.5)` is never worse
than **1.38×** anywhere.

## 5. What the sweep says about when any of this matters

`mix-N` holds the hot core fixed and adds *N* single-use patterns:

| N (cold patterns) | `never` | `always-jit` | `calls(2,4)` | `guarded/adaptive(K=0.5)` |
|---:|---:|---:|---:|---:|
| 0 | 92.66 | 1.00 | 1.16 | 1.25 |
| 50 | 72.36 | 3.71 | 1.14 | 1.13 |
| 200 | 43.99 | 7.21 | 1.08 | 1.09 |
| 800 | 18.44 | 9.89 | 1.05 | 1.04 |
| 2000 | 8.32 | 9.72 | 1.14 | 1.04 |
| 5000 | 4.05 | 7.90 | 1.52 | 1.13 |

With no cold patterns at all, compiling everything on sight *is* optimal and the
policies cost 13–25% for nothing. By 50 cold patterns `always-jit` is already
3.7× off; by 2000 it is worse than never compiling at all. Nothing about the hot
core changed. The entire value of a tiering policy is insurance against the
fraction of your patterns that are used once — and since you cannot know that
fraction in advance, the insurance is worth its 13%.

## 6. Two results that were not the plan

**The learned build cost barely helped.** `adaptive` replaces the static
build-cost estimate with a running average of builds actually observed. The
reasoning was sound — build cost is the model's worst-fitting term — but across
twelve workloads it is roughly a wash with `amortised`, winning clearly only at
the extreme end of the sweep (1.13× vs 1.27× at `mix-5000`). Reported because it
was the interesting idea and it mostly did not pay.

**No reactive policy can survive catastrophic backtracking.** On `adversarial`,
every policy that decides from observed cost lost by one to two orders of
magnitude, and no amount of tuning K moved it. That is structural: a reactive
policy needs evidence, and for these patterns the first bad call *is* the
disaster. The fix is not a better policy, it is not being reactive — a static
check for nested or ambiguous unbounded repetition
(`hp.re.parser.risky`), which flags all three pathological patterns and none of
the 36 benign ones in the test corpus, and sends them straight to the DFA. That
check plus a symmetric one for subjects too large to learn from is what takes
`adversarial` from 44× to 1.00× and `bulk` from 3.20× to 1.21×.

The best policy in this table is therefore not the cleverest one. It is a
mediocre cost model wrapped in two refusals to gather evidence.

## 7. How good is the cost model?

Mean relative error of each fitted term, from `results/calibration.json`:

| quantity | mean relative error |
|---|---:|
| backtrack step | 60% |
| code size | 8% |
| dfa build | 47% |
| dfa build constant only | 70% |
| dfa call | 3% |
| jit build | 17% |
| jit call | 10% |

Per-call costs fit well; **DFA build cost is the weak term and stays weak.**
Subset-construction cost tracks the number of DFA states *before* minimisation,
and that is not a function of anything visible beforehand:
`[A-Z]{2}\d{2}[A-Z0-9]{4,20}` has 59 NFA states, 9 DFA states, and passes
through 3312 on the way — a hundred times the cost of patterns that look just
like it. Fitting against pattern length gave a 1400%% error; a power law in NFA
size, class count and counted-repetition span gets it to the figure above, which
is better and still bad. The `adaptive` policy exists because of this term, and
section 6 reports how little that bought.

