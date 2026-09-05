# hotpath

A regular expression engine that compiles patterns to **x86-64 machine code at
runtime** — and an experiment about the question that creates: *which patterns
are worth compiling at all?*

```console
$ python -m hp compile 'a+b'
pattern   'a+b'

DFA: 3 states, 3 classes, 9 edges, start=0, start_mid=0
  s0: c0->0, c1->1, c2->0
  s1: c0->0, c1->1, c2->2
  s2 (accept): c0->2, c1->2, c2->2

generated 120 bytes of x86-64 (sysv ABI), 3 byte classes

  0000  xor ecx, ecx
  0002  movabs r11, 0x7f3c4a1e5000
  000c  jmp s0
s0:
  0011  cmp rcx, rsi
  0014  jae s0_end
  001a  movzx eax, byte [rdi+rcx]
  001e  movzx eax, byte [r11+rax]
  0023  inc rcx
  0026  cmp eax, 1
  0029  je s1
  002f  jmp s0
s0_end:
  0034  mov eax, 0
  0039  ret
...
```

Those bytes are written into a page mapped `PROT_EXEC` and jumped into. On
subjects large enough for scanning to dominate, they run **7–47× faster than
CPython's `re`**, and up to 34× on a pattern that `re` handles well.

## The result

Compiling is not free: a pattern costs 0.3–2 ms of subset construction plus
0.2–1.3 ms of code generation before it runs a single byte faster. So the
engine has three tiers — a backtracking interpreter, an interpreted DFA, and
generated machine code — and a *policy* decides when to move between them.

Twelve workloads were replayed against every policy and against a
**hindsight oracle** that knows the whole trace in advance and picks the exact
optimal moment to build each tier. Cost as a multiple of that optimum:

| policy | worst case over 12 workloads |
|---|---:|
| `guarded/adaptive(K=0.5)` | **1.38×** |
| `always-jit` (compile everything) | 12.80× |
| `always-dfa` | 12.84× |
| `calls(2,4)` (the counter every JIT ships) | 45.06× |
| `never` (pure backtracking) | 132.73× |

Full tables, per-workload, in [`results/summary.md`](results/summary.md).

The interesting part is *why* the winner wins. It is not a better cost model —
its cost model is mediocre, and the write-up quantifies exactly how mediocre.
It wins because of two places where it **refuses to gather evidence**:

- A pattern with nested or ambiguous unbounded repetition (`(a+)+`, `(x|xx)+`)
  never touches the backtracking tier. Every reactive policy lost this workload
  by one to two orders of magnitude and no amount of tuning fixed it, because a
  reactive policy needs evidence and here the first bad call *is* the disaster.
- A subject long enough that even the DFA's per-byte cost would exceed the
  build cost skips the interpreted tiers entirely. There is nothing to learn
  from that call except how long it takes.

Both checks are static, both run once per pattern, and together they take the
adversarial workload from 67× the optimum to exactly 1.00× and the bulk workload
from 3.20× to 1.21×.

## When is a tiering policy worth having at all?

The `mix-N` sweep holds a hot core of five patterns fixed and adds *N* patterns
used exactly once:

| cold patterns | `never` | `always-jit` | `guarded/adaptive(K=0.5)` |
|---:|---:|---:|---:|
| 0 | 92.66× | **1.00×** | 1.25× |
| 50 | 72.36× | 3.71× | **1.13×** |
| 200 | 43.99× | 7.21× | **1.09×** |
| 800 | 18.44× | 9.89× | **1.04×** |
| 2000 | 8.32× | 9.72× | **1.04×** |
| 5000 | 4.05× | 7.90× | **1.13×** |

With no single-use patterns, compiling everything on sight is optimal and the
policy costs 25% for nothing. By 2000 it is *worse than never compiling*.
Nothing about the hot traffic changed. The whole value of a tiering policy is
insurance against the fraction of your patterns that are used once — and since
you cannot know that fraction in advance, the 25% is what the insurance costs.

## How it works

```
pattern ──► parser ──► NFA ──► DFA ──► x86-64
             │          │       │         │
        byte-set     Thompson  subset  one basic block per state,
        classes      (linear)  + Moore byte-class table in the same mapping
```

**Search is `.*P.*` matched entirely.** Restarting a DFA at every offset is
quadratic; wrapping the pattern pushes the work into the automaton, where it
costs states rather than time, and the whole scan is one left-to-right pass.
Anchors survive it — `^` is an epsilon edge the closure only admits at offset
zero, and `$` is only consulted for the accept bit, never for a transition.

**Alphabet compression.** The 256 bytes are partitioned into classes no guard
can tell apart — usually three or four. That is the difference between a 2 KB
jump table per state and three `cmp` instructions, and it is why the generated
code is short enough to read.

**Two accept bits and two start states.** `accept_here` / `accept_at_end` and
`start` / `start_mid` push every position-dependent question into the DFA
construction, so the running matcher never looks at a position again. The
generated code inherits it: no bounds arithmetic beyond the loop counter.

**Data and code share one mapping**, W^X enforced — written while writable,
then flipped to read+execute, never both. Code is generated twice, once against
a placeholder to learn the size and once against the real address; `movabs` is
fixed-width so the sizes must agree, and that is asserted rather than trusted.

**The disassembler is independent.** `hp/jit/asm.py` builds bytes from
mnemonics; `hp/jit/disasm.py` builds mnemonics from bytes, written from a
separate reading of the encoding rules. The test suite assembles every program,
disassembles it, and requires the two texts to match — so a swapped REX bit is a
failing unit test instead of a segfault inside generated code with no traceback.
It also means `python -m hp compile` works on machines that cannot execute a
single generated byte, Windows included.

## Running it

```console
$ pip install -e .            # no dependencies; pytest for the tests
$ python -m hp compile '[a-z]+@[a-z]+'      # generated code, annotated
$ python -m hp match  'a+b' 'xxaab'         # all three tiers, side by side
$ python -m hp analyse '^(a+)+$'            # what the policy knows up front
$ python -m hp bench  '\d+\.\d+' 65536      # per-tier throughput vs re
$ make test                                 # 464 tests
$ make experiment                           # calibrate, measure, replay, report
```

`make experiment` takes a few minutes and rewrites `results/`. It is fully
deterministic given the machine: every workload is seeded, so a trace is a value
and a policy comparison is not a comparison of luck.

## How the experiment is put together

The measurement pass and the policy pass are separate, and the separation is
load-bearing.

**Measure once.** Each trace is replayed three times — pinned to each tier — and
the wall-clock nanoseconds of every individual call are recorded, along with the
real build costs including `mmap` and `mprotect`. Nothing here is modelled.

**Then replay policies over those recorded numbers.** Policies were originally
compared by re-running them for real, and some scored *better than the oracle* —
machine noise between two timing passes, showing up as a policy beating the
optimum. Driving the policies and the oracle from one set of measurements
removes that entirely. The policies are the real objects from `hp/policy.py`,
unmodified; they see a stand-in exposing exactly the attributes the live engine
exposes. `python -m hp.eval.run validate <workload>` runs them for real and
compares, and a test asserts the replay and the live engine take *identical
decisions* — same final tiers, same promotion count.

**The oracle is exact, not approximate.** Tiers are monotone, so a schedule is
two indices, and the minimum over all of them falls out of prefix sums in linear
time. A test checks it against exhaustive search on 40 random instances.

**The cost model is fitted to the machine**, not typed in, by least squares
minimising *relative* error — these quantities span four orders of magnitude and
ordinary least squares spends all its accuracy on the largest points.
`results/calibration.json` reports the residual of every term.

## Limitations, honestly

- **DFA build cost is not predictable from the pattern, and that is not fixable
  by choosing better features.** It tracks the number of DFA states *before*
  minimisation: `[A-Z]{2}\d{2}[A-Z0-9]{4,20}` has 59 NFA states, 9 DFA states,
  and passes through 3312 on the way — a hundred times the cost of patterns that
  look identical. A power law in NFA size, class count and counted-repetition
  span gets the mean relative error from 1400% down to roughly 50%, which is
  better and still bad. The `adaptive` policy — which learns build cost from
  builds it has watched instead of predicting it — was the intended answer, and
  across twelve workloads it is close to a wash. Reported because it was the
  interesting idea and it mostly did not pay.
- **No captures, no backreferences, no lookaround.** A DFA has nowhere to record
  where a submatch began, and backreferences make the language non-regular.
  Leaving them out is the premise, not a shortcut: it is what buys the
  linear-time guarantee. Backreferences are rejected with an error that says so.
- **`$` means end of subject**, always. Python's `$` also matches before a
  trailing newline; a byte-scanning engine should not be guessing at lines. The
  divergence has its own test.
- **The JIT is Unix and x86-64 only.** Everything else — parsing, the automata,
  code *generation*, and the disassembler — is pure Python and runs anywhere.
- **`ctypes` sets the floor.** A call into generated code costs ~0.6 µs, so for
  subjects under a few hundred bytes the FFI dominates and CPython's `re` wins.
  One benchmark row shows this losing 5×, and it is left in.
- **Tier 0 is censored** in the measurements at 50,000 backtracking steps. Its
  true cost is higher, so the savings available are understated and every
  policy's recovered fraction is conservative.
- **Timing on a shared machine is timing on a shared machine.** Per-call figures
  are minimum-of-N where that is affordable; the large traces rely on averaging
  over thousands of calls.

## Layout

| path | |
|---|---|
| `hp/re/` | parser, Thompson NFA, subset construction + Moore minimisation, backtracking interpreter |
| `hp/jit/` | x86-64 assembler, independent disassembler, code generator, W^X executable memory |
| `hp/engine.py` | the three tiers behind one `matches()` call |
| `hp/policy.py` | every tiering policy, the cost model, the static guards |
| `hp/eval/` | calibration, measurement, the oracle, the replay, the runner |
| `tests/` | 464 tests |
| `results/` | measurements, and `summary.md` built from them |
| `docs/ARCHITECTURE.md` | why each piece is shaped the way it is |

## Where this would go next

- **A real jump table** for states with many classes, instead of a compare
  chain. The current form is optimal for three or four classes and clearly not
  for twenty.
- **Sparse-set NFA simulation** as a fourth tier, between the backtracker and
  the DFA — linear time with no construction cost, which is what RE2 falls back
  to and would fill the gap the `one-shot` workload exposes.
- **SIMD prefiltering**: find candidate positions for a required literal with
  `pcmpeqb` before entering the automaton at all. For patterns with a literal
  anchor this is usually worth more than everything above.
- **A cost model with error bars.** The policy currently treats its build-cost
  estimate as a number when it is closer to an order of magnitude, and a policy
  that knew its own uncertainty could be appropriately cowardly about the one
  term it cannot predict.

## Licence

MIT.
