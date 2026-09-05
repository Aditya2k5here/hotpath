# Architecture

Why each piece is shaped the way it is. The README says what the project does
and what it found; this says what the alternatives were and why they lost.

## The pipeline

```
source ──parser──► AST ──Thompson──► NFA ──subset+Moore──► DFA ──codegen──► x86-64
           │                          │                     │                 │
      byte-set classes           linear size           byte classes      one block/state
      static risk check          static size est.      2 accept bits     W^X mapping
                                                       2 start states
```

Each stage produces something the next stage needs *and* something the tiering
policy can use before paying for the next stage. That is the non-obvious
constraint the layering is built around: a policy has to decide whether to run
the subset construction, so everything cheap enough to compute first. AST
shape, static NFA size, backtracking risk. Is available without it.

## Decisions

### The pattern language is regular, deliberately

No backreferences, no lookaround, no captures.

Backreferences make the language strictly more powerful than a finite
automaton, so an engine that supports them cannot compile to a DFA and cannot
promise linear time. Every catastrophic-backtracking outage is that trade being
made silently. Refusing them is what makes the rest possible, so `\1` is
rejected with an error that explains the reason rather than the symptom.

Captures are refused for a narrower reason: a DFA has nowhere to record where a
submatch began. Supporting them would mean either a tagged automaton (a much
larger project) or a lie the code generator could not keep. Groups exist for
precedence only.

### Search is `.*P.*` matched entirely, not a restart loop

The obvious way to search is to run the DFA from each offset until one matches.
That is quadratic on a non-matching subject, and the quadratic term is invisible
in small tests.

Wrapping the pattern instead pushes the work into the automaton, where it costs
*states* rather than *time*, and the whole scan becomes one left-to-right pass.
This is what RE2 does.

The part worth checking is that anchors survive it. They do, and for a reason
specific to how the closure is written: `^` is an epsilon edge the closure only
admits when `at_start` holds, which is true only for the start state's closure,
where the leading `.*` has consumed nothing. `$` is admitted only in the
*end-of-subject* closure, which is consulted for the accept bit and never for a
transition, so the trailing `.*` cannot smuggle bytes in after it.

### Two accept bits and two start states

`accept_here` / `accept_at_end`, and `start` / `start_mid`.

The alternative is a transition function that takes the position as an argument,
so `$` and `^` can be evaluated as the scan runs. That would put a branch on
position into the inner loop of the generated code, which is the one place there
is nothing to spare.

Deciding both at construction time costs two extra bits per state and one extra
start state, and buys an inner loop that is: load byte, load class, increment,
compare, jump. Nothing else.

### Alphabet compression before anything else

A DFA over bytes nominally has 256 transitions per state. Two bytes are
equivalent exactly when they belong to the same set of guards, so the partition
falls out of one signature per byte, and for real patterns it collapses to three
or four classes.

This is the difference between emitting a 2 KB jump table per state and emitting
three `cmp` instructions. It is also why `python -m hp compile` produces
something a person can read, which turned out to matter more than expected.
Most of the code-generation bugs were found by reading the listing.

Classes are numbered by their smallest member so the numbering is stable across
runs. Set iteration order varies with the hash seed, and generated code that
differs between runs of the same program is impossible to review.

### Minimisation is not cosmetic

Moore partition refinement, run on every DFA.

Fewer states means less generated code, which means the JIT pays for itself
sooner, and since the tiering policy is the thing being measured, making
compilation cheaper *moves the results* rather than tidying them.
`(foo|bar|baz)+` goes from 15 states to 6; `[A-Z]{2}\d{2}[A-Z0-9]{4,20}` from
3312 to 9.

That last number is also the reason build cost cannot be predicted from the
pattern: the cost tracks the *pre*-minimisation count, which nothing observable
beforehand determines.

### The assembler encodes eleven instructions and no more

Every encoding has to be correct to the bit, and there is no way to be confident
of that for an instruction nobody exercises. Adding one is cheap; carrying an
untested one is not.

The disassembler is written from a separate reading of the encoding rules, and
the test suite requires every generated program to round-trip. This catches the
classic x86 mistakes. REX bit in the wrong field, ModRM `reg` and `rm`
swapped. As a text mismatch in a unit test, rather than as a wrong answer or a
segfault inside generated code where there is no debugger and no traceback.

It also means the compile path is pure Python. You can read exactly what hotpath
would emit on a machine that cannot execute a single byte of it.

### Data before code in one mapping

The 256-byte class table sits at the base of the mapping, the code after it. One
allocation, one address to know, and the code's own offsets do not move when the
table changes.

W^X is enforced: the region is mapped writable, filled, then flipped to
read+execute, never both at once. One extra syscall, and it is the discipline
every browser JIT and every hardened kernel requires.

Code is generated twice. Once against a placeholder to learn the size, once
against the real address. `movabs` is fixed-width so the two passes must produce
identical lengths, and `Program.__init__` asserts it rather than trusting it. If
an address-dependent instruction ever creeps into the generator, that assertion
fires instead of the mapping being sized wrong.

### `ctypes` hands over a pointer, not a copy

The obvious `create_string_buffer` costs more than the match above a few
kilobytes and turns a measurement of generated code into a measurement of
`memcpy`. Passing the `bytes` object to a `c_char_p` argument hands over a
pointer to its own storage; the length is passed separately, so embedded NULs
are not a problem.

This one change took the 64 KB figure from 3.9 ns/byte to 0.9.

### The engine degrades instead of failing

A pattern whose subset construction exceeds its bound, or whose alphabet is too
wide for the code generator, lowers its own *ceiling* and keeps answering on a
slower tier. From the caller's point of view a pattern that cannot be compiled
is not an error; it is a pattern that stays interpreted.

The step budget works the same way. A tier-0 call that exhausts it forces a
promotion no policy gets a say in, because an engine that hangs is not a
baseline, it is an outage. Stack exhaustion is caught and treated identically:
a backtracker recurses once per byte it is holding a choice point for, so it
dies of depth on long subjects, which is the same failure that crashed real
regex engines for years.

## How the experiment is built, and why that shape

### Measure once, replay many times

Policies were originally compared by running each one for real and timing it.
Some scored *better than the oracle*. Machine noise between two separate timing
passes showing up as a policy beating the optimum, which is nonsense.

So the trace is replayed three times, pinned to each tier, and the wall-clock
cost of every individual call is recorded. Policies and the oracle are then both
driven from that one set of numbers. A policy now cannot score better than the
optimum, and the difference between two policies is attributable to their
decisions rather than to what else the machine was doing.

The shortcut is audited two ways: `hp.eval.run validate` re-runs policies for
real and reports the error, and a test asserts the replay and the live engine
take *identical decisions*. Same final tiers, same promotion count. The
policies themselves are the real objects, unmodified; they see a stand-in
exposing exactly the attributes the live `Pattern` exposes, so what is simulated
is the timing, never the policy.

### The oracle is exact

Tiers are monotone. Once promoted a pattern stays promoted, so a schedule is
two indices `(i, j)`, and

```
total(i, j) = S0[i] + (S1[j] - S1[i]) + (S2[n] - S2[j])
              + build_dfa if i < n
              + build_jit if j < n
```

with `Sk` the prefix sums of measured per-call costs at tier `k`. Minimising
over all `0 <= i <= j <= n` needs one pass and a running minimum. No search, no
approximation. A test checks it against exhaustive `O(n²)` search on 40 random
instances, because "the optimum" is a claim that has to be earned.

Two weaker oracles are computed alongside, and the gaps are results in
themselves: the best *fixed* tier chosen in hindsight (the difference is the
value of being able to tier at all) and the full oracle with build costs zeroed
(the difference is what compilation costs).

### Calibration minimises relative error

The quantities span four orders of magnitude, a backtracking call costs
3 µs or 30 ms. Ordinary least squares spends all its accuracy on the largest
points and produces a model that is 1000% wrong everywhere the policy actually
operates. Weighting each residual by `1/y²` and iterating targets the error that
matters: being right to within a factor, at every scale.

Negative coefficients are dropped and the fit repeated. A negative coefficient
in a cost model is not a fit, it is an artefact, and it produces negative
predicted costs on inputs outside the calibration set.

Build cost is fitted in log space instead, because its true shape is
multiplicative, a power law in NFA size, class count and counted-repetition
span. That change alone took its mean relative error from 352% to about 50%.

## What the results changed about the design

Two pieces of this codebase exist only because the experiment demanded them, and
both are worth more than the code that was planned.

**`hp.re.parser.risky`.** On a workload of catastrophic-backtracking patterns,
every reactive policy lost by one to two orders of magnitude, and sweeping `K`
did nothing. That is structural, not a tuning failure: a reactive policy decides
from evidence, and for these patterns the first bad call *is* the disaster. The
answer is a static check for nested or ambiguous unbounded repetition, which
sends those patterns straight to the DFA and never lets them near the
backtracker. It flags all three pathological patterns in the corpus and none of
the 36 benign ones.

**`Guarded._single_call_tier`.** The same principle with a different trigger. A
pattern seen six times on quarter-megabyte blobs never accumulates enough
*calls* to trip a counter, and an amortising policy has spent more than the
compiler would have cost by the time it has watched two of them. Since tier 0 is
never faster per byte than the DFA, one subject long enough that even the DFA's
scan would exceed the build cost settles the question before the call runs.

Both are refusals to gather evidence, and together they are why the best policy
in the table is a mediocre cost model with two static guards bolted on rather
than the sophisticated thing the project set out to build.
