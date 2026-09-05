"""DFA -> x86-64 machine code.

The shape of the generated function is one basic block per DFA state:

.. code-block:: none

    s3:
      cmp   rcx, rsi              ; index vs length
      jae   s3_end                ; input exhausted
      movzx eax, byte [rdi+rcx]   ; the subject byte
      movzx eax, byte [r11+rax]   ; -> its equivalence class
      inc   rcx
      cmp   eax, 0
      je    s1
      jmp   fail                  ; the majority target, as fallthrough
    s3_end:
      mov   eax, 0
      ret

Register assignment is fixed and deliberately tiny: ``rdi`` the subject,
``rsi`` its length, ``rcx`` the index, ``rax`` scratch, ``r11`` the class table.
Nothing is spilled, nothing is saved, there is no frame -- a DFA needs no stack,
which is the same property that makes it immune to the blowup in tier 0.

Three things the generator does that a transcription of the state table would
not:

**The majority target becomes the fallthrough.** Each state's dispatch emits a
compare only for the minority classes and a single ``jmp`` for the rest. For
``[a-z]+`` that is one comparison per byte instead of one per class.

**States that cannot change their mind return immediately.** When the set of
accepting states is closed under transitions -- exactly the case for the
``.*P.*`` automaton the search entry point uses -- reaching one settles the
answer, so it compiles to ``mov eax, 1; ret``. The mirror image matters just as
much: a non-accepting state whose every transition is a self-loop is hopeless,
and an anchored pattern reaches one after its first byte. Compiling that to
``mov eax, 0; ret`` is the difference between reading one byte of a megabyte and
reading all of it.

**Classes are numbered so the immediate fits in a byte.** Patterns needing more
than 128 equivalence classes are refused here rather than mis-encoded, and the
engine falls back to the interpreted DFA. That is a real limitation and it is
reported as one.
"""

from . import asm
from .asm import RAX, RCX, RSI, RDI, R11

MAX_CLASSES = 128

DATA_OFFSET = 0            # the 256-byte class table sits at the base
CODE_OFFSET = 256          # code follows it


class NotCompilable(Exception):
    """This DFA cannot be compiled by this back end.

    The engine treats it as "stay on the interpreted tier", not as an error.
    """


def _absorbing_accepts(dfa):
    """True when no path leaves the accepting set.

    If so, entering an accepting state settles the answer, and the generated
    code can return the moment it gets there.
    """
    accepting = [s for s in range(len(dfa)) if dfa.accept_here[s]]
    if not accepting:
        return False
    for s in accepting:
        for t in dfa.trans[s]:
            if t >= 0 and not dfa.accept_here[t]:
                return False
    return True


def _absorbing_rejects(dfa):
    """States that are not accepting and can never leave themselves.

    Minimisation collapses every hopeless state into one, but it stays a real
    state with real self-loops rather than becoming the ``-1`` dead edge, so the
    generated code would happily scan the rest of the subject inside it. That is
    exactly what happens to an anchored pattern: ``^GET`` wrapped as ``.*^GET.*``
    can only match at offset zero, so from byte one onwards the answer is
    already no -- and without this the engine reads the other megabyte to
    confirm it. Symmetric to :func:`_absorbing_accepts`, and it is what stopped
    anchored patterns losing to CPython by 10x.
    """
    out = set()
    for s in range(len(dfa)):
        if dfa.accept_here[s] or dfa.accept_at_end[s]:
            continue
        if all(t == s for t in dfa.trans[s]):
            out.add(s)
    return out


def _majority_target(row):
    """The transition target shared by the most classes (``-1`` counts)."""
    counts = {}
    for t in row:
        counts[t] = counts.get(t, 0) + 1
    return max(sorted(counts), key=lambda t: counts[t])


def generate(dfa, table_addr, abi="sysv"):
    """Emit the matcher for ``dfa``, with the class table at ``table_addr``.

    Returns the :class:`hp.jit.asm.Assembler`, un-assembled, so callers can read
    the listing as well as the bytes.
    """
    if dfa.n_classes > MAX_CLASSES:
        raise NotCompilable("%d byte classes exceeds the %d this back end "
                            "encodes" % (dfa.n_classes, MAX_CLASSES))

    a = asm.Assembler()
    early_return = _absorbing_accepts(dfa)
    hopeless = _absorbing_rejects(dfa)

    # -- prologue ----------------------------------------------------------
    if abi == "win64":
        # rdi and rsi are callee-saved on Windows, and the arguments arrive
        # elsewhere; everything after this point is ABI-independent.
        a.push(RDI)
        a.push(RSI)
        a.mov_r64_r64(RDI, asm.RCX)
        a.mov_r64_r64(RSI, asm.RDX)
    elif abi != "sysv":
        raise ValueError("unknown abi %r" % abi)

    a.xor_r32_r32(RCX, RCX)                     # index = 0
    a.movabs(R11, table_addr)
    a.jmp("s%d" % dfa.start)

    # -- one block per state -----------------------------------------------
    for s in range(len(dfa)):
        a.label("s%d" % s)

        if early_return and dfa.accept_here[s]:
            _ret(a, 1, abi)
            continue

        if s in hopeless:
            _ret(a, 0, abi)
            continue

        row = dfa.trans[s]
        if all(t < 0 for t in row):
            # Dead end: whether it is a match depends only on this state.
            _ret(a, 1 if dfa.accept_at_end[s] else 0, abi)
            continue

        a.cmp_r64_r64(RCX, RSI)
        a.jcc("ae", "s%d_end" % s)
        a.movzx_r32_byte_base_index(RAX, RDI, RCX)      # subject byte
        a.movzx_r32_byte_base_index(RAX, R11, RAX)      # -> class
        a.inc_r64(RCX)

        default = _majority_target(row)
        for c, t in enumerate(row):
            if t == default:
                continue
            a.cmp_r32_imm8(RAX, c)
            a.jcc("e", "fail" if t < 0 else "s%d" % t)
        a.jmp("fail" if default < 0 else "s%d" % default)

        a.label("s%d_end" % s)
        _ret(a, 1 if dfa.accept_at_end[s] else 0, abi)

    a.label("fail")
    _ret(a, 0, abi)
    return a


def _ret(a, value, abi):
    a.mov_r32_imm32(RAX, value)
    if abi == "win64":
        a.pop(RSI)
        a.pop(RDI)
    a.ret()


def class_table(dfa):
    return bytes(dfa.byte_class)


def compile_to_bytes(dfa, table_addr, abi="sysv"):
    return generate(dfa, table_addr, abi=abi).assemble()


def code_size(dfa, abi="sysv"):
    """How many bytes the code will occupy, without needing a real address.

    ``movabs`` is fixed-width, so generating against a placeholder gives exactly
    the size generating against the real address will.
    """
    return len(compile_to_bytes(dfa, 0, abi=abi))
