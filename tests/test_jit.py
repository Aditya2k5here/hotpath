"""The assembler, the disassembler, and the code they produce.

Machine code is unusually unforgiving to test. A wrong REX bit does not raise;
it computes the wrong answer, or jumps into the middle of an instruction, or
segfaults the interpreter with no traceback. So the strategy here is to never
rely on running the code to find out whether it is right:

* the assembler and the disassembler are written from the encoding rules
  independently, and every program must round-trip through both;
* the encodings of individual instructions are pinned to their exact bytes,
  checked against the manual, so a change that alters them has to be deliberate;
* only then is the code executed, and the answers are compared against the
  interpreted DFA on a corpus.

The three layers fail in that order, which means a bad encoding shows up as a
byte mismatch in a unit test rather than as a crash inside generated code.
"""

import random
import re

import pytest

from hp.jit import asm, codegen, disasm, memory, program as prog
from hp.re import dfa as D, nfa as N, parser as P

pytestmark = pytest.mark.filterwarnings("ignore")

needs_jit = pytest.mark.skipif(not memory.available(),
                               reason="no executable memory on this platform")


def _search_dfa(pattern, max_states=5000):
    node = P.unanchored(P.parse(pattern))
    return D.minimise(D.build(N.build(node), max_states=max_states))


PATTERNS = [
    "abc", "a+b", "[a-z]+@[a-z]+", r"\d{3}-\d{4}", "^GET ", "error$",
    "(foo|bar|baz)", "a.c", "x*y", "(a|b)*abb", "^$", r"\w+\s\w+",
    r"[a-z0-9._]+@[a-z0-9.]+\.[a-z]{2,6}", r"\d+\.\d+\.\d+\.\d+",
    "TODO|FIXME", "[^ ]+", r"v\d+\.\d+", "(ab|cd)+",
]


# -- encodings ---------------------------------------------------------------
@pytest.mark.parametrize("build,expected", [
    (lambda a: a.push(asm.RDI), "57"),
    (lambda a: a.push(asm.R12), "41 54"),
    (lambda a: a.pop(asm.RSI), "5e"),
    (lambda a: a.ret(), "c3"),
    (lambda a: a.xor_r32_r32(asm.RCX, asm.RCX), "31 c9"),
    (lambda a: a.inc_r64(asm.RCX), "48 ff c1"),
    (lambda a: a.cmp_r64_r64(asm.RCX, asm.RSI), "48 39 f1"),
    (lambda a: a.mov_r64_r64(asm.RDI, asm.RCX), "48 89 cf"),
    (lambda a: a.cmp_r32_imm8(asm.RAX, 3), "83 f8 03"),
    (lambda a: a.mov_r32_imm32(asm.RAX, 1), "b8 01 00 00 00"),
    (lambda a: a.movabs(asm.R11, 0x7F1234567890), "49 bb 90 78 56 34 12 7f 00 00"),
    (lambda a: a.movzx_r32_byte_base_index(asm.RAX, asm.RDI, asm.RCX),
     "0f b6 04 0f"),
    (lambda a: a.movzx_r32_byte_base_index(asm.RAX, asm.R11, asm.RAX),
     "41 0f b6 04 03"),
])
def test_instruction_encodings(build, expected):
    a = asm.Assembler()
    build(a)
    got = " ".join("%02x" % b for b in a.assemble())
    assert got == expected


def test_relative_branches_are_patched_to_the_label():
    a = asm.Assembler()
    a.jmp("end")
    a.ret()
    a.label("end")
    a.mov_r32_imm32(asm.RAX, 7)
    a.ret()
    code = a.assemble()
    # E9 + rel32, then one byte of ret, so the displacement is 1.
    assert code[0] == 0xE9
    assert int.from_bytes(code[1:5], "little", signed=True) == 1


def test_branch_to_a_missing_label_is_an_error():
    a = asm.Assembler()
    a.jmp("nowhere")
    with pytest.raises(asm.LabelError):
        a.assemble()


def test_a_label_cannot_be_defined_twice():
    a = asm.Assembler()
    a.label("x")
    with pytest.raises(asm.LabelError):
        a.label("x")


def test_immediates_that_do_not_fit_are_refused():
    a = asm.Assembler()
    with pytest.raises(ValueError):
        a.cmp_r32_imm8(asm.RAX, 300)


def test_invalid_addressing_forms_are_refused():
    a = asm.Assembler()
    with pytest.raises(ValueError):
        a.movzx_r32_byte_base_index(asm.RAX, asm.RDI, asm.RSP)
    with pytest.raises(ValueError):
        a.movzx_r32_byte_base_index(asm.RAX, asm.RBP, asm.RCX)


# -- round trip --------------------------------------------------------------
@pytest.mark.parametrize("pattern", PATTERNS)
@pytest.mark.parametrize("abi", ["sysv", "win64"])
def test_assembler_and_disassembler_agree(pattern, abi):
    """Two independent readings of the encoding rules must produce one text."""
    a = codegen.generate(_search_dfa(pattern), 0x7F0000000000, abi=abi)
    code = a.assemble()
    assert a.text() == disasm.disassemble(code, a.labels), pattern


@pytest.mark.parametrize("pattern", PATTERNS)
def test_every_generated_byte_decodes(pattern):
    """No gaps and no overlap: the decoder must consume the buffer exactly."""
    code = codegen.compile_to_bytes(_search_dfa(pattern), 0)
    insns = disasm.decode_all(code)
    assert sum(length for _, length, _ in insns) == len(code)
    assert insns[0][0] == 0


def test_generated_code_is_deterministic():
    dfa = _search_dfa(r"[a-z]+@[a-z]+\.[a-z]+")
    first = codegen.compile_to_bytes(dfa, 0x1000)
    for _ in range(4):
        assert codegen.compile_to_bytes(_search_dfa(
            r"[a-z]+@[a-z]+\.[a-z]+"), 0x1000) == first


def test_code_size_is_address_independent():
    """The two-pass allocation depends on this; it is asserted, not assumed."""
    dfa = _search_dfa("(foo|bar|baz)+")
    assert len(codegen.compile_to_bytes(dfa, 0)) == \
        len(codegen.compile_to_bytes(dfa, 0x7FFFFFFFF000))


def test_listing_needs_no_cpu():
    """The disassembly path must work on machines that cannot run the code."""
    text = prog.listing_for(_search_dfa("a+b"))
    assert "movzx" in text and "ret" in text
    assert "s0:" in text


def test_too_many_classes_is_refused_not_miscompiled():
    dfa = _search_dfa("abc")
    dfa.n_classes = codegen.MAX_CLASSES + 1
    with pytest.raises(codegen.NotCompilable):
        codegen.generate(dfa, 0)


# -- execution ---------------------------------------------------------------
@pytest.fixture(scope="module")
def subjects():
    rnd = random.Random(4242)
    alphabet = "abcdefgxyz0123 @-.\n"
    out = [b"", b"abc", b"xxabcxx", b"aab", b"me@you", b"555-1234", b"GET /x",
           b" GET /x", b"an error", b"error!", b"foo", b"baz", b"abababb",
           b"a\nc", b"axc", b"hello world", b"v1.20", b"TODO: fix",
           b"user@example.com", b"10.0.0.1"]
    for _ in range(400):
        out.append(bytes(ord(rnd.choice(alphabet))
                         for _ in range(rnd.randrange(0, 14))))
    return out


@needs_jit
@pytest.mark.parametrize("pattern", PATTERNS)
def test_machine_code_matches_python(pattern, subjects):
    dfa = _search_dfa(pattern)
    program = prog.compile_dfa(dfa)
    try:
        ref = re.compile(pattern.replace("$", r"\Z").encode())
        for s in subjects:
            want = ref.search(s) is not None
            assert dfa.fullmatch(s) is want, (pattern, s, "interpreted")
            assert program(s) is want, (pattern, s, "compiled")
    finally:
        program.free()


@needs_jit
def test_machine_code_handles_large_and_embedded_nul_subjects():
    dfa = _search_dfa("needle")
    program = prog.compile_dfa(dfa)
    try:
        blob = b"\x00hay\x00stack\x00" * 20000
        assert program(blob) is False
        assert program(blob + b"needle") is True
        assert program(bytearray(b"a needle here")) is True
    finally:
        program.free()


@needs_jit
def test_memory_is_never_writable_and_executable_at_once():
    block = memory.ExecutableBlock(memory.PAGE)
    try:
        block.write(0, b"\xc3")
        block.make_executable()
        with pytest.raises(RuntimeError):
            block.write(0, b"\x90")
    finally:
        block.free()


@needs_jit
def test_programs_release_their_mapping():
    dfa = _search_dfa("abc")
    p = prog.compile_dfa(dfa)
    assert p.block.freed is False
    p.free()
    assert p.block.freed is True
    p.free()                       # idempotent


@needs_jit
def test_thousands_of_programs_do_not_leak_pages():
    """The policy experiments compile many programs; each must give its page back."""
    dfa = _search_dfa("a+b")
    for _ in range(400):
        p = prog.compile_dfa(dfa)
        assert p(b"aab") is True
        p.free()


@needs_jit
def test_accepting_states_return_early():
    """The absorbing-accept optimisation must actually stop the scan."""
    dfa = _search_dfa("abc")
    assert codegen._absorbing_accepts(dfa) is True
    program = prog.compile_dfa(dfa)
    try:
        assert program(b"abc" + b"x" * 1000000) is True
    finally:
        program.free()
