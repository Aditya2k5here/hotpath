"""A very small x86-64 assembler.

Small on purpose. This encodes the eleven instructions the DFA code generator
actually emits and nothing else, because every instruction here has to be
correct to the bit and there is no way to be sure of that for instructions
nobody exercises. Adding an encoding is cheap; carrying an untested one is not.

Two things make it usable rather than merely present:

**Labels are resolved by the assembler, not the caller.** A DFA is a graph, so
its code is full of forward branches to states that have not been emitted yet.
Each branch records a fixup site; :meth:`Assembler.assemble` patches them at the
end. Callers name states, never offsets.

**Every instruction records what it was asked to emit.** ``self.listing`` keeps
the mnemonic alongside the byte range it produced. The disassembler in
:mod:`hp.jit.disasm` decodes those same bytes independently, and the test suite
asserts the two agree, so a wrong encoding shows up as a disagreement rather
than as a segfault three layers away.

Register numbering is the architectural one (RAX=0 .. R15=15), so ``reg & 7``
gives the ModRM field and ``reg >> 3`` gives the REX bit, which is the whole
reason the numbering is what it is.
"""

import struct

RAX, RCX, RDX, RBX, RSP, RBP, RSI, RDI = range(8)
R8, R9, R10, R11, R12, R13, R14, R15 = range(8, 16)

REG_NAMES64 = ["rax", "rcx", "rdx", "rbx", "rsp", "rbp", "rsi", "rdi",
               "r8", "r9", "r10", "r11", "r12", "r13", "r14", "r15"]
REG_NAMES32 = ["eax", "ecx", "edx", "ebx", "esp", "ebp", "esi", "edi",
               "r8d", "r9d", "r10d", "r11d", "r12d", "r13d", "r14d", "r15d"]

# Condition codes, by the low nibble of the 0F 8x / 7x opcodes.
CC = {"o": 0x0, "no": 0x1, "b": 0x2, "ae": 0x3, "e": 0x4, "ne": 0x5,
      "be": 0x6, "a": 0x7, "s": 0x8, "ns": 0x9, "p": 0xA, "np": 0xB,
      "l": 0xC, "ge": 0xD, "le": 0xE, "g": 0xF}


class LabelError(Exception):
    pass


def _rex(w=0, r=0, x=0, b=0):
    return 0x40 | (w << 3) | (r << 2) | (x << 1) | b


def _modrm(mod, reg, rm):
    return (mod << 6) | ((reg & 7) << 3) | (rm & 7)


def _sib(scale, index, base):
    return (scale << 6) | ((index & 7) << 3) | (base & 7)


class Assembler(object):
    def __init__(self):
        self.buf = bytearray()
        self.labels = {}          # name -> offset
        self.fixups = []          # (site_offset, next_insn_offset, name)
        self.listing = []         # (offset, length, text)

    # -- bookkeeping ---------------------------------------------------------
    def label(self, name):
        if name in self.labels:
            raise LabelError("label %r defined twice" % name)
        self.labels[name] = len(self.buf)
        return self

    def _emit(self, data, text):
        self.listing.append((len(self.buf), len(data), text))
        self.buf += bytes(data)
        return self

    def here(self):
        return len(self.buf)

    # -- instructions --------------------------------------------------------
    def push(self, reg):
        out = []
        if reg >= 8:
            out.append(_rex(b=1))
        out.append(0x50 + (reg & 7))
        return self._emit(out, "push %s" % REG_NAMES64[reg])

    def pop(self, reg):
        out = []
        if reg >= 8:
            out.append(_rex(b=1))
        out.append(0x58 + (reg & 7))
        return self._emit(out, "pop %s" % REG_NAMES64[reg])

    def mov_r64_r64(self, dst, src):
        out = [_rex(w=1, r=src >> 3, b=dst >> 3), 0x89, _modrm(3, src, dst)]
        return self._emit(out, "mov %s, %s" % (REG_NAMES64[dst],
                                               REG_NAMES64[src]))

    def mov_r32_imm32(self, dst, imm):
        out = []
        if dst >= 8:
            out.append(_rex(b=1))
        out.append(0xB8 + (dst & 7))
        out += list(struct.pack("<I", imm & 0xFFFFFFFF))
        return self._emit(out, "mov %s, %d" % (REG_NAMES32[dst], imm))

    def movabs(self, dst, imm):
        """``mov r64, imm64``: how a raw address gets into a register."""
        out = [_rex(w=1, b=dst >> 3), 0xB8 + (dst & 7)]
        out += list(struct.pack("<Q", imm & 0xFFFFFFFFFFFFFFFF))
        return self._emit(out, "movabs %s, 0x%x" % (REG_NAMES64[dst], imm))

    def xor_r32_r32(self, dst, src):
        out = []
        if dst >= 8 or src >= 8:
            out.append(_rex(r=src >> 3, b=dst >> 3))
        out += [0x31, _modrm(3, src, dst)]
        return self._emit(out, "xor %s, %s" % (REG_NAMES32[dst],
                                               REG_NAMES32[src]))

    def inc_r64(self, reg):
        out = [_rex(w=1, b=reg >> 3), 0xFF, _modrm(3, 0, reg)]
        return self._emit(out, "inc %s" % REG_NAMES64[reg])

    def cmp_r64_r64(self, a, b):
        out = [_rex(w=1, r=b >> 3, b=a >> 3), 0x39, _modrm(3, b, a)]
        return self._emit(out, "cmp %s, %s" % (REG_NAMES64[a], REG_NAMES64[b]))

    def cmp_r32_imm8(self, reg, imm):
        """``cmp r32, imm8`` with the immediate sign-extended.

        Class indices are small, so this is the encoding the dispatch chain
        uses; anything wider would just make the listing harder to read.
        """
        if not -128 <= imm <= 127:
            raise ValueError("immediate %d does not fit in imm8" % imm)
        out = []
        if reg >= 8:
            out.append(_rex(b=1))
        out += [0x83, _modrm(3, 7, reg), imm & 0xFF]
        return self._emit(out, "cmp %s, %d" % (REG_NAMES32[reg], imm))

    def movzx_r32_byte_base_index(self, dst, base, index):
        """``movzx r32, byte ptr [base + index]``.

        The one memory access in the whole engine. Scale is 1 because the
        subject is bytes and the class table is bytes; there is nothing else to
        load.
        """
        if index == RSP:
            raise ValueError("rsp cannot be an index register")
        if (base & 7) == RBP:
            raise ValueError("base %s needs a displacement form"
                             % REG_NAMES64[base])
        out = []
        rex = _rex(r=dst >> 3, x=index >> 3, b=base >> 3)
        if rex != 0x40:
            out.append(rex)
        out += [0x0F, 0xB6, _modrm(0, dst, 4), _sib(0, index, base)]
        return self._emit(out, "movzx %s, byte [%s+%s]"
                          % (REG_NAMES32[dst], REG_NAMES64[base],
                             REG_NAMES64[index]))

    def ret(self):
        return self._emit([0xC3], "ret")

    # -- branches ------------------------------------------------------------
    def jmp(self, name):
        self._emit([0xE9, 0, 0, 0, 0], "jmp %s" % name)
        self.fixups.append((len(self.buf) - 4, len(self.buf), name))
        return self

    def jcc(self, cond, name):
        if cond not in CC:
            raise ValueError("unknown condition %r" % cond)
        self._emit([0x0F, 0x80 + CC[cond], 0, 0, 0, 0], "j%s %s" % (cond, name))
        self.fixups.append((len(self.buf) - 4, len(self.buf), name))
        return self

    # -- finish --------------------------------------------------------------
    def assemble(self):
        """Patch every branch and return the finished bytes."""
        out = bytearray(self.buf)
        for site, after, name in self.fixups:
            if name not in self.labels:
                raise LabelError("branch to undefined label %r" % name)
            rel = self.labels[name] - after
            if not -(1 << 31) <= rel < (1 << 31):
                raise LabelError("branch to %r out of range" % name)
            out[site:site + 4] = struct.pack("<i", rel)
        return bytes(out)

    def text(self):
        """The listing, as the assembler intended it.

        Branch targets are printed as label names, so this stays readable even
        though the encoded form holds a relative displacement.
        """
        rev = {}
        for name, off in self.labels.items():
            rev.setdefault(off, []).append(name)
        lines = []
        for off, length, insn in self.listing:
            for name in sorted(rev.get(off, [])):
                lines.append("%s:" % name)
            lines.append("  %04x  %s" % (off, insn))
        for name in sorted(rev.get(len(self.buf), [])):
            lines.append("%s:" % name)
        return "\n".join(lines)
