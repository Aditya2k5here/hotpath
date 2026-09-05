"""An independent decoder for the instructions :mod:`hp.jit.asm` emits.

Independent is the point. The assembler builds bytes from a mnemonic; this
builds a mnemonic from bytes, by a separate reading of the encoding rules. The
test suite assembles a program, disassembles the result, and requires the two
texts to match. A REX bit in the wrong field, a ModRM ``reg`` and ``rm`` swapped,
the classic ways to get x86 wrong. Produce a text mismatch in a unit test
instead of a wrong answer or a crash inside generated code, where there is no
debugger and no stack trace.

It also means ``hotpath compile <pattern>`` can print the machine code it
generated on any machine, with no objdump, no capstone, and no assumption that
the reader is on the same platform the code was generated for.
"""

import struct

from .asm import REG_NAMES32, REG_NAMES64, CC

CC_BY_CODE = {v: k for k, v in CC.items()}


class DecodeError(Exception):
    pass


def decode_one(code, i):
    """Decode one instruction at ``code[i]``; return ``(text, length)``.

    ``text`` uses a numeric branch displacement; :func:`disassemble` turns those
    into label names once it knows where every instruction begins.
    """
    start = i
    rex = 0
    if 0x40 <= code[i] <= 0x4F:
        rex = code[i]
        i += 1
    w = (rex >> 3) & 1
    r = (rex >> 2) & 1
    x = (rex >> 1) & 1
    b = rex & 1

    op = code[i]
    i += 1

    if op == 0xC3:
        return "ret", i - start

    if 0x50 <= op <= 0x57:
        return "push %s" % REG_NAMES64[(op - 0x50) | (b << 3)], i - start

    if 0x58 <= op <= 0x5F:
        return "pop %s" % REG_NAMES64[(op - 0x58) | (b << 3)], i - start

    if 0xB8 <= op <= 0xBF:
        reg = (op - 0xB8) | (b << 3)
        if w:
            imm = struct.unpack_from("<Q", code, i)[0]
            return "movabs %s, 0x%x" % (REG_NAMES64[reg], imm), i + 8 - start
        imm = struct.unpack_from("<I", code, i)[0]
        return "mov %s, %d" % (REG_NAMES32[reg], imm), i + 4 - start

    if op == 0xE9:
        rel = struct.unpack_from("<i", code, i)[0]
        return "jmp %+d" % rel, i + 4 - start

    if op == 0x0F:
        op2 = code[i]
        i += 1
        if 0x80 <= op2 <= 0x8F:
            rel = struct.unpack_from("<i", code, i)[0]
            return "j%s %+d" % (CC_BY_CODE[op2 - 0x80], rel), i + 4 - start
        if op2 == 0xB6:
            modrm = code[i]
            i += 1
            mod, reg, rm = modrm >> 6, ((modrm >> 3) & 7) | (r << 3), modrm & 7
            if mod != 0 or rm != 4:
                raise DecodeError("unsupported movzx form at %d" % start)
            sib = code[i]
            i += 1
            index = ((sib >> 3) & 7) | (x << 3)
            base = (sib & 7) | (b << 3)
            return ("movzx %s, byte [%s+%s]"
                    % (REG_NAMES32[reg], REG_NAMES64[base], REG_NAMES64[index]),
                    i - start)
        raise DecodeError("unsupported 0F opcode %02x at %d" % (op2, start))

    if op in (0x31, 0x39, 0x89, 0x83, 0xFF):
        modrm = code[i]
        i += 1
        mod = modrm >> 6
        if mod != 3:
            raise DecodeError("unsupported memory form at %d" % start)
        reg = ((modrm >> 3) & 7) | (r << 3)
        rm = (modrm & 7) | (b << 3)
        if op == 0x31:
            return "xor %s, %s" % (REG_NAMES32[rm], REG_NAMES32[reg]), i - start
        if op == 0x39:
            return "cmp %s, %s" % (REG_NAMES64[rm], REG_NAMES64[reg]), i - start
        if op == 0x89:
            return "mov %s, %s" % (REG_NAMES64[rm], REG_NAMES64[reg]), i - start
        if op == 0x83:
            if (modrm >> 3) & 7 != 7:
                raise DecodeError("only /7 (cmp) is supported for 83 at %d"
                                  % start)
            imm = struct.unpack_from("<b", code, i)[0]
            return "cmp %s, %d" % (REG_NAMES32[rm], imm), i + 1 - start
        if op == 0xFF:
            if (modrm >> 3) & 7 != 0:
                raise DecodeError("only /0 (inc) is supported for FF at %d"
                                  % start)
            return "inc %s" % REG_NAMES64[rm], i - start

    raise DecodeError("unknown opcode %02x at offset %d" % (op, start))


def decode_all(code):
    """Every instruction in ``code`` as ``(offset, length, text)``."""
    out = []
    i = 0
    while i < len(code):
        text, length = decode_one(code, i)
        out.append((i, length, text))
        i += length
    return out


def disassemble(code, labels=None):
    """Render ``code`` the way the assembler's own listing renders it.

    Branch displacements are rewritten as label names when ``labels`` (a
    ``{name: offset}`` mapping) is supplied, and as absolute offsets otherwise,
    so the result can be compared with :meth:`hp.jit.asm.Assembler.text`
    directly.
    """
    insns = decode_all(code)
    by_offset = {}
    if labels:
        for name, off in labels.items():
            by_offset.setdefault(off, []).append(name)

    lines = []
    for off, length, text in insns:
        for name in sorted(by_offset.get(off, [])):
            lines.append("%s:" % name)
        mnemonic = text.split(" ", 1)[0]
        if mnemonic == "jmp" or (mnemonic[:1] == "j" and mnemonic[1:] in CC):
            head, rel = text.rsplit(" ", 1)
            target = off + length + int(rel)
            names = by_offset.get(target)
            text = "%s %s" % (head, names[0] if names else "0x%x" % target)
        lines.append("  %04x  %s" % (off, text))
    for name in sorted(by_offset.get(len(code), [])):
        lines.append("%s:" % name)
    return "\n".join(lines)


def hexdump(code, width=16):
    lines = []
    for off in range(0, len(code), width):
        chunk = code[off:off + width]
        lines.append("%04x  %s" % (off, " ".join("%02x" % b for b in chunk)))
    return "\n".join(lines)
