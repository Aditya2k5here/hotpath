"""Executable memory: mmap, mprotect, and a callable pointer to the result.

Three details here are not incidental.

**W^X.** The region is mapped writable, filled, and only then flipped to
read+execute. It is never both at once. That is the discipline hardened kernels
and every browser JIT enforce, and doing it here costs one extra syscall.

**The data table lives in the same mapping as the code.** The byte-class table
is 256 bytes of pure data placed at the base of the region, with code after it.
Same mapping means one allocation and one address to know; putting data *before*
code means the code's own addresses do not move when the table changes size.

**Code is generated twice.** The generated code needs the table's absolute
address in a ``movabs``, and the address is not known until the mapping exists,
and the mapping's size is not known until the code is generated. The loop is
broken by generating once against a dummy address purely to learn the size, then
again against the real one -- ``movabs`` is fixed-width, so the second pass is
byte-for-byte the same length, and :func:`allocate` asserts it rather than
trusting it.

This module is Unix-only: it is the one part of the project that cannot run on
Windows. Everything else -- parsing, the automata, code *generation*, and the
disassembler that prints the result -- is pure Python and runs anywhere, which
is deliberate. You can inspect the machine code hotpath generates on a machine
that cannot execute it.
"""

import ctypes
import ctypes.util
import mmap as _mmap
import platform

PROT_READ = 0x1
PROT_WRITE = 0x2
PROT_EXEC = 0x4
MAP_PRIVATE = 0x02
MAP_ANONYMOUS = 0x20 if platform.system() == "Linux" else 0x1000

PAGE = _mmap.PAGESIZE


class JitUnavailable(Exception):
    """This platform cannot execute generated code.

    Raised rather than silently degrading, because a tiering policy that thinks
    it promoted a pattern to machine code when it did not would produce results
    that are quietly meaningless.
    """


_LIBC = []


def _libc():
    """Resolved once and cached.

    ``ctypes.util.find_library`` shells out to ``ldconfig`` (or worse, a
    compiler) on some systems. Doing that per compilation put three
    milliseconds into every JIT promotion and made the tiering measurements
    a study of ``ldconfig``.
    """
    if not _LIBC:
        name = ctypes.util.find_library("c")
        if name is None:                                # pragma: no cover
            raise JitUnavailable("libc not found")
        lib = ctypes.CDLL(name, use_errno=True)
        lib.mmap.restype = ctypes.c_void_p
        lib.mmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int,
                             ctypes.c_int, ctypes.c_int, ctypes.c_long]
        lib.mprotect.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
        lib.munmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
        _LIBC.append(lib)
    return _LIBC[0]


_AVAILABLE = []


def available():
    """Can this machine execute generated code? Probed once, by trying it.

    Cached, because the answer cannot change during a run and the probe is a
    pair of syscalls -- and this is consulted on every promotion.
    """
    if _AVAILABLE:
        return _AVAILABLE[0]
    ok = True
    if platform.system() not in ("Linux", "Darwin"):
        ok = False
    elif platform.machine() not in ("x86_64", "AMD64"):
        ok = False
    else:
        try:
            block = ExecutableBlock(PAGE)
            block.free()
        except Exception:                               # pragma: no cover
            ok = False
    _AVAILABLE.append(ok)
    return ok


class ExecutableBlock(object):
    """One mapping holding a data table followed by machine code."""

    def __init__(self, size):
        self.libc = _libc()
        self.size = (size + PAGE - 1) // PAGE * PAGE
        addr = self.libc.mmap(None, self.size, PROT_READ | PROT_WRITE,
                              MAP_PRIVATE | MAP_ANONYMOUS, -1, 0)
        if not addr or addr == ctypes.c_void_p(-1).value:
            raise JitUnavailable("mmap failed (errno %d)"
                                 % ctypes.get_errno())
        self.addr = addr
        self.executable = False
        self.freed = False

    def write(self, offset, data):
        if self.executable:
            raise RuntimeError("block is already executable; it is W^X")
        ctypes.memmove(self.addr + offset, bytes(data), len(data))

    def make_executable(self):
        if self.libc.mprotect(ctypes.c_void_p(self.addr), self.size,
                              PROT_READ | PROT_EXEC) != 0:  # pragma: no cover
            raise JitUnavailable("mprotect failed (errno %d)"
                                 % ctypes.get_errno())
        self.executable = True

    def function(self, offset, restype, argtypes):
        proto = ctypes.CFUNCTYPE(restype, *argtypes)
        return proto(self.addr + offset)

    def free(self):
        if not self.freed:
            self.libc.munmap(ctypes.c_void_p(self.addr), self.size)
            self.freed = True

    def __del__(self):                                  # pragma: no cover
        try:
            self.free()
        except Exception:
            pass
