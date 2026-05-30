"""Pass 1: mechanical lift from a parsed Merlin file to IR1.

Scope of the current implementation: the AUTO.S combat-button pilot —
the smallest set of routines that exercises multi-entry labels (e.g.
`DoBlock` / `DoUp` on consecutive lines), `#-1` / `#0` immediate stores,
unconditional cross-routine `jmp` (tail calls), and fall-through into
the shared `]rts` trampoline.

What works:

* Routine discovery: the caller passes a list of entry-point labels.
  The lifter walks forward from each one and additionally chases any
  `jmp` to a global-style label name as a tail-call target, so the
  full reachable set inside the file gets lifted. Reachability via
  fall-through across `rts` boundaries is deliberately *not* done here
  — pass 2's CFG analysis owns that.
* Multiple labels above one instruction collapse into a single routine
  with `entry_aliases` populated.
* Opcodes lifted: `lda/ldx/ldy #imm`, `sta/stx/sty abs`, `rts`, `jmp`.
* Everything else becomes an `Unsupported` IR item with the original
  mnemonic and operand preserved, so dumps still line up 1:1 with the
  source and pass 2 can report what's left.

Out of scope until the next slice (`rndp`, `CheckFloor`):

* Indexed addressing modes (`,x`, `,y`), indirect-indexed `(ptr),y`,
  conditional branches, `cmp` / `cpx` / `cpy`, `jsr`, self-modifying
  code, 16-bit add/sub patterns.

The lifter is intentionally lossy on flags and arithmetic — pass 1's
job is to translate one instruction at a time and leave structure to
pass 2.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, replace
from pathlib import Path

from .ir1 import (
    Abs,
    AdcAbs,
    AdcImm,
    AdcIndexed,
    Asl,
    Bit,
    Bitwise,
    CmpIndexed,
    CmpLocal,
    IndexedAbs,
    LoadLocal,
    MemBitOp,
    Pha,
    Phy,
    Pla,
    Rol,
    Ror,
    SbcIndexed,
    ShiftMem,
    Branch,
    Call,
    Clc,
    CmpAbs,
    CmpImm,
    CmpIndirect,
    DecTarget,
    FlagOp,
    Goto,
    Imm,
    IncTarget,
    IndirectX,
    IndirectY,
    Label,
    LoadAbs,
    LoadImm,
    LoadIndexed,
    LoadIndirect,
    LocalRef,
    Lsr,
    ModuleIR1,
    Nop,
    Reg,
    Return,
    Routine,
    SbcAbs,
    SbcImm,
    SbcIndirect,
    Sec,
    SourceRef,
    StoreAbs,
    StoreIndexed,
    StoreIndirect,
    StoreLocal,
    Transfer,
    Unsupported,
)
from .pass0_lex import Line
from .pass0_parse import FileAST, eval_expr


# Opcodes that unconditionally end a routine. `bra` is the Merlin 16+
# unconditional short branch — listed for completeness even though the
# upstream source mostly uses `jmp`.
_TERMINATORS = frozenset({"rts", "rti", "jmp", "bra"})

# Conditional-assembly directives. They emit no bytes, so a label sitting
# on one floats to the next assembled instruction (e.g. `:stripe do
# EditorDisk` with EditorDisk=0 — `:stripe` names the code after `fin`).
_COND_DIRECTIVES = frozenset({"do", "else", "fin"})

# Lines that are not "real" instructions for routine-walking purposes.
# We skip them silently — they belong either to the equate header of the
# file (already consumed by pass 0) or to the data sections that pass 1
# leaves to a later pass.
_NON_CODE_DIRECTIVES = frozenset(
    {
        "=", "org", "put", "dum", "dend", "ds", "db", "dw", "ddb", "hex",
        "asc", "dfb", "dci", "str", "lst", "tr", "xc", "mx", "ent", "ext",
        "use", "rel", "obj", "sav", "lup", "--^", "if", "do", "else",
        "fin", "mac", "eom", "<<<", ">>>",
        # Merlin data pseudo-ops that emit inert literal bytes:
        #   `rev "STR"` — the ASCII of STR, reversed (POP's cheat-code
        #     table: `C_skip rev "SKIP"`, `C_devel rev "POP"`, ...).
        #   `da expr`   — "define address", a 2-byte little-endian
        #     word (same as `dw`).
        # Treating them as non-code stops `discover_entries` from
        # mistaking a data label (e.g. `C_skip`) for a routine entry,
        # and keeps inline data from surfacing as `??? rev`.
        #
        # NOTE: `usr` is deliberately NOT here. It's POP's wiring of
        # Merlin's user-defined hook to the RW18 fast disk routine
        # (see `04 Support/MakeDisk/USR18.S`): at pass 2 the hook reads
        # the just-assembled module out of `obj_buf` and writes it to a
        # disk track. A pure build-time side effect — zero bytes emitted
        # into the binary — but a *real* directive with real arguments
        # we'd lose entirely if we silently skipped it. Let it fall
        # through to `Unsupported(mnemonic="usr", …)` so the IR dump and
        # pass-4 Rust emission can render it as a documenting comment
        # (`format_item` and `_emit_iigs_unsupported` recognise the
        # mnemonic and emit a clear "build-time RW18 disk write" line).
        # All POP `usr` calls are unlabeled and sit after `rts`, so
        # `Unsupported` here doesn't reintroduce the data-label-as-entry
        # discovery bug.
        "rev", "da",
    }
)


@dataclass
class LiftReport:
    """Summary of what pass 1 did. Mainly for tests and CLI output."""

    module: ModuleIR1
    unsupported: list[Unsupported]  # all unsupported instructions across routines


# ---------------------------------------------------------------- helpers


def _is_local_label(name: str) -> bool:
    """Merlin local label forms: `:foo` (scope = enclosing global) and
    `]foo` (macro-style)."""
    return name.startswith(":") or name.startswith("]")


def _parse_hex_jsl(line: Line) -> str | None:
    """Return the 24-bit target of an IIgs `jsl` hand-assembled as
    `hex 22,lo,mid,hi`, or `None` if `line` isn't that pattern. POP
    embeds the control-panel toolbox calls this way (`hex 22,00,00,E1`
    = `jsl $E10000`) because the assembler builds the 6502 only —
    surfacing them as `Unsupported(jsl)` keeps them visible in dumps."""
    if line.mnemonic != "hex" or not line.operand:
        return None
    parts = [p.strip() for p in line.operand.split(",")]
    if len(parts) != 4:
        return None
    try:
        bytes_ = [int(p, 16) for p in parts]
    except ValueError:
        return None
    if bytes_[0] != 0x22 or any(b < 0 or b > 0xff for b in bytes_):
        return None
    lo, mid, hi = bytes_[1], bytes_[2], bytes_[3]
    return f"${(hi << 16) | (mid << 8) | lo:06x}"


def _parse_immediate(operand: str, equates: dict[str, int]) -> Imm | None:
    """Parse a `#expr` immediate operand. Returns `None` if `operand`
    isn't a `#`-prefixed immediate (the caller decides what to do).

    Merlin's `<expr` / `>expr` prefixes select the low / high byte of
    `expr`. We previously stripped these because the pilot operands
    were already byte-sized, but now that pass 0 collects label
    addresses (which are 16-bit values in the 0x10000+ synthetic
    range), the operator matters: `#<Label` should give the low
    byte of the synthetic address, `#>Label` the high byte. Both
    are then masked into the byte the lifted `LoadImm` will store
    into A/X/Y."""
    s = operand.strip()
    if not s.startswith("#"):
        return None
    expr = s[1:].lstrip()
    byte_op: str | None = None
    if expr.startswith("<"):
        byte_op = "low"
        expr = expr[1:].lstrip()
    elif expr.startswith(">"):
        byte_op = "high"
        expr = expr[1:].lstrip()
    try:
        value = eval_expr(expr, equates)
    except ValueError:
        return None
    if byte_op == "low":
        value = value & 0xff
    elif byte_op == "high":
        value = (value >> 8) & 0xff
    return Imm(value=value, text=s)


def _parse_absolute(operand: str, equates: dict[str, int]) -> Abs | None:
    """Parse a plain absolute / zero-page operand: just an expression
    that resolves to an address. Indexed forms (`,x` / `,y`) and indirect
    forms (`(...)`) are rejected — they need a richer operand type."""
    s = operand.strip()
    if not s or s.startswith("#"):
        return None
    if "," in s or s.startswith("("):
        return None
    try:
        addr = eval_expr(s, equates)
    except ValueError:
        return None
    return Abs(name=s, addr=addr & 0xffff)


def _parse_indexed(
    operand: str,
    equates: dict[str, int],
) -> tuple[Abs, Reg] | None:
    """Parse `expr,x` or `expr,y` — the 6502 indexed-absolute / indexed-
    zero-page form. Returns the base address and the index register, or
    `None` if the operand isn't a comma-suffixed form we recognise."""
    s = operand.strip()
    if "," not in s or s.startswith("("):
        return None
    base_str, _, idx_str = s.rpartition(",")
    base_str = base_str.strip()
    idx_str = idx_str.strip().lower()
    if idx_str == "x":
        idx_reg = Reg.X
    elif idx_str == "y":
        idx_reg = Reg.Y
    else:
        return None
    try:
        addr = eval_expr(base_str, equates)
    except ValueError:
        return None
    return Abs(name=base_str, addr=addr & 0xffff), idx_reg


_LOCAL_TARGET_RE = re.compile(
    # `:label`, `]label`, optionally `+N` (N a decimal byte offset).
    # `:` = Merlin local label, `]` = Merlin macro-local label.
    r"^([:\]][A-Za-z0-9_]+)\s*(?:\+\s*(\d+))?$"
)


def _parse_local_target(operand: str) -> tuple[str, int] | None:
    """Parse `:label+N` / `]label+N` / `:label` — a store target
    addressed relative to a *local* label. Returns `(label, offset)`
    or `None` if the operand isn't a local-label form.

    Used only by the store opcodes (sta/stx/sty) for self-modifying-
    code patches and local data stores; the address itself stays
    symbolic because local labels aren't in the resolved symbol
    table. See `ir1.StoreLocal`."""
    m = _LOCAL_TARGET_RE.match(operand.strip())
    if m is None:
        return None
    label = m.group(1)
    offset = int(m.group(2)) if m.group(2) is not None else 0
    return label, offset


def _parse_indirect_y(
    operand: str,
    equates: dict[str, int],
) -> IndirectY | None:
    """Parse `(name),y` — the 6502 post-indexed indirect form.
    Returns the resolved `IndirectY` or `None` if the operand doesn't
    match. POP only uses the `,y` variant; the `(zp,x)` pre-indexed
    form never appears in any of the source files we lift.

    The pointer's zero-page address is recorded on `IndirectY.ptr`
    as a normal `Abs` so dumps and downstream passes see the symbolic
    name. We don't enforce `ptr.addr < 0x100` here — Merlin's
    assembler accepts arbitrary expressions and the interpreter
    happens to work for any address; a real 6502 would only accept
    zero-page pointers, but the engine code is well-behaved on this
    front.
    """
    s = operand.strip()
    if not s.startswith("("):
        return None
    # Expect `(<name>),y` (case-insensitive on the `y`).
    close = s.find(")")
    if close < 0:
        return None
    inner = s[1:close].strip()
    # Normalise the tail to drop all whitespace + lowercase before
    # comparing — handles `,y`, `, y`, `,Y`, etc. in one pass.
    tail = s[close + 1:].replace(" ", "").replace("\t", "").lower()
    if tail != ",y":
        return None
    try:
        addr = eval_expr(inner, equates)
    except ValueError:
        return None
    return IndirectY(ptr=Abs(name=inner, addr=addr & 0xffff))


def _parse_indirect_x(
    operand: str,
    equates: dict[str, int],
) -> IndirectX | None:
    """Parse `(name,x)` — the 6502 pre-indexed indirect form, where the
    `,x` sits *inside* the parentheses (contrast `(name),y`). Returns the
    resolved `IndirectX` or `None`. The pointer's zero-page address is
    recorded on `IndirectX.ptr` as a normal `Abs` so dumps and downstream
    passes see the symbolic name."""
    s = operand.strip()
    if not s.startswith("(") or not s.endswith(")"):
        return None
    inner = s[1:-1]
    if inner.replace(" ", "").replace("\t", "").lower()[-2:] != ",x":
        return None
    name = inner[:inner.rfind(",")].strip()
    try:
        addr = eval_expr(name, equates)
    except ValueError:
        return None
    return IndirectX(ptr=Abs(name=name, addr=addr & 0xffff))


def _reg_of_load(mnemonic: str) -> Reg:
    return {"lda": Reg.A, "ldx": Reg.X, "ldy": Reg.Y}[mnemonic]


def _reg_of_store(mnemonic: str) -> Reg:
    return {"sta": Reg.A, "stx": Reg.X, "sty": Reg.Y}[mnemonic]


# ---------------------------------------------------------------- core lift


def _lift_instr(
    line: Line,
    equates: dict[str, int],
    entry_names: set[str],
):
    """Produce a single IR1 instruction for `line`, or `None` if the
    line is not code (directives, blanks, label-only lines). Caller
    handles the `None` case."""
    mnemonic = line.mnemonic
    if mnemonic is None or mnemonic in _NON_CODE_DIRECTIVES:
        return None

    src = SourceRef(
        file=str(line.file), line=line.lineno, raw=line.raw.rstrip("\n"),
        comment=line.comment,
    )

    if mnemonic == "rts":
        return Return(src=src)

    if mnemonic == "rti":
        # 6502 `rti` pops the processor status flags and the return PC.
        # POP uses it only in interrupt handlers (e.g. `GRAFIX/VBLI`,
        # reached via the $FFFE vector). Our model doesn't simulate IRQs,
        # so these routines are never entered through normal control
        # flow; lifting `rti` as `Return` is the closest honest
        # approximation and keeps the routine structured rather than
        # carrying an opaque `Unsupported` atom.
        return Return(src=src)

    if mnemonic == "jmp":
        if not line.operand:
            return Unsupported(mnemonic=mnemonic, operand=None, src=src)
        target = line.operand.strip()
        if _is_local_label(target):
            kind = "local"
        else:
            # Any non-local label name is treated as an external/tail-call
            # target. Pass 2 will refine this against the full call graph
            # — for now the IR1 interpreter resolves the name against the
            # module's routines and errors loudly if absent.
            kind = "tail_call"
        return Goto(target=target, kind=kind, src=src)

    if mnemonic in ("lda", "ldx", "ldy"):
        if line.operand is None:
            return Unsupported(mnemonic=mnemonic, operand=None, src=src)
        imm = _parse_immediate(line.operand, equates)
        if imm is not None:
            return LoadImm(reg=_reg_of_load(mnemonic), imm=imm, src=src)
        idx = _parse_indexed(line.operand, equates)
        if idx is not None:
            base, idx_reg = idx
            return LoadIndexed(
                reg=_reg_of_load(mnemonic), base=base, index=idx_reg, src=src,
            )
        # `lda` is the only *load mnemonic* that has a `(zp),y` form
        # — `ldx`/`ldy` don't. (The addressing mode itself exists for
        # plenty of other opcodes — `sta`, `cmp`, `adc`, `sbc`,
        # `and`, `ora`, `eor` — those are dispatched in their own
        # mnemonic branches below.) Try the indirect parse before
        # the plain-absolute parse so we don't mis-resolve `(name)`
        # as an absolute expression.
        if mnemonic == "lda":
            ind = _parse_indirect_y(line.operand, equates)
            if ind is not None:
                return LoadIndirect(reg=Reg.A, source=ind, src=src)
            indx = _parse_indirect_x(line.operand, equates)
            if indx is not None:
                return LoadIndirect(reg=Reg.A, source=indx, src=src)
        addr = _parse_absolute(line.operand, equates)
        if addr is not None:
            return LoadAbs(reg=_reg_of_load(mnemonic), source=addr, src=src)
        # `lda :sm` / `ldx ]temp1` / `lda ]cleanflag` — load from a
        # local-label / Merlin-variable address. The read counterpart to
        # StoreLocal; tried last so resolved symbols take the abs path.
        local = _parse_local_target(line.operand)
        if local is not None:
            label, offset = local
            return LoadLocal(
                reg=_reg_of_load(mnemonic),
                source_label=label,
                offset=offset,
                src=src,
            )
        return Unsupported(mnemonic=mnemonic, operand=line.operand, src=src)

    if mnemonic in ("sta", "stx", "sty"):
        if line.operand is None:
            return Unsupported(mnemonic=mnemonic, operand=None, src=src)
        idx = _parse_indexed(line.operand, equates)
        if idx is not None:
            base, idx_reg = idx
            return StoreIndexed(
                reg=_reg_of_store(mnemonic), base=base, index=idx_reg, src=src,
            )
        # Same indirect-indexed treatment as `lda` above — `sta` is
        # the only store with a `(ptr),y` form on stock 6502.
        if mnemonic == "sta":
            ind = _parse_indirect_y(line.operand, equates)
            if ind is not None:
                return StoreIndirect(reg=Reg.A, target=ind, src=src)
        addr = _parse_absolute(line.operand, equates)
        if addr is not None:
            return StoreAbs(reg=_reg_of_store(mnemonic), target=addr, src=src)
        # `sta :smXCO+1` / `sta :buffer` — store to a local-label-
        # relative address. Covers self-modifying-code operand
        # patches (the `+N` form, common in HIRES.S) and plain
        # stores to local data labels (`+0`). Tried last so a label
        # that *also* resolves as an equate/abs takes the normal
        # StoreAbs path.
        local = _parse_local_target(line.operand)
        if local is not None:
            label, offset = local
            return StoreLocal(
                reg=_reg_of_store(mnemonic),
                target_label=label,
                offset=offset,
                src=src,
            )
        return Unsupported(mnemonic=mnemonic, operand=line.operand, src=src)

    if mnemonic in ("cmp", "cpx", "cpy"):
        if line.operand is None:
            return Unsupported(mnemonic=mnemonic, operand=None, src=src)
        reg = {"cmp": Reg.A, "cpx": Reg.X, "cpy": Reg.Y}[mnemonic]
        imm = _parse_immediate(line.operand, equates)
        if imm is not None:
            return CmpImm(reg=reg, imm=imm, src=src)
        # `(ptr),y` only exists for `cmp` (not `cpx`/`cpy`).
        if mnemonic == "cmp":
            ind = _parse_indirect_y(line.operand, equates)
            if ind is not None:
                return CmpIndirect(reg=Reg.A, source=ind, src=src)
        # `cmp tbl,x` / `cmp tbl,y` — indexed-absolute. Only `cmp`
        # has the `,y` form on stock 6502; `cpx`/`cpy` don't have
        # indexed addressing at all (zero-page only, no `,y`).
        if mnemonic == "cmp":
            idx = _parse_indexed(line.operand, equates)
            if idx is not None:
                base, idx_reg = idx
                return CmpIndexed(
                    reg=Reg.A, base=base, index=idx_reg, src=src,
                )
        addr = _parse_absolute(line.operand, equates)
        if addr is not None:
            return CmpAbs(reg=reg, source=addr, src=src)
        # `cpx :pitch+1` / `cpy :pitch` — compare against a local-label
        # byte (read from the same store StoreLocal writes).
        local = _parse_local_target(line.operand)
        if local is not None:
            label, offset = local
            return CmpLocal(reg=reg, source_label=label, offset=offset, src=src)
        return Unsupported(mnemonic=mnemonic, operand=line.operand, src=src)

    if mnemonic in ("beq", "bne", "bcc", "bcs", "bpl", "bmi", "bvc", "bvs"):
        if not line.operand:
            return Unsupported(mnemonic=mnemonic, operand=None, src=src)
        cond = mnemonic[1:]      # strip the leading `b`
        return Branch(cond=cond, target=line.operand.strip(), src=src)

    if mnemonic == "bra":
        # 65C02 "branch always" — an unconditional relative branch.
        # Semantically identical to a local `jmp`, so it lowers to the
        # same `Goto`. (POP is mostly NMOS, but `bra` shows up in a
        # couple of routines / macros.)
        if not line.operand:
            return Unsupported(mnemonic=mnemonic, operand=None, src=src)
        target = line.operand.strip()
        kind = "local" if _is_local_label(target) else "tail_call"
        return Goto(target=target, kind=kind, src=src)

    if mnemonic == "jsr":
        if not line.operand:
            return Unsupported(mnemonic=mnemonic, operand=None, src=src)
        return Call(target=line.operand.strip(), src=src)

    if mnemonic in ("asl", "lsr", "rol", "ror"):
        # Accumulator form: `<op>` (no operand) or `<op> a` → Asl /
        # Lsr / Rol / Ror. Memory form: `<op> addr` → ShiftMem with
        # the same op. POP uses both extensively (the 16-bit-shift
        # idiom `asl lo ; rol hi` is a memory pair).
        op = (line.operand or "").strip().lower()
        if op in ("", "a"):
            return {
                "asl": Asl,
                "lsr": Lsr,
                "rol": Rol,
                "ror": Ror,
            }[mnemonic](src=src)
        addr = _parse_absolute(line.operand, equates)
        if addr is not None:
            return ShiftMem(op=mnemonic, target=addr, src=src)
        return Unsupported(mnemonic=mnemonic, operand=line.operand, src=src)

    if mnemonic in ("tsb", "trb"):
        # 65C02 test-and-set / test-and-reset bits in memory. Absolute
        # operand only (POP uses `tsb/trb $C036` for the IIgs speed reg).
        if line.operand is None:
            return Unsupported(mnemonic=mnemonic, operand=None, src=src)
        addr = _parse_absolute(line.operand, equates)
        if addr is not None:
            return MemBitOp(op=mnemonic, target=addr, src=src)
        return Unsupported(mnemonic=mnemonic, operand=line.operand, src=src)

    if mnemonic == "clc":
        return Clc(src=src)

    if mnemonic == "sec":
        return Sec(src=src)

    if mnemonic == "nop":
        return Nop(src=src)

    # Processor-status flag set/clear that we model as flag-only,
    # observable-but-unread ops (see `ir1.FlagOp`).
    if mnemonic in ("sei", "cli", "sed", "cld"):
        flag, value = {
            "sei": ("I", 1),
            "cli": ("I", 0),
            "sed": ("D", 1),
            "cld": ("D", 0),
        }[mnemonic]
        return FlagOp(flag=flag, value=value, src=src)

    if mnemonic == "adc":
        if line.operand is None:
            return Unsupported(mnemonic=mnemonic, operand=None, src=src)
        imm = _parse_immediate(line.operand, equates)
        if imm is not None:
            return AdcImm(imm=imm, src=src)
        idx = _parse_indexed(line.operand, equates)
        if idx is not None:
            base, idx_reg = idx
            return AdcIndexed(base=base, index=idx_reg, src=src)
        addr = _parse_absolute(line.operand, equates)
        if addr is not None:
            return AdcAbs(source=addr, src=src)
        return Unsupported(mnemonic=mnemonic, operand=line.operand, src=src)

    # `sbc` mirrors `adc` — same operand shape, opposite arithmetic
    # direction. We don't try to lift `sbc (zp),y` here because the
    # indirect-indexed form pairs with the `(zp),y` slice that hasn't
    # extended SbcAbs into an `Sbc + IndirectY` shape yet.
    if mnemonic == "sbc":
        if line.operand is None:
            return Unsupported(mnemonic=mnemonic, operand=None, src=src)
        imm = _parse_immediate(line.operand, equates)
        if imm is not None:
            return SbcImm(imm=imm, src=src)
        idx = _parse_indexed(line.operand, equates)
        if idx is not None:
            base, idx_reg = idx
            return SbcIndexed(base=base, index=idx_reg, src=src)
        ind = _parse_indirect_y(line.operand, equates)
        if ind is not None:
            return SbcIndirect(source=ind, src=src)
        addr = _parse_absolute(line.operand, equates)
        if addr is not None:
            return SbcAbs(source=addr, src=src)
        return Unsupported(mnemonic=mnemonic, operand=line.operand, src=src)

    if mnemonic == "bit":
        # `bit` accepts both `#imm` (rare; some assemblers don't even
        # accept it) and absolute. We try both.
        if line.operand is None:
            return Unsupported(mnemonic=mnemonic, operand=None, src=src)
        imm = _parse_immediate(line.operand, equates)
        if imm is not None:
            return Bit(source=imm, src=src)
        addr = _parse_absolute(line.operand, equates)
        if addr is not None:
            return Bit(source=addr, src=src)
        return Unsupported(mnemonic=mnemonic, operand=line.operand, src=src)

    # Stack push/pop. Single-byte opcodes — no operand. Modelled
    # against `Trace.value_stack` rather than the actual hardware
    # stack at `$0100..$01ff`; see the `Pha` docstring for the
    # caveats this implies for POP's source.
    if mnemonic == "pha":
        return Pha(src=src)
    if mnemonic == "pla":
        return Pla(src=src)
    if mnemonic == "phy":
        # 65C02 push-Y. No operand; rides the same value stack as pha.
        return Phy(src=src)

    # Index-register inc/dec — single-byte opcodes, no operand.
    if mnemonic in ("inx", "iny"):
        reg = Reg.X if mnemonic == "inx" else Reg.Y
        return IncTarget(target=reg, src=src)
    if mnemonic in ("dex", "dey"):
        reg = Reg.X if mnemonic == "dex" else Reg.Y
        return DecTarget(target=reg, src=src)

    # Memory inc/dec — single-operand against an absolute address.
    # POP doesn't use the zero-page-indexed form (`inc addr,x`) in code
    # paths we've lifted; mark those Unsupported when they surface.
    if mnemonic in ("inc", "dec"):
        if line.operand is None:
            return Unsupported(mnemonic=mnemonic, operand=None, src=src)
        node_cls = IncTarget if mnemonic == "inc" else DecTarget
        addr = _parse_absolute(line.operand, equates)
        if addr is not None:
            return node_cls(target=addr, src=src)
        # `inc joyX,x` / `dec pstarcount,x` — indexed memory inc/dec.
        idx = _parse_indexed(line.operand, equates)
        if idx is not None:
            base, idx_reg = idx
            return node_cls(target=IndexedAbs(base=base, index=idx_reg), src=src)
        # `inc :smod+2` — self-modifying-code operand bump (advancing
        # a patched pointer's high byte). Local-label target, address
        # stays symbolic. Tried after the absolute parse so resolved
        # symbols still take the memory path.
        local = _parse_local_target(line.operand)
        if local is not None:
            label, offset = local
            return node_cls(
                target=LocalRef(label=label, offset=offset), src=src,
            )
        return Unsupported(mnemonic=mnemonic, operand=line.operand, src=src)

    # Register transfers (`tax` / `tay` / `txa` / `tya`). `tsx`/`txs`
    # interact with the stack pointer — out of scope for now.
    if mnemonic in ("tax", "tay", "txa", "tya"):
        src_dst = {
            "tax": (Reg.A, Reg.X),
            "tay": (Reg.A, Reg.Y),
            "txa": (Reg.X, Reg.A),
            "tya": (Reg.Y, Reg.A),
        }[mnemonic]
        return Transfer(src_reg=src_dst[0], dst_reg=src_dst[1], src=src)

    # Bitwise on A — `and` / `ora` / `eor`. Each accepts both
    # immediate (`#imm`) and absolute (`addr`) forms; indirect-indexed
    # `(zp),y` and zero-page-X aren't in this slice. Indexed forms
    # (`and table,x`) would need a Bitwise-Indexed variant; defer.
    if mnemonic in ("and", "ora", "eor"):
        if line.operand is None:
            return Unsupported(mnemonic=mnemonic, operand=None, src=src)
        op_key = {"and": "and", "ora": "or", "eor": "eor"}[mnemonic]
        imm = _parse_immediate(line.operand, equates)
        if imm is not None:
            return Bitwise(op=op_key, source=imm, src=src)
        # `(ptr),y` form — `and`/`ora`/`eor` all have one. Try before
        # the plain-absolute parse for the same reason as `lda`.
        ind = _parse_indirect_y(line.operand, equates)
        if ind is not None:
            return Bitwise(op=op_key, source=ind, src=src)
        # Indexed-absolute: `and table,x`, `ora mask,y`, etc.
        idx = _parse_indexed(line.operand, equates)
        if idx is not None:
            base, idx_reg = idx
            return Bitwise(
                op=op_key,
                source=IndexedAbs(base=base, index=idx_reg),
                src=src,
            )
        addr = _parse_absolute(line.operand, equates)
        if addr is not None:
            return Bitwise(op=op_key, source=addr, src=src)
        return Unsupported(mnemonic=mnemonic, operand=line.operand, src=src)

    # All other opcodes are out of scope for this slice. Marking
    # `Unsupported` (instead of skipping) keeps the IR aligned with the
    # source and gives reviewers an exact list of what still needs work.
    del entry_names  # unused, reserved for future heuristics
    return Unsupported(mnemonic=mnemonic, operand=line.operand, src=src)


def discover_entries(file_ast: FileAST) -> list[str]:
    """Walk a parsed file and return every label that can plausibly
    serve as a routine entry: a global-style label (no `:` or `]`
    prefix) attached to a code line, in source order.

    The intent is feeding `lift_file` when the caller wants a full
    mechanical sweep instead of a hand-picked entry list. Labels on
    pure-data directives (`db`, `dw`, `hex`, `asc`, `ds`, `=`, etc.)
    are excluded — those don't introduce executable code and would
    otherwise produce all-`Unsupported` routines that just clutter the
    dump. Local (`:foo`) and macro (`]foo`) labels are excluded
    because they're internal jump targets, not callable entry points.

    A label that appears on a bare-label line gets attributed to the
    next code line. If two labels stack onto the same instruction
    (the `DoBlock` / `DoUp` pattern) both are returned; the lifter
    collapses them into a single routine with `entry_aliases`.
    """
    out: list[str] = []
    pending: list[str] = []
    for line in file_ast.lines:
        if line.is_blank:
            continue
        if line.label and line.mnemonic is None:
            if not _is_local_label(line.label):
                pending.append(line.label)
            continue
        if line.mnemonic is None or line.mnemonic in _NON_CODE_DIRECTIVES:
            # Non-code directive (e.g. `db`, `=`); any pending labels
            # belong to data, not code — drop them.
            pending.clear()
            continue
        # A code line.
        for lab in pending:
            if lab not in out:
                out.append(lab)
        pending.clear()
        if line.label and not _is_local_label(line.label):
            if line.label not in out:
                out.append(line.label)
    return out


def lift_file(
    file_ast: FileAST,
    equates: dict[str, int],
    entries: list[str],
) -> LiftReport:
    """Lift one parsed file. `entries` lists routine entry names the
    caller wants extracted; the lifter follows tail-call `jmp`s within
    the file transitively so any reachable callee is also lifted.

    Returns a `LiftReport` carrying the resulting `ModuleIR1` plus
    bookkeeping the CLI/tests use to summarise what happened.
    """
    file_path = Path(file_ast.path)
    module_name = file_path.stem.upper()
    module = ModuleIR1(name=module_name, file=str(file_path))

    entry_set = set(entries)
    requested: list[str] = list(entries)
    lifted_names: set[str] = set()
    all_unsupported: list[Unsupported] = []

    # Pre-index the lines so we can walk forward cheaply from any label.
    lines = file_ast.lines
    # Lines a false `do`/`else`/`fin` condition excludes from this build.
    inactive = file_ast.inactive_lines

    # Map label -> index in `lines` of the line whose *next* code
    # instruction the label refers to. Bare-label lines just attach
    # their label to the upcoming instruction.
    label_to_instr_index: dict[str, int] = {}
    pending_labels: list[str] = []
    for i, line in enumerate(lines):
        if line.is_blank or i in inactive:
            continue
        if line.label and (line.mnemonic is None or line.mnemonic in _NON_CODE_DIRECTIVES):
            # Bare-label line, or a label on a non-code directive. The
            # bare-label case is the one we care about for the pilot
            # (`DoBlock\n DoUp lda #-1`); the directive case can also
            # carry a label (e.g. equates) and we just ignore that
            # because pass 0 already absorbed it. A label on a
            # conditional-assembly directive (`:stripe do ...`) floats to
            # the next assembled instruction, the same as a bare label.
            if line.mnemonic is None or line.mnemonic in _COND_DIRECTIVES:
                pending_labels.append(line.label)
            continue
        if line.mnemonic is None:
            continue
        # A code line. Bind any pending labels plus this line's own label.
        labels_here = list(pending_labels)
        if line.label:
            labels_here.append(line.label)
        pending_labels.clear()
        for lab in labels_here:
            # If the same label appears twice (Merlin allows shadowing
            # via macro reuse — see the `]rts` trampolines), the *latest*
            # binding wins. That matches Merlin's pass-2 assemble order.
            label_to_instr_index[lab] = i

    def _extract_local_block(
        target: str, before_idx: int, defined: set[str]
    ) -> list | None:
        """Return the code block for local label `target`, resolved by its
        nearest definition at or before `before_idx`: a `Label(target)`
        followed by the lifted instructions through the block's terminator
        (inclusive), surfacing any intervening labels so chained branches
        resolve.

        Merlin clusters routines after shared `]name` / `:name` helper
        blocks that precede the entry points (e.g. `]no ldy #30 ... ]rts
        rts`), and reuses the same `]name` for the next cluster — so
        'nearest prior to the branch site' picks the binding actually in
        effect. Returns `None` if no definition is found before
        `before_idx` (the branch stays unresolved; the interpreter then
        surfaces a clear error).

        `defined` lists labels the routine already carries. If the block
        falls through into one of them (commonly the shared `]rts:`
        trailer that `]no` flows into when `]rts` was inlined separately),
        stop and emit a local `Goto` to it rather than re-emitting the
        label and its code — otherwise the routine would carry a duplicate
        `Label` that downstream name-keyed passes silently collapse."""
        def_idx = None
        scan = before_idx - 1
        while scan >= 0:
            ln = lines[scan]
            if scan not in inactive and not ln.is_blank and ln.label == target:
                def_idx = scan
                break
            scan -= 1
        if def_idx is None:
            return None

        out: list = []
        emitted_instr = False
        i = def_idx
        while i < len(lines):
            ln = lines[i]
            if ln.is_blank or i in inactive:
                i += 1
                continue
            ref = SourceRef(
                file=str(ln.file),
                line=ln.lineno,
                raw=ln.raw.rstrip("\n"),
            )
            if ln.label and ln.label != target:
                if ln.label in defined:
                    # Fall-through into a block already present in the
                    # routine — preserve the edge with a local jump, don't
                    # duplicate the label/code.
                    out.append(Goto(target=ln.label, kind="local", src=ref))
                    return out
                # Falling through into a new global-named routine ends the
                # shared block (helper blocks end in their own terminator
                # before the next entry, but defend against the rare
                # fall-through).
                if emitted_instr and not _is_local_label(ln.label):
                    break
            if ln.label:
                out.append(Label(name=ln.label, src=ref))
            if ln.mnemonic is None or ln.mnemonic in _NON_CODE_DIRECTIVES:
                i += 1
                continue
            instr = _lift_instr(ln, equates, entry_set)
            if instr is None:
                i += 1
                continue
            out.append(instr)
            emitted_instr = True
            if ln.mnemonic in _TERMINATORS:
                break
            i += 1
        return out

    def walk_from(start_idx: int, entry_labels: list[str]) -> Routine:
        # First label in source order is the canonical name.
        name, *aliases = entry_labels
        routine = Routine(name=name, entry_aliases=list(aliases))

        idx = start_idx
        first = True
        while idx < len(lines):
            line = lines[idx]
            if line.is_blank or idx in inactive:
                idx += 1
                continue

            # Labels on later lines, internal to the routine, get
            # surfaced as `Label` items so branches/local gotos within
            # the routine can resolve. (The pilot doesn't exercise this
            # yet but the lifter handles it correctly.)
            if not first and line.label and line.mnemonic and line.mnemonic not in _NON_CODE_DIRECTIVES:
                routine.body.append(
                    Label(
                        name=line.label,
                        src=SourceRef(
                            file=str(line.file),
                            line=line.lineno,
                            raw=line.raw.rstrip("\n"),
                        ),
                    )
                )

            if line.mnemonic is None or line.mnemonic in _NON_CODE_DIRECTIVES:
                # IIgs toolbox `jsl` hand-assembled as `hex 22,lo,mid,hi`
                # (the 65816 jsl opcode + 24-bit target). Merlin emits four
                # raw bytes via `hex`, so the call is otherwise invisible
                # in the lift; surface it as an `Unsupported(jsl)` atom
                # tagged 65816, alongside the rest of the IIgs ops.
                jsl_target = _parse_hex_jsl(line)
                if jsl_target is not None:
                    instr = Unsupported(
                        mnemonic="jsl",
                        operand=jsl_target,
                        src=SourceRef(
                            file=str(line.file),
                            line=line.lineno,
                            raw=line.raw.rstrip("\n"),
                            comment=line.comment,
                        ),
                    )
                    routine.body.append(instr)
                    all_unsupported.append(instr)
                    first = False
                    idx += 1
                    continue
                # Non-code line in the middle of a routine — typically a
                # bare label that the lifter will pick up via the
                # pending-labels mechanism on the next code line. We
                # don't add it to the body directly; it'll show up
                # attached to the next instruction's pre-labels. A label on
                # a conditional-assembly directive floats the same way.
                if (not first and line.label
                        and (line.mnemonic is None
                             or line.mnemonic in _COND_DIRECTIVES)):
                    routine.body.append(
                        Label(
                            name=line.label,
                            src=SourceRef(
                                file=str(line.file),
                                line=line.lineno,
                                raw=line.raw.rstrip("\n"),
                            ),
                        )
                    )
                idx += 1
                continue

            instr = _lift_instr(line, equates, entry_set)
            if instr is None:
                idx += 1
                continue

            # `*` is Merlin's location counter — the address of *this*
            # instruction. `jmp *` is the classic deliberate hang and
            # `beq *` a branch-to-self; both are tight self-loops, not
            # calls to an external `*` symbol. Label this instruction and
            # retarget the branch/jump at it so the relooper sees the
            # back-edge.
            if isinstance(instr, (Branch, Goto)) and instr.target == "*":
                self_label = f":__self{line.lineno}"
                routine.body.append(Label(name=self_label, src=instr.src))
                instr = replace(instr, target=self_label)
                if isinstance(instr, Goto):
                    instr = replace(instr, kind="local")

            routine.body.append(instr)
            first = False

            if isinstance(instr, Unsupported):
                all_unsupported.append(instr)
                # An unsupported opcode might or might not terminate a
                # routine. We conservatively keep walking until we hit a
                # known terminator; the routine still ends correctly,
                # the body just carries `Unsupported` items the
                # interpreter will refuse to execute.

            if line.mnemonic in _TERMINATORS:
                # An unconditional terminator (rts/jmp/bra) doesn't
                # necessarily end the routine — Merlin routines often
                # branch *forward* past a `jmp`, e.g. CHECKFLOOR's
                # `bne :2` skips over `:ong jmp onground`. The routine
                # really ends at the next *global* label, since that
                # marks where a new entry point starts. Walk forward
                # looking for either a global label (stop) or a local
                # label / further code (keep going).
                lookahead = idx + 1
                while lookahead < len(lines):
                    nxt = lines[lookahead]
                    if nxt.is_blank or lookahead in inactive or (
                        nxt.mnemonic is None and nxt.label is None
                    ):
                        lookahead += 1
                        continue
                    if nxt.label and not _is_local_label(nxt.label):
                        # A new global-named routine starts here.
                        return routine
                    # A local label, or unlabeled code that follows
                    # the terminator — keep walking.
                    break
                else:
                    # Hit EOF without finding any further code.
                    return routine

            idx += 1

        return routine

    def _attach_local_blocks(routine: Routine, start_idx: int) -> None:
        """Inline shared local-label blocks the routine branches to but
        doesn't define in its own body.

        Merlin clusters routines after shared `]name` / `:name` helper
        blocks that precede the entry points — e.g. `]no ldy #30 ... ]rts
        rts` ahead of the `check*` cluster, branched to via `beq ]no`.
        Those blocks aren't in the routine's sliced (`entry`…terminator)
        body, so the branch target has no local `Label` and would degrade
        to an external tail-call (a no-op stub in the crate). Resolve each
        by nearest-prior definition and append its block.

        Iterates to a fixpoint: an inlined block may branch to further
        local labels (`]above`→`]rts`). The generalized `]rts rts` case
        subsumes the old return-trampoline handling."""
        for _ in range(64):  # fixpoint guard; chains are short in practice
            defined = {it.name for it in routine.body if isinstance(it, Label)}
            missing: list[str] = []
            for item in routine.body:
                tgt: str | None = None
                if isinstance(item, Branch):
                    tgt = item.target
                elif isinstance(item, Goto) and item.kind == "local":
                    tgt = item.target
                if (tgt and _is_local_label(tgt)
                        and tgt not in defined and tgt not in missing):
                    missing.append(tgt)
            if not missing:
                return
            progress = False
            for target in missing:
                if target in defined:
                    # Defined by an earlier inline in this same pass (a
                    # block that fell through into it) — don't inline again.
                    continue
                block = _extract_local_block(target, start_idx, defined)
                if block:
                    routine.body.extend(block)
                    defined.update(
                        it.name for it in block if isinstance(it, Label)
                    )
                    progress = True
            if not progress:
                # Remaining targets have no definition before the entry —
                # leave them unresolved (interpreter surfaces a clear error).
                return

    while requested:
        name = requested.pop(0)
        if name in lifted_names:
            continue
        if name not in label_to_instr_index:
            # An entry the caller asked for but the file doesn't define.
            # Skip silently — the CLI / tests can detect this by
            # comparing requested vs. lifted names.
            continue
        idx = label_to_instr_index[name]

        # Already-lifted instruction range? Collapse aliases instead of
        # creating a duplicate routine.
        already = next(
            (r for r in module.routines
             if r.body and not isinstance(r.body[0], Label)
             and r.body[0].src.line == lines[idx].lineno
             and r.body[0].src.file == str(lines[idx].file)),
            None,
        )
        if already is not None:
            if name not in already.all_entry_names():
                already.entry_aliases.append(name)
            lifted_names.add(name)
            continue

        # Collect every label that binds to this same start instruction,
        # in source order.
        entry_labels = [
            lab for lab, j in label_to_instr_index.items() if j == idx
        ]
        # Stable order: by source line of the line that introduced the
        # label. We don't have that recorded directly but `lines[idx]`
        # plus the preceding label-only lines suffice. Walk back from
        # `idx` collecting bare-label lines.
        ordered: list[str] = []
        scan = idx - 1
        while scan >= 0:
            ln = lines[scan]
            if ln.is_blank or scan in inactive:
                scan -= 1
                continue
            if ln.label and ln.mnemonic is None:
                ordered.append(ln.label)
                scan -= 1
                continue
            break
        ordered.reverse()
        if lines[idx].label:
            ordered.append(lines[idx].label)
        # Anything still in `entry_labels` but not in `ordered` would be
        # a label from a different source location (shouldn't happen
        # given how we built the index, but defend against it).
        for extra in entry_labels:
            if extra not in ordered:
                ordered.append(extra)

        routine = walk_from(idx, ordered)
        _attach_local_blocks(routine, idx)
        module.routines.append(routine)
        for n in routine.all_entry_names():
            lifted_names.add(n)

        # Chase tail-call and JSR targets so the IR1 interpreter can
        # resolve them. We only chase labels we know live in this file
        # — cross-module callees are looked up at run time via the
        # module / alias maps the caller passes to the interpreter.
        for item in routine.body:
            target: str | None = None
            if isinstance(item, Goto) and item.kind == "tail_call":
                target = item.target
            elif isinstance(item, Call):
                target = item.target
            if target and target in label_to_instr_index and target not in lifted_names:
                requested.append(target)

    return LiftReport(module=module, unsupported=all_unsupported)
