"""Pass 1: mechanical lift from a parsed Merlin file to IR1.

Scope of the current implementation: the AUTO.S combat-button pilot —
the smallest set of routines that exercises multi-entry labels (e.g.
`DoBlock` / `DoUp` on consecutive lines), `#-1` / `#0` immediate stores,
unconditional cross-routine `jmp` (tail calls), and fall-through into
the shared `]rts` trampoline.

What works:

* Routine discovery: a routine begins at any instruction line whose
  pending label group contains a requested entry name, *or* whose group
  follows a terminator (rts/jmp) and is reached by fall-through from a
  previous routine via an unconditional `jmp`.
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

from dataclasses import dataclass
from pathlib import Path

from .ir1 import (
    Abs,
    Goto,
    Imm,
    Label,
    LoadImm,
    ModuleIR1,
    Reg,
    Return,
    Routine,
    SourceRef,
    StoreAbs,
    Unsupported,
)
from .pass0_lex import Line
from .pass0_parse import FileAST, eval_expr


# Opcodes that unconditionally end a routine. `bra` is the Merlin 16+
# unconditional short branch — listed for completeness even though the
# upstream source mostly uses `jmp`.
_TERMINATORS = frozenset({"rts", "rti", "jmp", "bra"})

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
    }
)


@dataclass
class LiftReport:
    """Summary of what pass 1 did. Mainly for tests and CLI output."""

    module: ModuleIR1
    unsupported: list[Unsupported]  # all unsupported instructions across routines
    skipped_lines: int                # source lines pass 1 ignored entirely


# ---------------------------------------------------------------- helpers


def _is_local_label(name: str) -> bool:
    """Merlin local label forms: `:foo` (scope = enclosing global) and
    `]foo` (macro-style)."""
    return name.startswith(":") or name.startswith("]")


def _parse_immediate(operand: str, equates: dict[str, int]) -> Imm | None:
    """Parse a `#expr` immediate operand. Returns `None` if `operand`
    isn't a `#`-prefixed immediate (the caller decides what to do)."""
    s = operand.strip()
    if not s.startswith("#"):
        return None
    expr = s[1:].lstrip("<>")  # Merlin `<expr` / `>expr` = low/high byte;
                               # for the pilot the operands are simple
                               # constants so stripping is safe. A later
                               # slice will keep the operator and apply it.
    try:
        value = eval_expr(expr, equates)
    except ValueError:
        return None
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

    src = SourceRef(file=str(line.file), line=line.lineno, raw=line.raw.rstrip("\n"))

    if mnemonic == "rts":
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
        # Absolute/indexed/indirect loads are part of the next pilot
        # slice (`rndp`, `CheckFloor`). Mark unsupported for now.
        return Unsupported(mnemonic=mnemonic, operand=line.operand, src=src)

    if mnemonic in ("sta", "stx", "sty"):
        if line.operand is None:
            return Unsupported(mnemonic=mnemonic, operand=None, src=src)
        addr = _parse_absolute(line.operand, equates)
        if addr is not None:
            return StoreAbs(reg=_reg_of_store(mnemonic), target=addr, src=src)
        return Unsupported(mnemonic=mnemonic, operand=line.operand, src=src)

    # All other opcodes are out of scope for this slice. Marking
    # `Unsupported` (instead of skipping) keeps the IR aligned with the
    # source and gives reviewers an exact list of what still needs work.
    del entry_names  # unused, reserved for future heuristics
    return Unsupported(mnemonic=mnemonic, operand=line.operand, src=src)


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
    skipped_lines = 0
    all_unsupported: list[Unsupported] = []

    # Pre-index the lines so we can walk forward cheaply from any label.
    lines = file_ast.lines

    # Map label -> index in `lines` of the line whose *next* code
    # instruction the label refers to. Bare-label lines just attach
    # their label to the upcoming instruction.
    label_to_instr_index: dict[str, int] = {}
    pending_labels: list[str] = []
    for i, line in enumerate(lines):
        if line.is_blank:
            continue
        if line.label and (line.mnemonic is None or line.mnemonic in _NON_CODE_DIRECTIVES):
            # Bare-label line, or a label on a non-code directive. The
            # bare-label case is the one we care about for the pilot
            # (`DoBlock\n DoUp lda #-1`); the directive case can also
            # carry a label (e.g. equates) and we just ignore that
            # because pass 0 already absorbed it.
            if line.mnemonic is None:
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

    def walk_from(start_idx: int, entry_labels: list[str]) -> Routine:
        # First label in source order is the canonical name.
        name, *aliases = entry_labels
        routine = Routine(name=name, entry_aliases=list(aliases))

        idx = start_idx
        first = True
        while idx < len(lines):
            line = lines[idx]
            if line.is_blank:
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
                # Non-code line in the middle of a routine — typically a
                # bare label that the lifter will pick up via the
                # pending-labels mechanism on the next code line. We
                # don't add it to the body directly; it'll show up
                # attached to the next instruction's pre-labels.
                if not first and line.label and line.mnemonic is None:
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
                return routine

            idx += 1

        return routine

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
             if r.body and isinstance(r.body[0], (LoadImm, StoreAbs, Goto, Return, Unsupported))
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
            if ln.is_blank:
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
        module.routines.append(routine)
        for n in routine.all_entry_names():
            lifted_names.add(n)

        # Chase tail-call targets so the IR1 interpreter can resolve
        # them. We only chase labels we know live in this file.
        for item in routine.body:
            if isinstance(item, Goto) and item.kind == "tail_call":
                if item.target in label_to_instr_index and item.target not in lifted_names:
                    requested.append(item.target)

    return LiftReport(module=module, unsupported=all_unsupported, skipped_lines=skipped_lines)
