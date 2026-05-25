"""Command-line entry point for `pop-lifter`.

Subcommands:

* `parse`     — pass 0: lex/parse Merlin sources, print equates and dum
                blocks. The default input set is `EQ.S` + `GAMEEQ.S`.
* `dump-ast`  — pass 0: write the equates / dum-block AST to disk as
                JSON, so the checked-in artifacts in `ir/pass0/` stay
                inspectable and reviewable.
* `lift`      — pass 1: lift a single `.S` file's specified entry-point
                routines to IR1 and print the result.

`dump-ir1` is folded into `lift --out` rather than a separate verb.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import ir1 as ir1_mod
from . import ir3 as ir3_mod
from .pass0_parse import ProgramAST, parse_files
from .pass1_lift import discover_entries, lift_file
from .pass2_reloop import is_unstructured, reloop_module
from .pass2_struct import elision_stats, fusion_stats, structure_module
from .pass3_expr import fold_module, fold_stats, wide16_stats
from .pass3_loops import dowhile_stats, for_stats, recover_loops, repeat_stats
from .pass3_match import match_stats, recognize_module
from .pass3_smc import recognize_smc, smc_store_count, smc_var_count
from .pass3_temps import recover_temps, temp_stats
from .pass4_emit_rust import emit_crate, emit_module, emit_modules, lower_stats

DEFAULT_SOURCE_REL = Path("01 POP Source/Source")


# ---------------------------------------------------------------- shared

def _resolve_source_dir(args: argparse.Namespace) -> Path | None:
    root = Path(args.source_root)
    src_dir = root / DEFAULT_SOURCE_REL
    if not src_dir.is_dir():
        print(f"error: source dir not found: {src_dir}", file=sys.stderr)
        return None
    return src_dir


def _print_parse_text(ast: ProgramAST) -> None:
    print(f"# equates ({len(ast.equates)})")
    for k, v in sorted(ast.equates.items()):
        print(f"{k:>20} = ${v:04x}")
    print(f"\n# dum blocks ({len(ast.dum_blocks)})")
    for b in ast.dum_blocks:
        named = sum(1 for f in b.fields if f.name)
        print(
            f"  {b.start_expr:>16} @ ${b.start_addr:04x} "
            f"({len(b.fields)} fields, {named} named) "
            f"[{Path(b.file).name}:{b.line}]"
        )
    if ast.diagnostics:
        print(f"\n# diagnostics ({len(ast.diagnostics)})")
        for d in ast.diagnostics:
            print(f"  {d}")


# ---------------------------------------------------------------- parse

def _cmd_parse(args: argparse.Namespace) -> int:
    src_dir = _resolve_source_dir(args)
    if src_dir is None:
        return 2

    inputs: list[Path] = []
    if args.files:
        missing: list[str] = []
        for f in args.files:
            p = Path(f)
            if not p.is_file():
                missing.append(f)
            else:
                inputs.append(p)
        if missing:
            for f in missing:
                print(f"error: input file not found: {f}", file=sys.stderr)
            return 2
    else:
        for name in ("EQ.S", "GAMEEQ.S"):
            p = src_dir / name
            if p.exists():
                inputs.append(p)

    if not inputs:
        print("error: no input files", file=sys.stderr)
        return 2

    ast = parse_files(inputs, search_paths=[src_dir])
    if args.format == "json":
        print(ast.to_json())
    else:
        _print_parse_text(ast)
    return 0


# ---------------------------------------------------------------- dump-ast

def _cmd_dump_ast(args: argparse.Namespace) -> int:
    """Persist the pass-0 AST as JSON under `ir/pass0/`.

    Defaults to dumping `EQ.S + GAMEEQ.S` because they're the
    authoritative address book the rest of the lifter feeds on. The
    intent is for the JSON to be committed: reviewers can diff symbol
    movements directly, and CI's regen test will catch any drift between
    the parser and the snapshot.
    """
    src_dir = _resolve_source_dir(args)
    if src_dir is None:
        return 2

    inputs = [src_dir / name for name in ("EQ.S", "GAMEEQ.S") if (src_dir / name).exists()]
    if not inputs:
        print("error: EQ.S / GAMEEQ.S not found in source tree", file=sys.stderr)
        return 2

    ast = parse_files(inputs, search_paths=[src_dir])
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(ast.to_json() + "\n", encoding="utf-8")
    print(f"wrote {out_path} ({len(ast.equates)} equates, "
          f"{len(ast.dum_blocks)} dum blocks)")
    return 0


# ---------------------------------------------------------------- lift

def _cmd_lift(args: argparse.Namespace) -> int:
    src_dir = _resolve_source_dir(args)
    if src_dir is None:
        return 2

    # Resolve each input file: absolute paths used as-is; bare names are
    # tried under the source dir first, then cwd.
    file_paths: list[Path] = []
    for raw in args.file:
        p = Path(raw)
        if not p.is_absolute():
            candidate = src_dir / p
            if candidate.is_file():
                p = candidate
        if not p.is_file():
            print(f"error: input file not found: {raw}", file=sys.stderr)
            return 2
        file_paths.append(p)

    # Parse the equate base (EQ.S + GAMEEQ.S) plus every target file so
    # every operand symbol the lifter encounters resolves to a concrete
    # address. Each target file's own equates flow through the same
    # parse_files call.
    base = [src_dir / n for n in ("EQ.S", "GAMEEQ.S") if (src_dir / n).exists()]
    ast = parse_files([*base, *file_paths], search_paths=[src_dir])
    symbols = ast.symbols()

    # For each file, lift only those `--entry` labels that this file
    # actually defines. This lets a single CLI invocation produce a
    # cross-module dump (e.g. AUTO.S's `rndp` + GRAFIX.S's `RND`)
    # without forcing the caller to know which entry lives where.
    dumps: list[tuple[Path, int, int, str]] = []
    handled: set[str] = set()
    for file_path in file_paths:
        file_ast = next(
            (f for f in ast.files if Path(f.path).resolve() == file_path.resolve()),
            None,
        )
        if file_ast is None:
            print(
                f"error: file {file_path} was not loaded by the parser",
                file=sys.stderr,
            )
            return 1
        defined = set(discover_entries(file_ast))
        local_entries = [e for e in args.entry if e in defined and e not in handled]
        if not local_entries:
            continue
        report = lift_file(file_ast, symbols, local_entries)
        if not report.module.routines:
            continue
        dumps.append((
            file_path,
            len(report.module.routines),
            len(report.unsupported),
            ir1_mod.format_module(report.module),
        ))
        handled.update(local_entries)

    missing = [e for e in args.entry if e not in handled]
    if missing:
        print(
            f"error: entries not found in any input file: {missing}",
            file=sys.stderr,
        )
        return 1

    text = "\n".join(d[3] for d in dumps)
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(text, encoding="utf-8")
        total_routines = sum(d[1] for d in dumps)
        total_unsupp = sum(d[2] for d in dumps)
        print(
            f"wrote {out_path} ({total_routines} routines across "
            f"{len(dumps)} module(s), {total_unsupp} unsupported ops)"
        )
    else:
        sys.stdout.write(text)
    return 0


# ---------------------------------------------------------------- struct (pass 2)

def _cmd_struct(args: argparse.Namespace) -> int:
    """Run pass 1 on the given files / entries and then pass 2's
    structurer. Writes the IR2 dump to `--out` (or stdout)."""
    src_dir = _resolve_source_dir(args)
    if src_dir is None:
        return 2

    file_paths: list[Path] = []
    for raw in args.file:
        p = Path(raw)
        if not p.is_absolute():
            candidate = src_dir / p
            if candidate.is_file():
                p = candidate
        if not p.is_file():
            print(f"error: input file not found: {raw}", file=sys.stderr)
            return 2
        file_paths.append(p)

    base = [src_dir / n for n in ("EQ.S", "GAMEEQ.S") if (src_dir / n).exists()]
    ast = parse_files([*base, *file_paths], search_paths=[src_dir])
    symbols = ast.symbols()

    dumps: list[str] = []
    total_fused = 0
    total_unfused = 0
    total_cmp_left = 0
    total_setc_left = 0
    total_routines = 0
    handled: set[str] = set()
    for file_path in file_paths:
        file_ast = next(
            (f for f in ast.files if Path(f.path).resolve() == file_path.resolve()),
            None,
        )
        if file_ast is None:
            print(
                f"error: file {file_path} was not loaded by the parser",
                file=sys.stderr,
            )
            return 1
        defined = set(discover_entries(file_ast))
        local_entries = [e for e in args.entry if e in defined and e not in handled]
        if not local_entries:
            continue
        ir1_module = lift_file(file_ast, symbols, local_entries).module
        ir2_module = structure_module(ir1_module)
        f, u = fusion_stats(ir2_module)
        cmp_left, setc_left = elision_stats(ir2_module)
        total_fused += f
        total_unfused += u
        total_cmp_left += cmp_left
        total_setc_left += setc_left
        total_routines += len(ir2_module.routines)
        dumps.append(ir1_mod.format_module(ir2_module))
        handled.update(local_entries)

    missing = [e for e in args.entry if e not in handled]
    if missing:
        print(
            f"error: entries not found in any input file: {missing}",
            file=sys.stderr,
        )
        return 1

    text = "\n".join(dumps)
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(text, encoding="utf-8")
        print(
            f"wrote {out_path} ({total_routines} routines, "
            f"{total_fused} fused-if, {total_unfused} unfused-branch, "
            f"{total_cmp_left} cmp left, "
            f"{total_setc_left} clc/sec left)"
        )
    else:
        sys.stdout.write(text)
    return 0


# ---------------------------------------------------------------- reloop (pass 2 phase 2)


def _cmd_reloop(args: argparse.Namespace) -> int:
    """Run passes 1 + 2 (fusion/elision) + reloop (CFG → structured
    IR3) for the given files / entries. Writes the IR3 dump to `--out`
    (or stdout). Routines the relooper can't reduce to natural
    loops/conditionals (irreducible flow, multi-back-edge loops) fall
    back to a `loop { match pc { ... } }` dispatcher — still valid
    structured IR3, flagged by a higher `unstructured` count in the
    summary line."""
    src_dir = _resolve_source_dir(args)
    if src_dir is None:
        return 2

    file_paths: list[Path] = []
    for raw in args.file:
        p = Path(raw)
        if not p.is_absolute():
            candidate = src_dir / p
            if candidate.is_file():
                p = candidate
        if not p.is_file():
            print(f"error: input file not found: {raw}", file=sys.stderr)
            return 2
        file_paths.append(p)

    base = [src_dir / n for n in ("EQ.S", "GAMEEQ.S") if (src_dir / n).exists()]
    ast = parse_files([*base, *file_paths], search_paths=[src_dir])
    symbols = ast.symbols()

    dumps: list[str] = []
    total_routines = 0
    total_unstructured = 0
    handled: set[str] = set()
    for file_path in file_paths:
        file_ast = next(
            (f for f in ast.files if Path(f.path).resolve() == file_path.resolve()),
            None,
        )
        if file_ast is None:
            print(
                f"error: file {file_path} was not loaded by the parser",
                file=sys.stderr,
            )
            return 1
        defined = set(discover_entries(file_ast))
        local_entries = [e for e in args.entry if e in defined and e not in handled]
        if not local_entries:
            continue
        ir1_module = lift_file(file_ast, symbols, local_entries).module
        ir2_module = structure_module(ir1_module)
        ir3_module = reloop_module(ir2_module)
        total_routines += len(ir3_module.routines)
        total_unstructured += sum(1 for r in ir3_module.routines if is_unstructured(r))
        dumps.append(ir3_mod.format_module(ir3_module))
        handled.update(local_entries)

    missing = [e for e in args.entry if e not in handled]
    if missing:
        print(
            f"error: entries not found in any input file: {missing}",
            file=sys.stderr,
        )
        return 1

    text = "\n".join(dumps)
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(text, encoding="utf-8")
        print(
            f"wrote {out_path} ({total_routines} routines, "
            f"{total_unstructured} unstructured)"
        )
    else:
        sys.stdout.write(text)
    return 0


# ---------------------------------------------------------------- fold (pass 3 slice 1)


def _cmd_fold(args: argparse.Namespace) -> int:
    """Run passes 1 + 2 (fusion/elision) + reloop + pass-3 accumulator
    copy folding for the given files / entries. Writes the folded IR3
    dump to `--out` (or stdout). The summary line reports how many
    `lda`/`sta` round-trips collapsed into direct `Assign`s."""
    src_dir = _resolve_source_dir(args)
    if src_dir is None:
        return 2

    file_paths: list[Path] = []
    for raw in args.file:
        p = Path(raw)
        if not p.is_absolute():
            candidate = src_dir / p
            if candidate.is_file():
                p = candidate
        if not p.is_file():
            print(f"error: input file not found: {raw}", file=sys.stderr)
            return 2
        file_paths.append(p)

    base = [src_dir / n for n in ("EQ.S", "GAMEEQ.S") if (src_dir / n).exists()]
    ast = parse_files([*base, *file_paths], search_paths=[src_dir])
    symbols = ast.symbols()

    dumps: list[str] = []
    total_routines = 0
    total_assigns = 0
    total_wide16 = 0
    handled: set[str] = set()
    for file_path in file_paths:
        file_ast = next(
            (f for f in ast.files if Path(f.path).resolve() == file_path.resolve()),
            None,
        )
        if file_ast is None:
            print(
                f"error: file {file_path} was not loaded by the parser",
                file=sys.stderr,
            )
            return 1
        defined = set(discover_entries(file_ast))
        local_entries = [e for e in args.entry if e in defined and e not in handled]
        if not local_entries:
            continue
        ir1_module = lift_file(file_ast, symbols, local_entries).module
        ir2_module = structure_module(ir1_module)
        ir3_module = reloop_module(ir2_module)
        folded = fold_module(ir3_module)
        total_routines += len(folded.routines)
        total_assigns += fold_stats(folded)
        total_wide16 += wide16_stats(folded)
        dumps.append(ir3_mod.format_module(folded))
        handled.update(local_entries)

    missing = [e for e in args.entry if e not in handled]
    if missing:
        print(
            f"error: entries not found in any input file: {missing}",
            file=sys.stderr,
        )
        return 1

    text = "\n".join(dumps)
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(text, encoding="utf-8")
        print(
            f"wrote {out_path} ({total_routines} routines, "
            f"{total_assigns} folded assigns, {total_wide16} wide16)"
        )
    else:
        sys.stdout.write(text)
    return 0


# ---------------------------------------------------------------- match (pass 3)


def _cmd_match(args: argparse.Namespace) -> int:
    """Run passes 1 + 2 + reloop + fold, then recognise the jump-table
    dispatch idiom as structured `match` statements. Writes the IR3 dump
    to `--out` (or stdout); the summary reports how many `match`es were
    recognised."""
    src_dir = _resolve_source_dir(args)
    if src_dir is None:
        return 2

    file_paths: list[Path] = []
    for raw in args.file:
        p = Path(raw)
        if not p.is_absolute():
            candidate = src_dir / p
            if candidate.is_file():
                p = candidate
        if not p.is_file():
            print(f"error: input file not found: {raw}", file=sys.stderr)
            return 2
        file_paths.append(p)

    base = [src_dir / n for n in ("EQ.S", "GAMEEQ.S") if (src_dir / n).exists()]
    ast = parse_files([*base, *file_paths], search_paths=[src_dir])
    symbols = ast.symbols()

    dumps: list[str] = []
    total_routines = 0
    total_matches = 0
    handled: set[str] = set()
    for file_path in file_paths:
        file_ast = next(
            (f for f in ast.files if Path(f.path).resolve() == file_path.resolve()),
            None,
        )
        if file_ast is None:
            print(
                f"error: file {file_path} was not loaded by the parser",
                file=sys.stderr,
            )
            return 1
        defined = set(discover_entries(file_ast))
        local_entries = [e for e in args.entry if e in defined and e not in handled]
        if not local_entries:
            continue
        ir1_module = lift_file(file_ast, symbols, local_entries).module
        ir3_module = reloop_module(structure_module(ir1_module))
        matched = recognize_module(fold_module(ir3_module))
        total_routines += len(matched.routines)
        total_matches += match_stats(matched)
        dumps.append(ir3_mod.format_module(matched))
        handled.update(local_entries)

    missing = [e for e in args.entry if e not in handled]
    if missing:
        print(
            f"error: entries not found in any input file: {missing}",
            file=sys.stderr,
        )
        return 1

    text = "\n".join(dumps)
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(text, encoding="utf-8")
        print(
            f"wrote {out_path} ({total_routines} routines, "
            f"{total_matches} matches recognised)"
        )
    else:
        sys.stdout.write(text)
    return 0


# ---------------------------------------------------------------- loops (pass 3)


def _cmd_loops(args: argparse.Namespace) -> int:
    """Run pass 1 + 2 + reloop + fold, then recover loops: a bottom-test
    `loop { body ; if exit { break } }` becomes `do { body } while
    !exit`, and a counted loop is promoted to a `for` (down-counter
    `y = #N ; do { … ; y -= 1 } while y >= 0` ⇒ `for y in (0..=N).rev()`;
    up-counter `x = #i ; do { … ; x += 1 } while x != N` ⇒ `for x in
    i..N`); a full-wrap busy-wait becomes `repeat 0x100 { … }`. Writes
    the IR3 dump to `--out` (or stdout); the summary reports how many
    `for`, `repeat`, and `do-while` loops were recovered."""
    src_dir = _resolve_source_dir(args)
    if src_dir is None:
        return 2

    file_paths: list[Path] = []
    for raw in args.file:
        p = Path(raw)
        if not p.is_absolute():
            candidate = src_dir / p
            if candidate.is_file():
                p = candidate
        if not p.is_file():
            print(f"error: input file not found: {raw}", file=sys.stderr)
            return 2
        file_paths.append(p)

    base = [src_dir / n for n in ("EQ.S", "GAMEEQ.S") if (src_dir / n).exists()]
    ast = parse_files([*base, *file_paths], search_paths=[src_dir])
    symbols = ast.symbols()

    dumps: list[str] = []
    total_routines = 0
    total_dowhile = 0
    total_for = 0
    total_repeat = 0
    handled: set[str] = set()
    for file_path in file_paths:
        file_ast = next(
            (f for f in ast.files if Path(f.path).resolve() == file_path.resolve()),
            None,
        )
        if file_ast is None:
            print(
                f"error: file {file_path} was not loaded by the parser",
                file=sys.stderr,
            )
            return 1
        defined = set(discover_entries(file_ast))
        local_entries = [e for e in args.entry if e in defined and e not in handled]
        if not local_entries:
            continue
        ir3_module = reloop_module(structure_module(lift_file(file_ast, symbols, local_entries).module))
        recovered = recover_loops(fold_module(ir3_module))
        total_routines += len(recovered.routines)
        total_dowhile += dowhile_stats(recovered)
        total_for += for_stats(recovered)
        total_repeat += repeat_stats(recovered)
        dumps.append(ir3_mod.format_module(recovered))
        handled.update(local_entries)

    missing = [e for e in args.entry if e not in handled]
    if missing:
        print(
            f"error: entries not found in any input file: {missing}",
            file=sys.stderr,
        )
        return 1

    text = "\n".join(dumps)
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(text, encoding="utf-8")
        print(
            f"wrote {out_path} ({total_routines} routines, "
            f"{total_for} for + {total_repeat} repeat + {total_dowhile} "
            f"do-while loops recovered)"
        )
    else:
        sys.stdout.write(text)
    return 0


# ---------------------------------------------------------------- temps (pass 3)


def _cmd_temps(args: argparse.Namespace) -> int:
    """Run the full pass-3 chain (reloop + fold + loop recovery) then
    recover pha/pla scoped temporaries: a `push a ; … ; a = pop` bracket
    becomes `tmp{n} = a ; … ; a = tmp{n}`. Writes the IR3 dump to `--out`
    (or stdout); the summary reports how many temporaries were
    recovered."""
    src_dir = _resolve_source_dir(args)
    if src_dir is None:
        return 2

    file_paths: list[Path] = []
    for raw in args.file:
        p = Path(raw)
        if not p.is_absolute():
            candidate = src_dir / p
            if candidate.is_file():
                p = candidate
        if not p.is_file():
            print(f"error: input file not found: {raw}", file=sys.stderr)
            return 2
        file_paths.append(p)

    base = [src_dir / n for n in ("EQ.S", "GAMEEQ.S") if (src_dir / n).exists()]
    ast = parse_files([*base, *file_paths], search_paths=[src_dir])
    symbols = ast.symbols()

    dumps: list[str] = []
    total_routines = 0
    total_temps = 0
    handled: set[str] = set()
    for file_path in file_paths:
        file_ast = next(
            (f for f in ast.files if Path(f.path).resolve() == file_path.resolve()),
            None,
        )
        if file_ast is None:
            print(
                f"error: file {file_path} was not loaded by the parser",
                file=sys.stderr,
            )
            return 1
        defined = set(discover_entries(file_ast))
        local_entries = [e for e in args.entry if e in defined and e not in handled]
        if not local_entries:
            continue
        ir3_module = reloop_module(structure_module(lift_file(file_ast, symbols, local_entries).module))
        recovered = recover_temps(recover_loops(fold_module(ir3_module)))
        total_routines += len(recovered.routines)
        total_temps += temp_stats(recovered)
        dumps.append(ir3_mod.format_module(recovered))
        handled.update(local_entries)

    missing = [e for e in args.entry if e not in handled]
    if missing:
        print(
            f"error: entries not found in any input file: {missing}",
            file=sys.stderr,
        )
        return 1

    text = "\n".join(dumps)
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(text, encoding="utf-8")
        print(
            f"wrote {out_path} ({total_routines} routines, "
            f"{total_temps} scoped temporaries recovered)"
        )
    else:
        sys.stdout.write(text)
    return 0


# ---------------------------------------------------------------- emit (pass 4)


def _cmd_emit(args: argparse.Namespace) -> int:
    """Run the full pass-3 chain (reloop + fold + match recognition +
    loop + temp recovery) then emit Rust source from the structured
    IR3. Recognised jump-table dispatches emit as `match`; remaining
    statements not yet lowered appear as `// TODO(pass4)` / `// raw:`
    comments. Writes the `.rs` text to `--out` (or stdout); the summary
    reports lowered vs. deferred top-level statements."""
    src_dir = _resolve_source_dir(args)
    if src_dir is None:
        return 2

    file_paths: list[Path] = []
    for raw in args.file:
        p = Path(raw)
        if not p.is_absolute():
            candidate = src_dir / p
            if candidate.is_file():
                p = candidate
        if not p.is_file():
            print(f"error: input file not found: {raw}", file=sys.stderr)
            return 2
        file_paths.append(p)

    base = [src_dir / n for n in ("EQ.S", "GAMEEQ.S") if (src_dir / n).exists()]
    ast = parse_files([*base, *file_paths], search_paths=[src_dir])
    symbols = ast.symbols()

    modules = []
    total_routines = 0
    total_lowered = 0
    total_deferred = 0
    handled: set[str] = set()
    for file_path in file_paths:
        file_ast = next(
            (f for f in ast.files if Path(f.path).resolve() == file_path.resolve()),
            None,
        )
        if file_ast is None:
            print(
                f"error: file {file_path} was not loaded by the parser",
                file=sys.stderr,
            )
            return 1
        defined = set(discover_entries(file_ast))
        local_entries = [e for e in args.entry if e in defined and e not in handled]
        if not local_entries:
            continue
        ir1_module = recognize_smc(lift_file(file_ast, symbols, local_entries).module)
        ir3_module = reloop_module(structure_module(ir1_module))
        recovered = recover_temps(recover_loops(recognize_module(fold_module(ir3_module))))
        total_routines += len(recovered.routines)
        lowered, deferred = lower_stats(recovered)
        total_lowered += lowered
        total_deferred += deferred
        modules.append(recovered)
        handled.update(local_entries)

    missing = [e for e in args.entry if e not in handled]
    if missing:
        print(
            f"error: entries not found in any input file: {missing}",
            file=sys.stderr,
        )
        return 1

    # Emit every module into one file sharing a single `mod sym` block,
    # so concatenated output stays a valid single Rust module.
    text = emit_modules(modules) if modules else ""
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(text, encoding="utf-8")
        print(
            f"wrote {out_path} ({total_routines} routines, "
            f"{total_lowered} stmts lowered, {total_deferred} deferred)"
        )
    else:
        sys.stdout.write(text)
    return 0


# ---------------------------------------------------------------- smc (pass 3)


def _cmd_smc(args: argparse.Namespace) -> int:
    """Run pass 1, then recognise self-modifying-code immediate patches
    as operand variables. Writes the IR1 dump to `--out` (or stdout);
    the summary reports how many patch stores became `StoreOpVar`s."""
    src_dir = _resolve_source_dir(args)
    if src_dir is None:
        return 2

    file_paths: list[Path] = []
    for raw in args.file:
        p = Path(raw)
        if not p.is_absolute():
            candidate = src_dir / p
            if candidate.is_file():
                p = candidate
        if not p.is_file():
            print(f"error: input file not found: {raw}", file=sys.stderr)
            return 2
        file_paths.append(p)

    base = [src_dir / n for n in ("EQ.S", "GAMEEQ.S") if (src_dir / n).exists()]
    ast = parse_files([*base, *file_paths], search_paths=[src_dir])
    symbols = ast.symbols()

    dumps: list[str] = []
    total_routines = 0
    total_vars = 0
    total_stores = 0
    handled: set[str] = set()
    for file_path in file_paths:
        file_ast = next(
            (f for f in ast.files if Path(f.path).resolve() == file_path.resolve()),
            None,
        )
        if file_ast is None:
            print(
                f"error: file {file_path} was not loaded by the parser",
                file=sys.stderr,
            )
            return 1
        defined = set(discover_entries(file_ast))
        local_entries = [e for e in args.entry if e in defined and e not in handled]
        if not local_entries:
            continue
        module = recognize_smc(lift_file(file_ast, symbols, local_entries).module)
        total_routines += len(module.routines)
        total_vars += smc_var_count(module)
        total_stores += smc_store_count(module)
        dumps.append(ir1_mod.format_module(module))
        handled.update(local_entries)

    missing = [e for e in args.entry if e not in handled]
    if missing:
        print(
            f"error: entries not found in any input file: {missing}",
            file=sys.stderr,
        )
        return 1

    text = "\n".join(dumps)
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(text, encoding="utf-8")
        print(
            f"wrote {out_path} ({total_routines} routines, "
            f"{total_vars} SMC operand variables across "
            f"{total_stores} patch stores)"
        )
    else:
        sys.stdout.write(text)
    return 0


# ---------------------------------------------------------------- lift-all

def _cmd_lift_all(args: argparse.Namespace) -> int:
    """Mechanical sweep: parse every `.S` in the source tree, auto-
    discover entry points, lift each file's routines to IR1, and write
    one dump per file under `--out-dir` (default `ir/raw/`).

    Two artifacts per lifted module:

    * `<NAME>.ir1` — raw pass-1 IR1. `cmp + branch` pairs show as
      separate items (`cmp a, #foo` then `if eq goto :label`) — the
      mechanical "what does pass 1 see" view, useful when you want
      to spot opcodes the lifter hasn't yet handled.
    * `<NAME>.ir2` — pass-2 fused + elided form. The same pairs
      collapse into self-contained `if a == #foo goto :label`, dead
      cmps drop out, and dumps read like real C-ish code. This is
      the form most downstream consumers (pass 3, pass 4, reviewers)
      actually want.

    Also emits `SUMMARY.md` with per-file routine / Unsupported /
    fused-If counts. Files that contain zero global code labels
    (e.g. equate-only `EQ.S` / `GAMEEQ.S`) are skipped entirely —
    they have nothing to lift.
    """
    src_dir = _resolve_source_dir(args)
    if src_dir is None:
        return 2

    files = sorted(src_dir.glob("*.S"))
    if not files:
        print(f"error: no .S files in {src_dir}", file=sys.stderr)
        return 2

    # Parse the whole tree once so cross-file equate references resolve.
    # Parsing EQ.S and GAMEEQ.S first guarantees the address book is in
    # place before any code file is reached.
    base_order = [src_dir / "EQ.S", src_dir / "GAMEEQ.S"]
    base = [p for p in base_order if p.exists()]
    others = [p for p in files if p not in base]
    ast = parse_files([*base, *others], search_paths=[src_dir])
    symbols = ast.symbols()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Each row records both the raw IR1 stats and the pass-2 IR2
    # follow-up so SUMMARY.md can show what fusion bought us per
    # module.
    rows: list[tuple[str, int, int, int, int, int]] = []
    # (file, routines, instrs, unsupported, fused_if, unfused_branch)
    skipped: list[str] = []

    for src_path in files:
        file_ast = next(
            (f for f in ast.files if Path(f.path).resolve() == src_path.resolve()),
            None,
        )
        if file_ast is None:
            print(
                f"warning: {src_path.name} was not loaded by the parser; "
                f"skipping",
                file=sys.stderr,
            )
            continue

        entries = discover_entries(file_ast)
        if not entries:
            skipped.append(src_path.name)
            continue

        report = lift_file(file_ast, symbols, entries)
        if not report.module.routines:
            skipped.append(src_path.name)
            continue

        # Pass 1 → .ir1 (raw mechanical view).
        ir1_text = ir1_mod.format_module(report.module)
        ir1_path = out_dir / f"{src_path.stem.upper()}.ir1"
        ir1_path.write_text(ir1_text, encoding="utf-8")

        # Pass 2 → .ir2 (fused + elided view). Same Routine type
        # shape, so the same `format_module` does the dump. `If`
        # nodes carry the structured Compare so `cmp + branch` pairs
        # become self-contained `if a == #foo goto :L` lines instead
        # of the cryptic raw `if eq goto :L` form.
        ir2_module = structure_module(report.module)
        ir2_text = ir1_mod.format_module(ir2_module)
        ir2_path = out_dir / f"{src_path.stem.upper()}.ir2"
        ir2_path.write_text(ir2_text, encoding="utf-8")

        # Count items for the summary (Label items don't count as
        # "instructions").
        instrs = sum(
            1
            for r in report.module.routines
            for item in r.body
            if not isinstance(item, ir1_mod.Label)
        )
        fused, unfused = fusion_stats(ir2_module)
        rows.append((
            src_path.name,
            len(report.module.routines),
            instrs,
            len(report.unsupported),
            fused,
            unfused,
        ))

    # Write SUMMARY.md. The table doubles as a "how complete is pass 1"
    # snapshot — reviewers can scan the Unsupported column to see which
    # files are still mostly opaque to the lifter.
    summary_lines = [
        "# Pass-1/2 mechanical lift — per-file summary",
        "",
        f"Generated by `pop-lifter lift-all`. {len(rows)} files lifted "
        f"(each produces a `.ir1` raw view + a `.ir2` fused view), "
        f"{len(skipped)} skipped (no global code labels).",
        "",
        "| File | Routines | Instructions | Unsupported | Fused-If | Unfused-Branch |",
        "|------|---------:|-------------:|------------:|---------:|---------------:|",
    ]
    total_routines = 0
    total_instrs = 0
    total_unsupp = 0
    total_fused = 0
    total_unfused = 0
    for name, n_routines, n_instrs, n_unsupp, n_fused, n_unfused in sorted(rows):
        summary_lines.append(
            f"| `{name}` | {n_routines} | {n_instrs} | {n_unsupp} | "
            f"{n_fused} | {n_unfused} |"
        )
        total_routines += n_routines
        total_instrs += n_instrs
        total_unsupp += n_unsupp
        total_fused += n_fused
        total_unfused += n_unfused
    summary_lines.append(
        f"| **total** | **{total_routines}** | **{total_instrs}** | "
        f"**{total_unsupp}** | **{total_fused}** | **{total_unfused}** |"
    )
    if skipped:
        summary_lines += [
            "",
            "## Skipped (no liftable entry points)",
            "",
        ]
        for name in sorted(skipped):
            summary_lines.append(f"- `{name}`")
    summary_lines.append("")
    (out_dir / "SUMMARY.md").write_text("\n".join(summary_lines), encoding="utf-8")

    print(
        f"wrote {len(rows)} .ir1 + {len(rows)} .ir2 files + SUMMARY.md "
        f"under {out_dir} (total {total_routines} routines, "
        f"{total_instrs} instructions, {total_unsupp} unsupported, "
        f"{total_fused} fused-If, {total_unfused} unfused-Branch)"
    )
    return 0


def _recovered_module(file_ast, symbols, entries) -> ir3_mod.ModuleIR3 | None:
    """Run one file's `entries` through the full pass-1→3 chain and
    return the recovered IR3 module pass 4 emits from — or `None` when
    the file lifts no routines. Shared by `_emit_all_artifacts` and
    `lift_all_modules` so both consume an identical pipeline
    (`smc → structure → reloop → fold → match → recover_loops →
    recover_temps`)."""
    module = recognize_smc(lift_file(file_ast, symbols, entries).module)
    if not module.routines:
        return None
    ir3 = reloop_module(structure_module(module))
    return recover_temps(recover_loops(recognize_module(fold_module(ir3))))


def lift_all_modules(src_dir: Path) -> list[ir3_mod.ModuleIR3]:
    """Lift every code-bearing `.S` file through the full pass-1→3 chain
    and return the recovered IR3 modules — the program-wide module set
    crate-assembly analysis (issue #47) reasons over. Mirrors
    `_emit_all_artifacts`'s discovery/skip rules but yields the module
    objects instead of emitted text."""
    files = sorted(src_dir.glob("*.S"))
    base = [p for p in (src_dir / "EQ.S", src_dir / "GAMEEQ.S") if p.exists()]
    others = [p for p in files if p not in base]
    ast = parse_files([*base, *others], search_paths=[src_dir])
    symbols = ast.symbols()

    modules: list[ir3_mod.ModuleIR3] = []
    for src_path in files:
        file_ast = next(
            (f for f in ast.files if Path(f.path).resolve() == src_path.resolve()),
            None,
        )
        if file_ast is None:
            continue
        entries = discover_entries(file_ast)
        if not entries:
            continue
        recovered = _recovered_module(file_ast, symbols, entries)
        if recovered is not None:
            modules.append(recovered)
    return modules


def _emit_all_artifacts(src_dir: Path) -> list[tuple[str, str]]:
    """Sweep every code-bearing `.S` file through the full pass-1→4
    chain and return `[(filename, content)]` — one `<NAME>.rs` per
    lifted module plus a trailing `SUMMARY.md`. Pure (performs no
    writes) so `_cmd_emit_all` and the regen test share one
    implementation. Deterministic, so the output can be pinned.

    The chain mirrors `pop-lifter emit` exactly
    (`smc → structure → reloop → fold → match → recover_loops →
    recover_temps → emit_module`); it is intentionally strict (a
    routine the pipeline can't process raises rather than being
    skipped) so the pinned tree can never silently drop coverage."""
    files = sorted(src_dir.glob("*.S"))
    base = [p for p in (src_dir / "EQ.S", src_dir / "GAMEEQ.S") if p.exists()]
    others = [p for p in files if p not in base]
    ast = parse_files([*base, *others], search_paths=[src_dir])
    symbols = ast.symbols()

    artifacts: list[tuple[str, str]] = []
    rows: list[tuple[str, int, int, int]] = []  # file, routines, lowered, deferred
    skipped: list[str] = []

    for src_path in files:
        file_ast = next(
            (f for f in ast.files if Path(f.path).resolve() == src_path.resolve()),
            None,
        )
        if file_ast is None:
            continue
        entries = discover_entries(file_ast)
        if not entries:
            skipped.append(src_path.name)
            continue
        recovered = _recovered_module(file_ast, symbols, entries)
        if recovered is None:
            skipped.append(src_path.name)
            continue
        lowered, deferred = lower_stats(recovered)
        artifacts.append((f"{src_path.stem.upper()}.rs", emit_module(recovered)))
        rows.append((src_path.name, len(recovered.routines), lowered, deferred))

    artifacts.append(("SUMMARY.md", _render_emit_all_summary(rows, skipped)))
    return artifacts


def _render_emit_all_summary(
    rows: list[tuple[str, int, int, int]], skipped: list[str]
) -> str:
    """Render the per-file `emit-all` SUMMARY.md. The lowered-% column
    is the progress signal: it climbs as later slices lower the atoms
    still emitted as `// raw:` / `// TODO(pass4)` comments."""
    lines = [
        "# Pass-4 full Rust emission — per-file summary",
        "",
        f"Generated by `pop-lifter emit-all`. {len(rows)} files emitted "
        f"(one `<NAME>.rs` per module), {len(skipped)} skipped (no "
        f"liftable entry points).",
        "",
        "Lowered / Deferred count *top-level* statements that became real "
        "Rust vs. those still emitted as `// raw:` / `// TODO(pass4)` "
        "comments.",
        "",
        "| File | Routines | Lowered | Deferred | Lowered % |",
        "|------|---------:|--------:|---------:|----------:|",
    ]
    t_r = t_l = t_d = 0
    for name, n_r, n_l, n_d in sorted(rows):
        tot = n_l + n_d
        pct = f"{100 * n_l / tot:.1f}%" if tot else "n/a"
        lines.append(f"| `{name}` | {n_r} | {n_l} | {n_d} | {pct} |")
        t_r += n_r
        t_l += n_l
        t_d += n_d
    tot = t_l + t_d
    pct = f"{100 * t_l / tot:.1f}%" if tot else "n/a"
    lines.append(f"| **total** | **{t_r}** | **{t_l}** | **{t_d}** | **{pct}** |")
    if skipped:
        lines += ["", "## Skipped (no liftable entry points)", ""]
        lines += [f"- `{name}`" for name in sorted(skipped)]
    lines.append("")
    return "\n".join(lines)


def _emit_crate_artifacts(src_dir: Path) -> dict[str, str]:
    """Pure crate build shared by `_cmd_emit_crate` and the regen test:
    lift the whole program and assemble it into one coherent crate."""
    return emit_crate(lift_all_modules(src_dir))


def _cmd_emit_crate(args: argparse.Namespace) -> int:
    """Assemble the lifted program into one crate under `--out-dir`
    (default `ir/crate/`): a shared `cpu` module (the single `Cpu`
    state), a shared `sym` module (address constants), an `ext` module of
    external-call stubs, and one module per POP source segment holding its
    routines as free functions over `&mut Cpu`, plus `Cargo.toml`. Unlike
    the per-file `emit-all` tree, this `cargo build`s as one crate (#47)."""
    src_dir = _resolve_source_dir(args)
    if src_dir is None:
        return 2

    out_dir = Path(args.out_dir)
    files = _emit_crate_artifacts(src_dir)
    for rel, content in files.items():
        path = out_dir / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")

    shared = ("src/lib.rs", "src/cpu.rs", "src/sym.rs", "src/ext.rs")
    n_seg = sum(1 for rel in files if rel.startswith("src/") and rel not in shared)
    print(f"wrote crate ({len(files)} files, {n_seg} segment modules) under {out_dir}")
    return 0


def _cmd_emit_all(args: argparse.Namespace) -> int:
    """Pass-4 counterpart of `lift-all`: sweep every `.S` in the source
    tree and write one `<NAME>.rs` per module under `--out-dir`
    (default `ir/raw-rs/`), plus `SUMMARY.md` tracking the lowered vs.
    deferred statement counts. Committing the tree turns every lifting
    change into a reviewable Rust diff."""
    src_dir = _resolve_source_dir(args)
    if src_dir is None:
        return 2

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    artifacts = _emit_all_artifacts(src_dir)
    for name, content in artifacts:
        (out_dir / name).write_text(content, encoding="utf-8")

    n_rs = sum(1 for name, _ in artifacts if name.endswith(".rs"))
    print(f"wrote {n_rs} .rs files + SUMMARY.md under {out_dir}")
    return 0


# ---------------------------------------------------------------- main

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="pop-lifter")
    parser.add_argument(
        "--source-root",
        default="vendor/pop-apple2",
        help="Path to the upstream Apple II source checkout.",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_parse = sub.add_parser("parse", help="Pass 0: lex/parse Merlin sources.")
    p_parse.add_argument(
        "--format", choices=("text", "json"), default="text",
        help="Output format.",
    )
    p_parse.add_argument(
        "files", nargs="*",
        help="Specific .S files to parse (default: EQ.S + GAMEEQ.S).",
    )
    p_parse.set_defaults(func=_cmd_parse)

    p_dump = sub.add_parser(
        "dump-ast",
        help="Pass 0: write equates/dum-blocks JSON to disk.",
    )
    p_dump.add_argument(
        "--out", default="ir/pass0/equates.json",
        help="Output path for the JSON dump.",
    )
    p_dump.set_defaults(func=_cmd_dump_ast)

    p_lift = sub.add_parser(
        "lift",
        help="Pass 1: lift selected entry-point routines from one or more .S "
             "files to IR1.",
    )
    p_lift.add_argument(
        "file", nargs="+",
        help=".S file (absolute path or relative to the source dir). "
             "Multiple files may be passed; each --entry is routed to the "
             "file that defines it.",
    )
    p_lift.add_argument(
        "--entry", action="append", required=True,
        help="Routine entry-point label. May be passed multiple times.",
    )
    p_lift.add_argument(
        "--out", default=None,
        help="If given, write IR1 dump here instead of stdout.",
    )
    p_lift.set_defaults(func=_cmd_lift)

    p_struct = sub.add_parser(
        "struct",
        help="Pass 2: run pass 1 on the given file(s)/entries, then "
             "fuse cmp+branch pairs into structured `if` nodes.",
    )
    p_struct.add_argument(
        "file", nargs="+",
        help=".S file (absolute path or relative to the source dir).",
    )
    p_struct.add_argument(
        "--entry", action="append", required=True,
        help="Routine entry-point label. May be passed multiple times.",
    )
    p_struct.add_argument(
        "--out", default=None,
        help="If given, write IR2 dump here instead of stdout.",
    )
    p_struct.set_defaults(func=_cmd_struct)

    p_reloop = sub.add_parser(
        "reloop",
        help="Pass 2 phase 2: run pass 1, fuse cmp+branch, elide dead "
             "flags, then reloop the goto-flow IR2 into structured IR3.",
    )
    p_reloop.add_argument(
        "file", nargs="+",
        help=".S file (absolute path or relative to the source dir).",
    )
    p_reloop.add_argument(
        "--entry", action="append", required=True,
        help="Routine entry-point label. May be passed multiple times.",
    )
    p_reloop.add_argument(
        "--out", default=None,
        help="If given, write IR3 dump here instead of stdout.",
    )
    p_reloop.set_defaults(func=_cmd_reloop)

    p_fold = sub.add_parser(
        "fold",
        help="Pass 3 slice 1: run pass 1 + 2 + reloop, then fold "
             "accumulator copy round-trips (lda SRC ; sta DST) into "
             "direct memory assignments.",
    )
    p_fold.add_argument(
        "file", nargs="+",
        help=".S file (absolute path or relative to the source dir).",
    )
    p_fold.add_argument(
        "--entry", action="append", required=True,
        help="Routine entry-point label. May be passed multiple times.",
    )
    p_fold.add_argument(
        "--out", default=None,
        help="If given, write folded IR3 dump here instead of stdout.",
    )
    p_fold.set_defaults(func=_cmd_fold)

    p_match = sub.add_parser(
        "match",
        help="Pass 3: run pass 1 + 2 + reloop + fold, then recognise the "
             "jump-table dispatch idiom as structured `match` statements.",
    )
    p_match.add_argument(
        "file", nargs="+",
        help=".S file (absolute path or relative to the source dir).",
    )
    p_match.add_argument(
        "--entry", action="append", required=True,
        help="Routine entry-point label. May be passed multiple times.",
    )
    p_match.add_argument(
        "--out", default=None,
        help="If given, write the IR3 dump here instead of stdout.",
    )
    p_match.set_defaults(func=_cmd_match)

    p_smc = sub.add_parser(
        "smc",
        help="Pass 3: run pass 1, then recognise self-modifying-code "
             "immediate patches as named operand variables.",
    )
    p_smc.add_argument(
        "file", nargs="+",
        help=".S file (absolute path or relative to the source dir).",
    )
    p_smc.add_argument(
        "--entry", action="append", required=True,
        help="Routine entry-point label. May be passed multiple times.",
    )
    p_smc.add_argument(
        "--out", default=None,
        help="If given, write the IR1 dump here instead of stdout.",
    )
    p_smc.set_defaults(func=_cmd_smc)

    p_loops = sub.add_parser(
        "loops",
        help="Pass 3: run pass 1 + 2 + reloop + fold, then recover "
             "bottom-tested loops as `do { … } while` shapes and "
             "down/up-counters as `for` loops and full-wrap delays as "
             "`repeat` loops.",
    )
    p_loops.add_argument(
        "file", nargs="+",
        help=".S file (absolute path or relative to the source dir).",
    )
    p_loops.add_argument(
        "--entry", action="append", required=True,
        help="Routine entry-point label. May be passed multiple times.",
    )
    p_loops.add_argument(
        "--out", default=None,
        help="If given, write the IR3 dump here instead of stdout.",
    )
    p_loops.set_defaults(func=_cmd_loops)

    p_temps = sub.add_parser(
        "temps",
        help="Pass 3: run pass 1 + 2 + reloop + fold + loop recovery, then "
             "recover pha/pla pairs as named scoped temporaries "
             "(`push a ; … ; a = pop` ⇒ `tmp0 = a ; … ; a = tmp0`).",
    )
    p_temps.add_argument(
        "file", nargs="+",
        help=".S file (absolute path or relative to the source dir).",
    )
    p_temps.add_argument(
        "--entry", action="append", required=True,
        help="Routine entry-point label. May be passed multiple times.",
    )
    p_temps.add_argument(
        "--out", default=None,
        help="If given, write the IR3 dump here instead of stdout.",
    )
    p_temps.set_defaults(func=_cmd_temps)

    p_emit = sub.add_parser(
        "emit",
        help="Pass 4: run the full pass-3 chain (incl. match recognition), "
             "then emit Rust source — structured control flow, raw-atom "
             "lowering, and `match` for recognised dispatches.",
    )
    p_emit.add_argument(
        "file", nargs="+",
        help=".S file (absolute path or relative to the source dir).",
    )
    p_emit.add_argument(
        "--entry", action="append", required=True,
        help="Routine entry-point label. May be passed multiple times.",
    )
    p_emit.add_argument(
        "--out", default=None,
        help="If given, write the .rs output here instead of stdout.",
    )
    p_emit.set_defaults(func=_cmd_emit)

    p_lift_all = sub.add_parser(
        "lift-all",
        help="Pass 1: sweep every .S in the source tree, auto-discover "
             "entry points, write one IR1 dump per file plus SUMMARY.md.",
    )
    p_lift_all.add_argument(
        "--out-dir", default="ir/raw",
        help="Directory to write <MODULE>.ir1 files into.",
    )
    p_lift_all.set_defaults(func=_cmd_lift_all)

    p_emit_all = sub.add_parser(
        "emit-all",
        help="Pass 4: sweep every .S in the source tree, auto-discover "
             "entry points, write one <MODULE>.rs per file plus SUMMARY.md.",
    )
    p_emit_all.add_argument(
        "--out-dir", default="ir/raw-rs",
        help="Directory to write <MODULE>.rs files into.",
    )
    p_emit_all.set_defaults(func=_cmd_emit_all)

    p_emit_crate = sub.add_parser(
        "emit-crate",
        help="Pass 4: assemble the whole lifted program into one crate — "
             "shared cpu/sym/ext modules + one module of free functions per "
             "source segment + Cargo.toml (issue #47).",
    )
    p_emit_crate.add_argument(
        "--out-dir", default="ir/crate",
        help="Directory to write the crate into.",
    )
    p_emit_crate.set_defaults(func=_cmd_emit_crate)

    args = parser.parse_args(argv)
    return args.func(args)
