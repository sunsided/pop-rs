"""IR1 interpreter tests: execute the lifted pilot routines and check
the resulting RAM diff.

This is the pass-1 half of the differential test harness from
`docs/architecture.md`. A subsequent slice will replace the
hand-computed expected diffs here with snapshots from an emulator
harness; until then the values are pinned against what each routine
*literally* writes per its source.
"""

from __future__ import annotations

from pathlib import Path

from pop_lifter.interp_ir1 import run
from pop_lifter.pass0_parse import parse_files
from pop_lifter.pass1_lift import lift_file


# Addresses come from the equate files. Pinned here as concrete bytes so
# a regression in pass-0 equate resolution is also caught.
CLR_F = 0xD4
CLR_B = 0xD5
CLR_U = 0xD6
CLR_D = 0xD7
CLR_BTN = 0xD8
JSTKX = 0x18
JSTKY = 0x19
BTN = 0x3D

PILOT_ENTRIES = [
    "DoStrike", "DoBlock", "DoTurn",
    "DoStandup", "DoEngarde", "DoRelBtn", "DoRelease",
]


def _module(source_dir: Path):
    auto = source_dir / "AUTO.S"
    ast = parse_files(
        [source_dir / "EQ.S", source_dir / "GAMEEQ.S", auto],
        search_paths=[source_dir],
    )
    file_ast = next(f for f in ast.files if Path(f.path).name == "AUTO.S")
    return lift_file(file_ast, ast.symbols(), PILOT_ENTRIES).module


def _run_pilot(source_dir, entry):
    module = _module(source_dir)
    return run(module, entry)


def test_dostrike_writes_ff_to_clrbtn_and_btn(source_dir):
    trace = _run_pilot(source_dir, "DoStrike")
    assert trace.writes == {CLR_BTN: 0xff, BTN: 0xff}
    assert trace.a == 0xff


def test_dopress_alias_executes_dostrike(source_dir):
    # The alias must resolve to the same routine and produce the same
    # effects.
    a = _run_pilot(source_dir, "DoStrike").writes
    b = _run_pilot(source_dir, "DoPress").writes
    assert a == b


def test_doblock_writes_ff_to_clru_and_jstky(source_dir):
    trace = _run_pilot(source_dir, "DoBlock")
    assert trace.writes == {CLR_U: 0xff, JSTKY: 0xff}


def test_doturn_writes_ff_to_clrd_then_1_to_jstky(source_dir):
    trace = _run_pilot(source_dir, "DoTurn")
    # Two distinct values written; the JSTKY=1 store happens after the
    # second lda reloads A.
    assert trace.writes == {CLR_D: 0xff, JSTKY: 0x01}
    assert trace.a == 0x01


def test_dostandup_tail_calls_into_doback(source_dir):
    # DoStandup writes clrU=0xff itself, then tail-calls DoBack which
    # writes clrB=0xff and JSTKX=0x01.
    trace = _run_pilot(source_dir, "DoStandup")
    assert trace.writes == {
        CLR_U: 0xff,
        CLR_B: 0xff,
        JSTKX: 0x01,
    }


def test_doengarde_tail_calls_into_dofwd(source_dir):
    # DoEngarde writes clrD=0xff, then tail-calls DoFwd which writes
    # clrF=0xff and JSTKX=0xff.
    trace = _run_pilot(source_dir, "DoEngarde")
    assert trace.writes == {
        CLR_D: 0xff,
        CLR_F: 0xff,
        JSTKX: 0xff,
    }


def test_dorelbtn_zeroes_btn_only(source_dir):
    trace = _run_pilot(source_dir, "DoRelBtn")
    # The `]rts` label is just a marker; only the explicit store happens.
    assert trace.writes == {BTN: 0x00}
    assert trace.a == 0x00


def test_dorelease_zeroes_eight_buttons(source_dir):
    trace = _run_pilot(source_dir, "DoRelease")
    assert trace.writes == {
        CLR_F: 0x00, CLR_B: 0x00, CLR_U: 0x00, CLR_D: 0x00,
        CLR_BTN: 0x00, JSTKX: 0x00, JSTKY: 0x00, BTN: 0x00,
    }


def test_ram_outside_writes_is_untouched(source_dir):
    trace = _run_pilot(source_dir, "DoStrike")
    # Verify the interpreter doesn't scribble outside the addresses it
    # claims to have written. A simple checksum suffices.
    untouched = sum(trace.ram[i] for i in range(0x10000) if i not in trace.writes)
    assert untouched == 0


def test_step_limit_is_exclusive_upper_bound(source_dir):
    """`max_steps=N` permits exactly N executed IR1 items, then raises.

    The pilot routine `DoStrike` executes 4 items (LoadImm, StoreAbs,
    StoreAbs, Return). With `max_steps=4` the run must complete; with
    `max_steps=3` it must raise before reaching the Return.
    """
    import pytest

    from pop_lifter.interp_ir1 import InterpError

    module = _module(source_dir)
    # Allow exactly the four items DoStrike executes — should succeed.
    trace = run(module, "DoStrike", max_steps=4)
    assert trace.steps == 4
    # One fewer than required — should raise.
    with pytest.raises(InterpError, match="step limit"):
        run(module, "DoStrike", max_steps=3)


# ---- rndp + RND slice: new opcodes, cross-module tail call, jsr stack


RND_SEED_ADDR = 0x9E
GUARDPROG_ADDR = 0xDD


def _two_module(source_dir):
    from pop_lifter.pass1_lift import lift_file
    ast = parse_files(
        [
            source_dir / "EQ.S",
            source_dir / "GAMEEQ.S",
            source_dir / "AUTO.S",
            source_dir / "GRAFIX.S",
        ],
        search_paths=[source_dir],
    )
    auto = next(f for f in ast.files if Path(f.path).name == "AUTO.S")
    grafix = next(f for f in ast.files if Path(f.path).name == "GRAFIX.S")
    m_auto = lift_file(auto, ast.symbols(), ["rndp"]).module
    m_grafix = lift_file(grafix, ast.symbols(), ["RND"]).module
    return m_auto, m_grafix


def test_rnd_computes_5seed_plus_23(source_dir):
    """Per the source comment in GRAFIX.S: `RNDseed := (5 * RNDseed + 23) mod 256`.
    We pin a handful of representative seed values."""
    _, grafix = _two_module(source_dir)
    for seed in (0, 1, 7, 23, 100, 200, 255):
        ram = bytearray(0x10000)
        ram[RND_SEED_ADDR] = seed
        trace = run(grafix, "RND", ram=ram)
        expected = (5 * seed + 23) & 0xff
        assert ram[RND_SEED_ADDR] == expected, (
            f"seed={seed}: RND wrote {ram[RND_SEED_ADDR]:#04x}, "
            f"expected {expected:#04x}"
        )
        assert trace.a == expected


def test_rnd_carry_after_first_asl_does_not_corrupt_5seed(source_dir):
    """The first `asl` can set carry; the subsequent `adc` happens
    after a `clc`, so the carry must NOT contribute. Verify with a
    seed whose top bit is 1, which guarantees C=1 after `asl`."""
    _, grafix = _two_module(source_dir)
    seed = 0x80  # bit 7 set → asl produces 0x00 with C=1
    ram = bytearray(0x10000)
    ram[RND_SEED_ADDR] = seed
    trace = run(grafix, "RND", ram=ram)
    # 5*0x80 = 0x280; mod 256 = 0x80; +23 = 0x97
    assert ram[RND_SEED_ADDR] == (5 * seed + 23) & 0xff == 0x97
    assert trace.c == 0  # last adc #23 doesn't overflow when low byte is small


def test_rndp_tail_calls_into_RND_via_alias(source_dir):
    """`rndp` loads guardprog into X and tail-calls `rnd`, which is a
    grafix jump-table slot pointing at GRAFIX::RND. The interpreter
    resolves the slot via the explicit alias map."""
    auto, grafix = _two_module(source_dir)
    ram = bytearray(0x10000)
    ram[GUARDPROG_ADDR] = 7
    ram[RND_SEED_ADDR] = 100
    trace = run(
        [auto, grafix],
        "rndp",
        ram=ram,
        aliases={"rnd": "RND"},
    )
    # rndp loaded guardprog into X
    assert trace.x == 7
    # RND ran end-to-end and updated the seed
    assert ram[RND_SEED_ADDR] == (5 * 100 + 23) & 0xff
    # tail-call doesn't grow the stack
    assert trace.max_stack_depth == 0


# ---- CheckFloor slice: cmp + branches + Z/N flags, every code path


# Addresses the CHECKFLOOR routine reads. Pinned here so a regression
# in pass-0 equate resolution also fails this test.
CHAR_ACTION = 0x46
CHAR_POSN = 0x40


def _checkfloor_module(source_dir):
    """Lift CHECKFLOOR plus tiny stub routines for the three tail-call
    targets (`onground`, `falling`, `fallon`) so every path can run."""
    from pop_lifter.ir1 import (
        Abs,
        Imm,
        LoadImm,
        ModuleIR1,
        Reg,
        Return,
        Routine,
        SourceRef,
        StoreAbs,
    )
    from pop_lifter.pass1_lift import lift_file

    ast = parse_files(
        [
            source_dir / "EQ.S",
            source_dir / "GAMEEQ.S",
            source_dir / "CTRL.S",
        ],
        search_paths=[source_dir],
    )
    ctrl = next(f for f in ast.files if Path(f.path).name == "CTRL.S")
    real = lift_file(ctrl, ast.symbols(), ["CHECKFLOOR"]).module

    # The lifted module already includes a partial `falling`, `fallon`,
    # `onground` (the tail-call chase pulled them in). Replace them
    # with minimal stubs that just record their own visit by writing
    # a sentinel byte to a known address — that lets us assert which
    # tail-call CHECKFLOOR took.
    SENTINEL_ONGROUND = 0x200
    SENTINEL_FALLING = 0x201
    SENTINEL_FALLON = 0x202

    src = SourceRef(file="synthetic", line=0, raw="")

    def _stub(name: str, sentinel_addr: int) -> Routine:
        return Routine(
            name=name,
            body=[
                LoadImm(reg=Reg.A, imm=Imm(value=1, text="#1"), src=src),
                StoreAbs(
                    reg=Reg.A,
                    target=Abs(name=f"<{name}_sentinel>", addr=sentinel_addr),
                    src=src,
                ),
                Return(src=src),
            ],
        )

    stubs = ModuleIR1(
        name="STUBS",
        file="synthetic",
        routines=[
            _stub("onground", SENTINEL_ONGROUND),
            _stub("falling", SENTINEL_FALLING),
            _stub("fallon", SENTINEL_FALLON),
        ],
    )

    # Strip the real (Unsupported-laden) implementations from the
    # CTRL module so the interpreter doesn't trip on them.
    real.routines = [
        r for r in real.routines
        if r.name not in ("onground", "falling", "fallon")
    ]
    return real, stubs, SENTINEL_ONGROUND, SENTINEL_FALLING, SENTINEL_FALLON


def _run_checkfloor(source_dir, action: int, posn: int = 0):
    ctrl, stubs, _, _, _ = _checkfloor_module(source_dir)
    ram = bytearray(0x10000)
    ram[CHAR_ACTION] = action
    ram[CHAR_POSN] = posn
    return run([ctrl, stubs], "CHECKFLOOR", ram=ram), ram


def test_checkfloor_action_6_returns_immediately(source_dir):
    """CharAction = 6 (hanging) — first `cmp #6; beq ]rts` exits."""
    trace, ram = _run_checkfloor(source_dir, action=6)
    # No tail-call sentinels touched.
    assert ram[0x200] == ram[0x201] == ram[0x202] == 0
    # The `]rts` synthesized trampoline returned at stack depth 0.
    assert trace.max_stack_depth == 0


def test_checkfloor_action_2_returns_immediately(source_dir):
    """CharAction = 2 (hanging) — reaches `:1 cmp #2; beq ]rts`."""
    trace, ram = _run_checkfloor(source_dir, action=2)
    assert ram[0x200] == ram[0x201] == ram[0x202] == 0
    assert trace.z == 1  # last cmp matched


def test_checkfloor_action_5_other_posn_returns(source_dir):
    """CharAction = 5 (bumped), CharPosn not in {109, 185} — the
    `cmp #185; bne ]rts` path exits."""
    trace, ram = _run_checkfloor(source_dir, action=5, posn=42)
    assert ram[0x200] == ram[0x201] == ram[0x202] == 0
    # bne taken means Z=0 at the moment of the branch.
    assert trace.z == 0


def test_checkfloor_action_5_posn_109_tail_calls_onground(source_dir):
    """CharAction = 5, CharPosn = 109 (crouched on loose floor) —
    `:ong jmp onground`."""
    _, ram = _run_checkfloor(source_dir, action=5, posn=109)
    assert ram[0x200] == 1   # onground sentinel
    assert ram[0x201] == 0   # falling untouched
    assert ram[0x202] == 0   # fallon untouched


def test_checkfloor_action_5_posn_185_tail_calls_onground(source_dir):
    """CharAction = 5, CharPosn = 185 (dead) — falls through to
    `:ong jmp onground` rather than the `bne ]rts`."""
    _, ram = _run_checkfloor(source_dir, action=5, posn=185)
    assert ram[0x200] == 1


def test_checkfloor_action_4_branches_to_falling(source_dir):
    """CharAction = 4 (freefall) — `cmp #4; beq falling`. This is the
    cross-routine conditional branch path the interpreter has to
    resolve via the module list."""
    _, ram = _run_checkfloor(source_dir, action=4)
    assert ram[0x201] == 1   # falling sentinel
    assert ram[0x200] == 0
    assert ram[0x202] == 0


def test_checkfloor_action_3_posn_in_range_calls_fallon(source_dir):
    """CharAction = 3, CharPosn in [102, 105] — both `bcc ]rts` and
    `bcs ]rts` skipped, hitting `jmp fallon`."""
    _, ram = _run_checkfloor(source_dir, action=3, posn=104)
    assert ram[0x202] == 1   # fallon sentinel


def test_checkfloor_action_3_posn_below_returns(source_dir):
    """CharAction = 3, CharPosn = 50 (< 102) — `cmp #102; bcc ]rts`."""
    _, ram = _run_checkfloor(source_dir, action=3, posn=50)
    assert ram[0x200] == ram[0x201] == ram[0x202] == 0


def test_checkfloor_action_3_posn_above_returns(source_dir):
    """CharAction = 3, CharPosn = 200 (>= 106) — `cmp #106; bcs ]rts`."""
    _, ram = _run_checkfloor(source_dir, action=3, posn=200)
    assert ram[0x200] == ram[0x201] == ram[0x202] == 0


def test_checkfloor_action_0_or_1_or_7_tail_calls_onground(source_dir):
    """CharAction in {0, 1, 7} reaches `:1 cmp #2; beq ]rts` (not
    taken), then `jmp onground`."""
    for action in (0, 1, 7):
        _, ram = _run_checkfloor(source_dir, action=action)
        assert ram[0x200] == 1, f"action={action} should reach onground"


# ---- flag and indexed-addressing primitives in isolation


def test_cmp_sets_carry_correctly():
    """`cmp` sets C iff `reg >= operand` (6502 datasheet)."""
    from pop_lifter.ir1 import (
        CmpImm,
        Imm,
        LoadImm,
        ModuleIR1,
        Reg,
        Return,
        Routine,
        SourceRef,
    )

    src = SourceRef(file="synthetic", line=0, raw="")

    def go(a_val: int, rhs: int) -> tuple[int, int, int]:
        module = ModuleIR1(
            name="SYN",
            file="synthetic",
            routines=[Routine(name="f", body=[
                LoadImm(reg=Reg.A, imm=Imm(value=a_val, text=""), src=src),
                CmpImm(reg=Reg.A, imm=Imm(value=rhs, text=""), src=src),
                Return(src=src),
            ])],
        )
        t = run(module, "f")
        return t.c, t.z, t.n

    assert go(5, 3) == (1, 0, 0)        # 5 > 3 → C=1, Z=0, N=0
    assert go(3, 3) == (1, 1, 0)        # equal → C=1, Z=1
    assert go(2, 3) == (0, 0, 1)        # 2 < 3 → C=0, N=1 (bit 7 of (2-3)&0xff=0xff)


def test_load_indexed_reads_ram_at_base_plus_index():
    from pop_lifter.ir1 import (
        Abs,
        Imm,
        LoadImm,
        LoadIndexed,
        ModuleIR1,
        Reg,
        Return,
        Routine,
        SourceRef,
    )

    src = SourceRef(file="synthetic", line=0, raw="")
    module = ModuleIR1(
        name="SYN",
        file="synthetic",
        routines=[Routine(name="f", body=[
            LoadImm(reg=Reg.X, imm=Imm(value=3, text=""), src=src),
            LoadIndexed(
                reg=Reg.A,
                base=Abs(name="tbl", addr=0x300),
                index=Reg.X,
                src=src,
            ),
            Return(src=src),
        ])],
    )
    ram = bytearray(0x10000)
    ram[0x300:0x305] = bytes([10, 20, 30, 40, 50])
    trace = run(module, "f", ram=ram)
    assert trace.a == 40   # ram[0x300 + 3]
    assert trace.x == 3


def test_jsr_rts_returns_to_caller(source_dir):
    """`jsr` followed by `rts` must resume at the next IR1 item, not
    at routine entry. A minimal sanity check: lift AUTOCTRL's prefix
    and run with everything past the first `jsr DoRelease` mocked out.

    For this test we synthesize a tiny module by hand — that's cheaper
    than building a fixture that survives the existing AUTOCTRL body
    of Unsupported opcodes."""
    from pop_lifter.ir1 import (
        Abs,
        Call,
        Imm,
        LoadImm,
        ModuleIR1,
        Reg,
        Return,
        Routine,
        SourceRef,
        StoreAbs,
    )

    src = SourceRef(file="synthetic", line=1, raw="")

    callee = Routine(
        name="callee",
        body=[
            LoadImm(reg=Reg.A, imm=Imm(value=0x42, text="#$42"), src=src),
            StoreAbs(reg=Reg.A, target=Abs(name="slot", addr=0x100), src=src),
            Return(src=src),
        ],
    )
    caller = Routine(
        name="caller",
        body=[
            Call(target="callee", src=src),
            # The post-return code: writes 0xAA to addr 0x101 so a test
            # can verify we resumed *after* the jsr.
            LoadImm(reg=Reg.A, imm=Imm(value=0xAA, text="#$aa"), src=src),
            StoreAbs(reg=Reg.A, target=Abs(name="post", addr=0x101), src=src),
            Return(src=src),
        ],
    )
    module = ModuleIR1(name="SYN", file="synthetic", routines=[caller, callee])

    trace = run(module, "caller")
    assert trace.ram[0x100] == 0x42
    assert trace.ram[0x101] == 0xAA
    # One nested call observed.
    assert trace.max_stack_depth == 1
