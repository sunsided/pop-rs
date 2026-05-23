"""Pass 3 — `match` recognition.

* **Synthetic unit tests** drive the recogniser over hand-built IR3 to
  pin the rules: a run of ≥2 `if reg == K { terminating }` becomes a
  `MatchStmt`; identical bodies merge into one multi-key arm; and a run
  is rejected/cut when the register differs, the op isn't `==`, a key
  repeats, an arm doesn't terminate, or a non-`if` statement intrudes.
* **Behavioural equivalence** interprets a synthetic dispatch routine
  before and after recognition for every key and a default, asserting
  byte-identical RAM — the rewrite must not change behaviour.
"""

from __future__ import annotations

from pop_lifter.interp_ir3 import run as ir3_run
from pop_lifter.ir1 import (
    Abs,
    Compare,
    Imm,
    LoadAbs,
    LoadImm,
    Reg,
    SourceRef,
    StoreAbs,
)
from pop_lifter.ir3 import (
    Block,
    IfStmt,
    MatchStmt,
    ModuleIR3,
    RawStmt,
    ReturnStmt,
    RoutineIR3,
    TailCallStmt,
)
from pop_lifter.pass3_match import match_stats, recognize_routine

SRC = SourceRef(file="syn", line=0, raw="")


def _R(it) -> RawStmt:
    return RawStmt(item=it)


def _imm(v: int) -> Imm:
    return Imm(value=v, text=f"#{v}")


def _eq_if(reg: Reg, k: int, body: list, op: str = "==") -> IfStmt:
    return IfStmt(
        cond=Compare(reg=reg, op=op, rhs=_imm(k)),
        then_block=Block.of(body),
        else_block=None,
        src=SRC,
    )


def _tail(name: str) -> list:
    return [TailCallStmt(target=name, src=SRC)]


def _recognize(stmts: list):
    return list(recognize_routine(RoutineIR3(name="syn", body=Block.of(stmts))).body.stmts)


# --------------------------------------------------------------- structural


def test_basic_dispatch_becomes_match():
    out = _recognize([
        _eq_if(Reg.A, 1, _tail("h1")),
        _eq_if(Reg.A, 2, _tail("h2")),
        _eq_if(Reg.A, 3, _tail("h3")),
    ])
    assert len(out) == 1
    match = out[0]
    assert isinstance(match, MatchStmt) and match.reg is Reg.A
    # Distinct bodies → one arm each, in order, one key apiece.
    assert [tuple(v.value for v in a.values) for a in match.arms] == [(1,), (2,), (3,)]


def test_identical_bodies_merge_into_one_arm():
    out = _recognize([
        _eq_if(Reg.A, 1, [_R(LoadImm(reg=Reg.A, imm=_imm(0), src=SRC)), ReturnStmt(src=SRC)]),
        _eq_if(Reg.A, 2, [_R(LoadImm(reg=Reg.A, imm=_imm(0), src=SRC)), ReturnStmt(src=SRC)]),
        _eq_if(Reg.A, 3, [_R(LoadImm(reg=Reg.A, imm=_imm(0), src=SRC)), ReturnStmt(src=SRC)]),
    ])
    assert len(out) == 1 and isinstance(out[0], MatchStmt)
    assert len(out[0].arms) == 1
    assert tuple(v.value for v in out[0].arms[0].values) == (1, 2, 3)


def test_single_if_is_not_a_match():
    out = _recognize([_eq_if(Reg.A, 1, _tail("h1")), ReturnStmt(src=SRC)])
    assert not any(isinstance(s, MatchStmt) for s in out)


def test_different_register_breaks_run():
    out = _recognize([
        _eq_if(Reg.A, 1, _tail("h1")),
        _eq_if(Reg.X, 2, _tail("h2")),  # different register
    ])
    assert not any(isinstance(s, MatchStmt) for s in out)


def test_non_eq_op_not_recognized():
    out = _recognize([
        _eq_if(Reg.A, 1, _tail("h1"), op=">="),
        _eq_if(Reg.A, 2, _tail("h2"), op=">="),
    ])
    assert not any(isinstance(s, MatchStmt) for s in out)


def test_non_terminating_arm_breaks_run():
    """An arm that falls through (no terminator) can't be a match arm —
    folding it in would lose the fall-through semantics."""
    out = _recognize([
        _eq_if(Reg.A, 1, [_R(StoreAbs(reg=Reg.A, target=Abs(name="X", addr=0x10), src=SRC))]),
        _eq_if(Reg.A, 2, _tail("h2")),
    ])
    assert not any(isinstance(s, MatchStmt) for s in out)


def test_duplicate_key_stops_run():
    out = _recognize([
        _eq_if(Reg.A, 1, _tail("h1")),
        _eq_if(Reg.A, 2, _tail("h2")),
        _eq_if(Reg.A, 1, _tail("h1b")),  # repeat of key 1
    ])
    matches = [s for s in out if isinstance(s, MatchStmt)]
    assert len(matches) == 1
    assert [tuple(v.value for v in a.values) for a in matches[0].arms] == [(1,), (2,)]
    # The duplicate stays as a plain if after the match.
    assert isinstance(out[-1], IfStmt)


def test_intervening_statement_breaks_run():
    out = _recognize([
        _eq_if(Reg.A, 1, _tail("h1")),
        _R(StoreAbs(reg=Reg.A, target=Abs(name="X", addr=0x10), src=SRC)),  # not an if
        _eq_if(Reg.A, 2, _tail("h2")),
    ])
    assert not any(isinstance(s, MatchStmt) for s in out)


# --------------------------------------------------------------- behavioural


def _dispatch_routine() -> RoutineIR3:
    """`a = *INPUT ; if a==1 {*OUT=0xAA;ret} if a==2 {*OUT=0xBB;ret} ;
    *OUT=0xCC ; ret` — a 2-key dispatch with a default tail."""
    def arm(val):
        return [
            _R(LoadImm(reg=Reg.A, imm=_imm(val), src=SRC)),
            _R(StoreAbs(reg=Reg.A, target=Abs(name="OUT", addr=0x301), src=SRC)),
            ReturnStmt(src=SRC),
        ]
    return RoutineIR3(name="disp", body=Block.of([
        _R(LoadAbs(reg=Reg.A, source=Abs(name="INPUT", addr=0x300), src=SRC)),
        _eq_if(Reg.A, 1, arm(0xAA)),
        _eq_if(Reg.A, 2, arm(0xBB)),
        _R(LoadImm(reg=Reg.A, imm=_imm(0xCC), src=SRC)),
        _R(StoreAbs(reg=Reg.A, target=Abs(name="OUT", addr=0x301), src=SRC)),
        ReturnStmt(src=SRC),
    ]))


def test_match_recognition_is_behaviour_preserving():
    routine = _dispatch_routine()
    matched = recognize_routine(routine)
    assert match_stats(ModuleIR3("M", "syn", [matched])) == 1

    pre = ModuleIR3("M", "syn", [routine])
    post = ModuleIR3("M", "syn", [matched])
    for inp, expected_out in [(1, 0xAA), (2, 0xBB), (3, 0xCC), (0, 0xCC)]:
        r1 = bytearray(0x10000)
        r1[0x300] = inp
        ir3_run([pre], "disp", ram=r1)
        r2 = bytearray(0x10000)
        r2[0x300] = inp
        ir3_run([post], "disp", ram=r2)
        assert r1 == r2, f"match diverged for input {inp}"
        assert r2[0x301] == expected_out
