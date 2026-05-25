"""Tests for `pass4_crate` — the crate-assembly analysis foundation
(issue #47, slice 1): the program-wide name → owning-module map and the
call-resolution policy.

Synthetic unit tests pin the policy itself (the four dispositions, the
intra-module preference, the call-site walker). One integration test
over the vendored POP source checks the headline findings the slice
exists to establish — chiefly that overlay name reuse is real
(`DoAdvance` lives in both AUTO and CTRL) yet resolvable, because no
overlay calls a reused name it doesn't itself define.
"""

from __future__ import annotations

from pop_lifter import ir1
from pop_lifter import pass4_crate as pc
from pop_lifter.ir1 import Compare, Reg, SourceRef
from pop_lifter.ir3 import (
    Block,
    CallStmt,
    DispatchArm,
    DispatchStmt,
    IfStmt,
    LoopStmt,
    MatchArm,
    MatchStmt,
    ModuleIR3,
    RawStmt,
    ReturnStmt,
    RoutineIR3,
    TailCallStmt,
)

_SR = SourceRef("X.S", 1, "raw")


def _routine(name: str, stmts: list, aliases: list[str] | None = None) -> RoutineIR3:
    return RoutineIR3(name=name, entry_aliases=aliases or [], body=Block.of(stmts))


def _module(name: str, routines: list[RoutineIR3]) -> ModuleIR3:
    return ModuleIR3(name=name, file=f"{name}.S", routines=routines)


# ---------------------------------------------------------------- name map


def test_defined_entry_names_includes_aliases():
    m = _module("A", [_routine("Foo", [], aliases=["Bar", "Baz"])])
    assert pc.defined_entry_names(m) == {"Foo", "Bar", "Baz"}


def test_build_name_map_groups_owners_sorted():
    mods = [
        _module("B", [_routine("Shared", [])]),
        _module("A", [_routine("Shared", []), _routine("OnlyA", [])]),
    ]
    name_map = pc.build_name_map(mods)
    # Owners come back sorted regardless of module order.
    assert name_map["Shared"] == ["A", "B"]
    assert name_map["OnlyA"] == ["A"]


def test_collisions_are_multi_owner_names_only():
    name_map = {"Shared": ["A", "B"], "OnlyA": ["A"], "Triple": ["A", "B", "C"]}
    assert pc.collisions(name_map) == {"Shared": ["A", "B"], "Triple": ["A", "B", "C"]}


# ---------------------------------------------------------------- resolution policy


def test_resolve_intra_module_preferred_even_when_name_is_reused():
    name_map = {"Foo": ["A", "B"]}
    res = pc.resolve_call(name_map, "A", "Foo")
    assert res.disposition is pc.CallDisposition.INTRA_MODULE
    assert res.owner == "A"
    assert res.candidates == ("A", "B")


def test_resolve_cross_module_unique_owner():
    name_map = {"Foo": ["A"]}
    res = pc.resolve_call(name_map, "B", "Foo")
    assert res.disposition is pc.CallDisposition.CROSS_MODULE
    assert res.owner == "A"
    assert res.candidates == ("A",)


def test_resolve_external_when_no_module_defines_target():
    res = pc.resolve_call({"Foo": ["A"]}, "A", "RomRoutine")
    assert res.disposition is pc.CallDisposition.EXTERNAL
    assert res.owner is None
    assert res.candidates == ()


def test_resolve_ambiguous_when_caller_absent_and_many_owners():
    name_map = {"Foo": ["A", "B"]}
    res = pc.resolve_call(name_map, "C", "Foo")
    assert res.disposition is pc.CallDisposition.AMBIGUOUS
    assert res.owner is None
    assert res.candidates == ("A", "B")


def test_resolve_intra_wins_for_triple_owner_when_caller_among_them():
    name_map = {"Foo": ["A", "B", "C"]}
    assert pc.resolve_call(name_map, "C", "Foo").disposition is pc.CallDisposition.INTRA_MODULE
    assert pc.resolve_call(name_map, "D", "Foo").disposition is pc.CallDisposition.AMBIGUOUS


# ---------------------------------------------------------------- call-site walker


def test_collect_calls_walks_nested_blocks_and_raw_stmts():
    inner_match = MatchStmt(
        reg=Reg.A,
        arms=(
            MatchArm(values=(), body=Block.of([CallStmt("InMatch", _SR)])),
        ),
        src=_SR,
    )
    loop = LoopStmt(body=Block.of([inner_match]), src=_SR)
    iff = IfStmt(
        cond=Compare(Reg.A, "==", None),
        then_block=Block.of([CallStmt("InThen", _SR), loop]),
        else_block=Block.of([TailCallStmt("InElse", _SR)]),
        src=_SR,
    )
    dispatch = DispatchStmt(
        entry=0,
        arms=(DispatchArm(state=0, body=Block.of([CallStmt("InDispatch", _SR)])),),
        src=_SR,
    )
    # A call / tail-call still wrapped in a RawStmt (pre-reloop shape).
    raw_call = RawStmt(ir1.Call("RawCall", _SR))
    raw_tail = RawStmt(ir1.Goto("RawTail", "tail_call", _SR))
    raw_local = RawStmt(ir1.Goto("LocalLabel", "local", _SR))  # not a call
    r = _routine("Top", [iff, dispatch, raw_call, raw_tail, raw_local, ReturnStmt(_SR)])

    sites = pc.routine_call_sites(_module("M", [r]), r)
    pairs = [(s.kind, s.target) for s in sites]
    assert pairs == [
        ("call", "InThen"),
        ("call", "InMatch"),
        ("tail_call", "InElse"),
        ("call", "InDispatch"),
        ("call", "RawCall"),
        ("tail_call", "RawTail"),
    ]
    assert all(s.caller_module == "M" and s.caller_routine == "Top" for s in sites)


# ---------------------------------------------------------------- whole-program


def test_analyze_program_classifies_every_call_site():
    # B::Worker calls Helper (defined only in A → cross) and itself calls
    # Shared (defined in both → intra from B). A::Shared tail-calls a ROM
    # routine (external).
    a = _module("A", [
        _routine("Helper", [ReturnStmt(_SR)]),
        _routine("Shared", [TailCallStmt("MonitorBell", _SR)]),
    ])
    b = _module("B", [
        _routine("Worker", [CallStmt("Helper", _SR), CallStmt("Shared", _SR)]),
        _routine("Shared", [ReturnStmt(_SR)]),
    ])
    g = pc.analyze_program([a, b])

    assert g.collisions == {"Shared": ["A", "B"]}
    counts = g.disposition_counts()
    assert counts[pc.CallDisposition.CROSS_MODULE] == 1     # B → A::Helper
    assert counts[pc.CallDisposition.INTRA_MODULE] == 1     # B → B::Shared
    assert counts[pc.CallDisposition.EXTERNAL] == 1         # A::Shared → MonitorBell
    assert counts[pc.CallDisposition.AMBIGUOUS] == 0
    assert sum(counts.values()) == len(g.resolutions) == len(g.call_sites) == 3
    assert g.unresolved_targets() == {"MonitorBell"}


def test_analyze_program_over_pop_source(source_dir):
    """Headline findings on the real POP tree: overlay name reuse is
    real but resolvable under the intra-module-preference policy."""
    from pop_lifter.cli import lift_all_modules

    modules = lift_all_modules(source_dir)
    names = {m.name for m in modules}
    assert {"AUTO", "CTRL", "GRAFIX", "HIRES"} <= names

    g = pc.analyze_program(modules)

    # `DoAdvance` is the canonical example: defined *differently* in the
    # AUTO and CTRL overlay segments. Several combat verbs reuse names the
    # same way.
    assert g.name_map["DoAdvance"] == ["AUTO", "CTRL"]
    assert "DoAdvance" in g.collisions
    for verb in ("DoStrike", "DoBlock", "DoTurn"):
        assert g.name_map[verb] == ["AUTO", "CTRL"]

    counts = g.disposition_counts()
    # The foundational finding: no overlay calls a reused name it doesn't
    # itself define, so intra-module preference uniquely resolves every
    # collision and nothing is ambiguous.
    assert counts[pc.CallDisposition.AMBIGUOUS] == 0
    # Every call site got exactly one resolution.
    assert len(g.resolutions) == len(g.call_sites)
    assert sum(counts.values()) == len(g.resolutions)
    # All three resolvable dispositions occur in the real program.
    assert counts[pc.CallDisposition.INTRA_MODULE] > 0
    assert counts[pc.CallDisposition.CROSS_MODULE] > 0
    assert counts[pc.CallDisposition.EXTERNAL] > 0

    # A collision target called from a defining module binds to that
    # module — never the other segment's same-named routine.
    res = pc.resolve_call(g.name_map, "AUTO", "DoAdvance")
    assert res.disposition is pc.CallDisposition.INTRA_MODULE
    assert res.owner == "AUTO"
