"""Snapshot tests pinning known addresses from upstream EQ.S and GAMEEQ.S.

Any drift here is a real signal: either upstream changed (we'd see it in
the submodule pin) or the parser regressed.
"""

from __future__ import annotations

from pop_lifter.pass0_parse import parse_files


# ---- EQ.S ----


def test_eq_main_equates(source_dir):
    ast = parse_files([source_dir / "EQ.S"], search_paths=[source_dir])
    eq = ast.equates
    assert eq["rw18"] == 0xD000
    assert eq["peelbuf1"] == 0xD000
    assert eq["peelbuf2"] == 0xD800
    assert eq["hrtables"] == 0xE000
    assert eq["unpack"] == 0xEA00
    assert eq["hires"] == 0xEE00
    assert eq["master"] == 0xF880
    assert eq["grafix"] == 0x400
    assert eq["blueprnt"] == 0xB700


def test_eq_master_jumptable_slots(source_dir):
    ast = parse_files([source_dir / "EQ.S"], search_paths=[source_dir])
    eq = ast.equates
    assert eq["_firstboot"] == 0xF880
    assert eq["_loadlevel"] == 0xF883
    assert eq["_reload"] == 0xF886
    assert eq["_loadstage2"] == 0xF889
    # the `ds 3` (anonymous) at +12 leaves a gap; next named entry
    assert eq["_attractmode"] == 0xF880 + 15


def test_eq_hrtables(source_dir):
    ast = parse_files([source_dir / "EQ.S"], search_paths=[source_dir])
    eq = ast.equates
    assert eq["YLO"] == 0xE000
    assert eq["YHI"] == 0xE0C0


def test_eq_blueprint_layout(source_dir):
    ast = parse_files([source_dir / "EQ.S"], search_paths=[source_dir])
    eq = ast.equates
    # dum blueprnt @ $b700; BLUETYPE/BLUESPEC are 24*30 each,
    # then LINKLOC and LINKMAP are 256 each, then MAP is 24*4.
    assert eq["BLUETYPE"] == 0xB700
    assert eq["BLUESPEC"] == 0xB700 + 24 * 30
    assert eq["LINKLOC"] == 0xB700 + 2 * 24 * 30
    assert eq["LINKMAP"] == eq["LINKLOC"] + 256
    assert eq["MAP"] == eq["LINKMAP"] + 256
    assert eq["INFO"] == eq["MAP"] + 24 * 4
    # INFO @ $bf00; KidStartScrn is after the 64-byte pad.
    assert eq["INFO"] == 0xBF00
    assert eq["KidStartScrn"] == 0xBF00 + 64


def test_eq_constants(source_dir):
    ast = parse_files([source_dir / "EQ.S"], search_paths=[source_dir])
    eq = ast.equates
    assert eq["ScrnWidth"] == 140
    assert eq["ScrnHeight"] == 192
    assert eq["ScrnRight"] == 58 + 140 - 1  # 197
    assert eq["ScrnBottom"] == 0 + 192 - 1  # 191
    assert eq["secmask"] == 0b11000000
    assert eq["reqmask"] == 0b00100000
    assert eq["idmask"] == 0b00011111
    # opcode mnemonics shadowed as integers
    assert eq["and"] == 0
    assert eq["ora"] == 1
    assert eq["sta"] == 2
    assert eq["eor"] == 3
    assert eq["mask"] == 4


def test_eq_height_width_aliases(source_dir):
    ast = parse_files([source_dir / "EQ.S"], search_paths=[source_dir])
    eq = ast.equates
    assert eq["height"] == eq["IMAGE"]
    assert eq["width"] == eq["IMAGE"] + 1


# ---- GAMEEQ.S (parsed together with EQ.S because it references symbols
# defined there only indirectly; both load cleanly stand-alone too) ----


def test_gameeq_module_bases(source_dir):
    ast = parse_files([source_dir / "GAMEEQ.S"], search_paths=[source_dir])
    eq = ast.equates
    assert eq["topctrl"] == 0x2000
    assert eq["seqtable"] == 0x2800
    assert eq["ctrl"] == 0x3A00
    assert eq["coll"] == 0x4500
    assert eq["auto"] == 0x5400
    assert eq["mobtables"] == 0xB600
    assert eq["savedgame"] == 0xB6F0
    assert eq["mover"] == 0xEE00
    assert eq["misc"] == 0xF900


def test_gameeq_mob_parallel_arrays(source_dir):
    ast = parse_files([source_dir / "GAMEEQ.S"], search_paths=[source_dir])
    eq = ast.equates
    # trobspace=$20 (32), mobspace=$10 (16), each `ds` is sized accordingly
    assert eq["trobspace"] == 0x20
    assert eq["mobspace"] == 0x10
    assert eq["maxsfx"] == 0x20
    # trloc through trdirec sit on the trobspace grid:
    assert eq["trloc"] == 0xB600
    assert eq["trscrn"] == 0xB620
    assert eq["trdirec"] == 0xB640
    # mobx through moblevel sit on the mobspace grid:
    assert eq["mobx"] == 0xB660
    assert eq["moby"] == 0xB670
    assert eq["mobscrn"] == 0xB680
    assert eq["mobvel"] == 0xB690
    assert eq["mobtype"] == 0xB6A0
    assert eq["moblevel"] == 0xB6B0
    assert eq["soundtable"] == 0xB6C0
    assert eq["trobcount"] == 0xB6E0


def test_gameeq_savedgame(source_dir):
    ast = parse_files([source_dir / "GAMEEQ.S"], search_paths=[source_dir])
    eq = ast.equates
    assert eq["SavLevel"] == 0xB6F0
    assert eq["SavStrength"] == 0xB6F1
    assert eq["SavMaxed"] == 0xB6F2
    assert eq["SavTimer"] == 0xB6F3  # 2 bytes
    # then an anonymous `ds 1` pads to $b6f6
    assert eq["SavNextMsg"] == 0xB6F6


def test_gameeq_jumptable_slots(source_dir):
    ast = parse_files([source_dir / "GAMEEQ.S"], search_paths=[source_dir])
    eq = ast.equates
    # mover @ $ee00; first slot animtrans, next trigspikes at +3, etc.
    assert eq["animtrans"] == 0xEE00
    assert eq["trigspikes"] == 0xEE03
    assert eq["pushpp"] == 0xEE06
    # ctrl @ $3a00
    assert eq["PlayerCtrl"] == 0x3A00
    assert eq["checkfloor"] == 0x3A03
    assert eq["ShadCtrl"] == 0x3A06


def test_gameeq_char_overlay(source_dir):
    ast = parse_files([source_dir / "GAMEEQ.S"], search_paths=[source_dir])
    eq = ast.equates
    # `Char`, `Kid`, `Shad` are 16-byte slots in `dum $40` zero-page;
    # `Op` is a separate 16-byte slot in `dum $320`. The trailing
    # `dum Char`/`dum Op`/`dum Kid`/`dum Shad` overlays name the per-
    # character fields and resolve at those bases.
    assert eq["Char"] == 0x40
    assert eq["Kid"] == 0x50
    assert eq["Shad"] == 0x60
    assert eq["Op"] == 0x39C
    assert eq["CharPosn"] == 0x40
    assert eq["CharX"] == 0x41
    assert eq["KidPosn"] == 0x50
    assert eq["KidX"] == 0x51
    assert eq["ShadPosn"] == 0x60
    assert eq["OpPosn"] == 0x39C
    assert eq["OpX"] == 0x39D


def test_gameeq_type_constants(source_dir):
    ast = parse_files([source_dir / "GAMEEQ.S"], search_paths=[source_dir])
    eq = ast.equates
    assert eq["TypeKid"] == 0
    assert eq["TypeShad"] == 1
    assert eq["TypeGd"] == 2
    assert eq["TypeSword"] == 3
    assert eq["TypeReflect"] == 4
    assert eq["TypeComix"] == 5
    assert eq["TypeFF"] == 0x80


def test_dum_blocks_recorded(source_dir):
    ast = parse_files([source_dir / "GAMEEQ.S"], search_paths=[source_dir])
    # spot-check that the mobtables block was captured with parallel arrays
    mob = next(b for b in ast.dum_blocks if b.start_expr == "mobtables")
    names = [f.name for f in mob.fields if f.name]
    assert names[:3] == ["trloc", "trscrn", "trdirec"]
    assert "mobx" in names and "moblevel" in names

    # the savedgame block has a 1-byte anonymous pad between SavTimer and
    # SavNextMsg
    sav = next(b for b in ast.dum_blocks if b.start_expr == "savedgame")
    sizes = [(f.name, f.size) for f in sav.fields]
    assert sizes == [
        ("SavLevel", 1),
        ("SavStrength", 1),
        ("SavMaxed", 1),
        ("SavTimer", 2),
        (None, 1),
        ("SavNextMsg", 1),
    ]


def test_no_diagnostics_on_eq_files(source_dir):
    ast = parse_files(
        [source_dir / "EQ.S", source_dir / "GAMEEQ.S"], search_paths=[source_dir]
    )
    assert ast.diagnostics == []


def test_dum_with_no_operand_is_dropped(tmp_path):
    # A `dum` line with no address operand cannot be honored: there is no
    # base address for the `ds` fields that follow. The parser should
    # warn and then *skip* the malformed block — neither appending it to
    # the AST nor letting subsequent `ds` lines define equates at the
    # implicit (and incorrect) address 0.
    f = tmp_path / "bad_dum.S"
    f.write_text(
        " dum\n"
        "ghost ds 1\n"
        " dend\n"
        "real = $1234\n"
    )
    from pop_lifter.pass0_parse import parse_files
    ast = parse_files([f])
    assert "ghost" not in ast.equates
    assert ast.equates["real"] == 0x1234
    assert ast.dum_blocks == []
    assert any("dum with no operand" in d for d in ast.diagnostics)
