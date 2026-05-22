# pop-lifter

The lifter pipeline for `pop-rs`. Translates Apple II 6502 assembly written
in Merlin 16+ syntax into Rust, via three intermediate representations.

| Module           | Pass | Purpose                                        |
|------------------|------|------------------------------------------------|
| `pass0_lex`      | 0    | Tokenize Merlin syntax                         |
| `pass0_parse`    | 0    | AST + symbol table; follow `put` includes      |
| `pass1_lift`     | 1    | Per-opcode lift to C-like IR1                  |
| `pass2_struct`   | 2    | CFG → structured control flow (IR2)            |
| `pass3_domain`   | 3    | Domain abstraction (backend, data tables)      |
| `pass4_emit_rust`| 4    | Rust source emission                           |

See the top-level `docs/architecture.md` for the full design.
