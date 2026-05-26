//! `diffrun` — run one lifted routine from the generated `pop` crate and
//! dump its final CPU state, so a differential test can compare it against
//! the IR interpreter on the same initial state.
//!
//! Usage: `diffrun <segment> <name> <a> <x> <y> <c>`, with an optional
//! 65536-byte RAM seed on stdin (absent ⇒ all-zero RAM). Output on stdout
//! is binary: four register bytes `a x y c` followed by the full 64 KiB of
//! final RAM. Exit codes: `0` ran, `2` unknown (segment, name), `3` the
//! routine panicked (e.g. an out-of-range memory index the interpreter
//! would have wrapped).

use std::io::{Read, Write};
use std::panic::{self, AssertUnwindSafe};

use pop::cpu::Cpu;
use pop::dispatch;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 7 {
        eprintln!("usage: diffrun <segment> <name> <a> <x> <y> <c>");
        std::process::exit(64);
    }
    let module = &args[1];
    let name = &args[2];

    let mut cpu = Cpu::new();
    cpu.reg.a = args[3].parse().expect("a");
    cpu.reg.x = args[4].parse().expect("x");
    cpu.reg.y = args[5].parse().expect("y");
    cpu.flags.c = args[6].parse::<u8>().expect("c") != 0;

    let mut seed = Vec::new();
    std::io::stdin().read_to_end(&mut seed).expect("read seed");
    if seed.len() == 0x10000 {
        cpu.mem.copy_from_slice(&seed);
    } else if !seed.is_empty() {
        eprintln!("seed must be 0 or 65536 bytes, got {}", seed.len());
        std::process::exit(64);
    }

    // The routine may index memory out of range where the interpreter
    // wraps; catch the panic so the harness records it rather than dying.
    let prev = panic::take_hook();
    panic::set_hook(Box::new(|_| {}));
    let ran = panic::catch_unwind(AssertUnwindSafe(|| dispatch::call(module, name, &mut cpu)));
    panic::set_hook(prev);

    match ran {
        Ok(true) => {}
        Ok(false) => std::process::exit(2),
        Err(_) => std::process::exit(3),
    }

    let mut out = Vec::with_capacity(4 + 0x10000);
    out.extend_from_slice(&[cpu.reg.a, cpu.reg.x, cpu.reg.y, u8::from(cpu.flags.c)]);
    out.extend_from_slice(&cpu.mem[..]);
    std::io::stdout().write_all(&out).expect("write output");
}
