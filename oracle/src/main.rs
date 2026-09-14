use qp_poseidon_core::poseidon2::{Poseidon2, SPONGE_WIDTH, POSEIDON2_OUTPUT};
use qp_poseidon_core::goldilocks::Goldilocks;
use qp_poseidon_core::serialization::{bytes_to_u64s, digest_to_u64s};

fn main() {
    let header = [0u8;32];
    let nonce = [0u8;64];
    let mut input = Vec::new();
    input.extend_from_slice(&header);
    input.extend_from_slice(&nonce);

    let felts = bytes_to_u64s(&input);
    println!("felts len={}:", felts.len());
    for (i,f) in felts.iter().enumerate() {
        println!("  f[{}] = {:08x}", i, f);
    }

    let mut state = [Goldilocks::from_u64(0); SPONGE_WIDTH];
    let mut buf = [Goldilocks::from_u64(0); 8];
    let mut buf_len = 0usize;

    macro_rules! push {
        ($f:expr) => {{
            buf[buf_len] = Goldilocks::from_u64($f);
            buf_len += 1;
            if buf_len == 8 {
                for i in 0..8 { state[i] = state[i] + buf[i]; }
                Poseidon2::new().permute_mut(&mut state);
                buf_len = 0;
            }
        }};
    }

    for f in felts { push!(f); }
    push!(1); // finalize ONE
    while buf_len != 0 { push!(0); }

    println!("after finalize state (as u64):");
    for (i,s) in state.iter().enumerate() {
        println!("  s[{}] = {:016x}", i, s.as_canonical_u64());
    }

    let digest1: [u8;32] = {
        let mut out = [0u8;32];
        for i in 0..POSEIDON2_OUTPUT {
            out[i*8..i*8+8].copy_from_slice(&state[i].as_canonical_u64().to_le_bytes());
        }
        out
    };
    print!("digest1: ");
    for b in digest1 { print!("{:02x}", b); }
    println!();

    Poseidon2::new().permute_mut(&mut state);
    let digest2: [u8;32] = {
        let mut out = [0u8;32];
        for i in 0..POSEIDON2_OUTPUT {
            out[i*8..i*8+8].copy_from_slice(&state[i].as_canonical_u64().to_le_bytes());
        }
        out
    };
    print!("digest2: ");
    for b in digest2 { print!("{:02x}", b); }
    println!();
}
