//! BLAKE3 Merkle tree backends.
//!
//! Node definitions follow BLAKE3's own tree semantics, in the stable `hazmat` API:
//!
//! * a leaf is the non-root chaining value of the leaf bytes
//!   (`Hasher::new().update(bytes).finalize_non_root()`);
//! * a parent is the PARENT-flagged compression of its two children
//!   (`hazmat::merge_subtrees_non_root`), one compression and no chunk state.
//!
//! The PARENT flag domain-separates internal nodes from leaves. Whole levels and whole
//! leaf sets are hashed through `blake3::platform::hash_many`, which runs several
//! compressions at once with SIMD (NEON on aarch64, AVX2/AVX-512 on x86); the batched
//! paths are tested bit-for-bit against the scalar definitions above, which are also
//! what `Proof::verify` uses. Leaves can only be batched when their byte length is a
//! multiple of 64 and at most 1024 (one BLAKE3 chunk); other sizes fall back to the
//! scalar definition. This layout is the one used by the Flock prover
//! (`flock-core/src/merkle.rs`).

use alloc::vec::Vec;
use core::marker::PhantomData;
use std::sync::OnceLock;

use blake3::hazmat::{merge_subtrees_non_root, HasherExt, Mode};
use blake3::platform::Platform;
use blake3::IncrementCounter;
use lambdaworks_math::{
    field::{element::FieldElement, traits::IsField},
    traits::AsBytes,
};
#[cfg(feature = "parallel")]
use rayon::prelude::*;

use crate::merkle_tree::traits::IsMerkleTreeBackend;

pub type Node = [u8; 32];

/// BLAKE3's IV: the key words for unkeyed hashing, fixed by the specification.
const IV: [u32; 8] = [
    0x6A09E667, 0xBB67AE85, 0x3C6EF372, 0xA54FF53A, 0x510E527F, 0x9B05688C, 0x1F83D9AB, 0x5BE0CD19,
];
const CHUNK_START: u8 = 1;
const CHUNK_END: u8 = 2;
const PARENT: u8 = 4;

/// Messages per `hash_many` call: fills the widest SIMD path on every platform.
const BATCH: usize = 16;
/// Nodes handled per parallel task.
const PAR_CHUNK: usize = 1024;

fn platform() -> Platform {
    static PLATFORM: OnceLock<Platform> = OnceLock::new();
    *PLATFORM.get_or_init(Platform::detect)
}

/// The scalar leaf definition.
#[inline]
pub fn leaf_cv(bytes: &[u8]) -> Node {
    blake3::Hasher::new().update(bytes).finalize_non_root()
}

/// The scalar parent definition.
#[inline]
pub fn parent_cv(left: &Node, right: &Node) -> Node {
    merge_subtrees_non_root(left, right, Mode::Hash)
}

/// Hash `out.len()` messages of `N` bytes each (`data` is their concatenation) with the
/// given flags, `BATCH` at a time. `N` must be a multiple of 64.
fn hash_many_n<const N: usize>(data: &[u8], out: &mut [Node], flags: u8, start: u8, end: u8) {
    debug_assert_eq!(data.len(), out.len() * N);
    debug_assert_eq!(N % 64, 0);
    let plat = platform();
    for (outs, msgs) in out.chunks_mut(BATCH).zip(data.chunks(BATCH * N)) {
        let n = outs.len();
        let first: &[u8; N] = msgs[..N].try_into().expect("N bytes");
        let mut inputs: [&[u8; N]; BATCH] = [first; BATCH];
        for (i, slot) in inputs[..n].iter_mut().enumerate() {
            *slot = msgs[i * N..(i + 1) * N].try_into().expect("N bytes");
        }
        // `[u8; 32]` has no padding, so a slice of nodes is a contiguous byte slice.
        let out_bytes: &mut [u8] =
            unsafe { core::slice::from_raw_parts_mut(outs.as_mut_ptr() as *mut u8, n * 32) };
        plat.hash_many(
            &inputs[..n],
            &IV,
            0,
            IncrementCounter::No,
            flags,
            start,
            end,
            out_bytes,
        );
    }
}

/// Parents of consecutive pairs of `children` (even length), batched.
fn hash_parents(children: &[Node]) -> Vec<Node> {
    debug_assert_eq!(children.len() % 2, 0);
    let mut out = vec![[0u8; 32]; children.len() / 2];
    let work = |(outs, pairs): (&mut [Node], &[Node])| {
        let bytes: &[u8] =
            unsafe { core::slice::from_raw_parts(pairs.as_ptr() as *const u8, pairs.len() * 32) };
        hash_many_n::<64>(bytes, outs, PARENT, 0, 0);
    };
    #[cfg(feature = "parallel")]
    out.par_chunks_mut(PAR_CHUNK)
        .zip(children.par_chunks(2 * PAR_CHUNK))
        .for_each(work);
    #[cfg(not(feature = "parallel"))]
    out.chunks_mut(PAR_CHUNK)
        .zip(children.chunks(2 * PAR_CHUNK))
        .for_each(work);
    out
}

/// Chunk chaining values of `out.len()` leaves of `leaf_len` bytes each, batched when the
/// length allows it. Returns false (and leaves `out` untouched) otherwise.
fn hash_leaves_batched(data: &[u8], leaf_len: usize, out: &mut [Node]) -> bool {
    macro_rules! dispatch {
        ($($n:literal),+ $(,)?) => {
            match leaf_len {
                $($n => {
                    hash_many_n::<$n>(data, out, 0, CHUNK_START, CHUNK_END);
                    true
                })+
                _ => false,
            }
        };
    }
    dispatch!(64, 128, 192, 256, 320, 384, 448, 512, 576, 640, 704, 768, 832, 896, 960, 1024)
}

/// Serialise a leaf exactly as the scalar definition hashes it.
fn leaf_bytes<F: IsField>(leaf: &[FieldElement<F>]) -> Vec<u8>
where
    FieldElement<F>: AsBytes,
{
    let mut bytes = Vec::with_capacity(leaf.len() * 32);
    for element in leaf {
        bytes.extend_from_slice(&element.as_bytes());
    }
    bytes
}

fn hash_vector_leaves<F: IsField>(leaves: &[Vec<FieldElement<F>>]) -> Vec<Node>
where
    FieldElement<F>: AsBytes,
    Vec<FieldElement<F>>: Sync + Send,
{
    let mut out = vec![[0u8; 32]; leaves.len()];
    let work = |(outs, leaves): (&mut [Node], &[Vec<FieldElement<F>>])| {
        let leaf_len = leaves.first().map(|l| leaf_bytes(l).len()).unwrap_or(0);
        let uniform = leaves.iter().all(|l| l.len() == leaves[0].len());
        if uniform && leaf_len > 0 && leaf_len % 64 == 0 && leaf_len <= 1024 {
            let mut data = Vec::with_capacity(leaf_len * leaves.len());
            for leaf in leaves {
                data.extend_from_slice(&leaf_bytes(leaf));
            }
            if hash_leaves_batched(&data, leaf_len, outs) {
                return;
            }
        }
        for (o, leaf) in outs.iter_mut().zip(leaves) {
            *o = leaf_cv(&leaf_bytes(leaf));
        }
    };
    #[cfg(feature = "parallel")]
    out.par_chunks_mut(PAR_CHUNK)
        .zip(leaves.par_chunks(PAR_CHUNK))
        .for_each(work);
    #[cfg(not(feature = "parallel"))]
    out.chunks_mut(PAR_CHUNK)
        .zip(leaves.chunks(PAR_CHUNK))
        .for_each(work);
    out
}

/// Leaves are rows of field elements (trace and composition polynomial commitments).
#[derive(Clone)]
pub struct Blake3VectorBackend<F> {
    phantom: PhantomData<F>,
}

impl<F> Default for Blake3VectorBackend<F> {
    fn default() -> Self {
        Self {
            phantom: PhantomData,
        }
    }
}

impl<F> IsMerkleTreeBackend for Blake3VectorBackend<F>
where
    F: IsField,
    FieldElement<F>: AsBytes,
    Vec<FieldElement<F>>: Sync + Send,
{
    type Node = Node;
    type Data = Vec<FieldElement<F>>;

    fn hash_data(leaf: &Vec<FieldElement<F>>) -> Node {
        leaf_cv(&leaf_bytes(leaf))
    }

    fn hash_leaves(unhashed_leaves: &[Vec<FieldElement<F>>]) -> Vec<Node> {
        hash_vector_leaves(unhashed_leaves)
    }

    fn hash_new_parent(left: &Node, right: &Node) -> Node {
        parent_cv(left, right)
    }

    fn hash_level(children: &[Node]) -> Vec<Node> {
        hash_parents(children)
    }
}

/// Leaves are single field elements (FRI layer commitments). Leaves are 32 bytes, below
/// one block, so they use the scalar definition; levels are batched.
#[derive(Clone)]
pub struct Blake3ElementBackend<F> {
    phantom: PhantomData<F>,
}

impl<F> Default for Blake3ElementBackend<F> {
    fn default() -> Self {
        Self {
            phantom: PhantomData,
        }
    }
}

impl<F> IsMerkleTreeBackend for Blake3ElementBackend<F>
where
    F: IsField,
    FieldElement<F>: AsBytes + Sync + Send,
{
    type Node = Node;
    type Data = FieldElement<F>;

    fn hash_data(leaf: &FieldElement<F>) -> Node {
        leaf_cv(&leaf.as_bytes())
    }

    fn hash_new_parent(left: &Node, right: &Node) -> Node {
        parent_cv(left, right)
    }

    fn hash_level(children: &[Node]) -> Vec<Node> {
        hash_parents(children)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::merkle_tree::merkle::MerkleTree;
    use lambdaworks_math::field::fields::fft_friendly::stark_252_prime_field::Stark252PrimeField;

    type F = Stark252PrimeField;
    type FE = FieldElement<F>;

    fn pseudo_random_bytes(n: usize, seed: u64) -> Vec<u8> {
        let mut state = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
        (0..n)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                state as u8
            })
            .collect()
    }

    #[test]
    fn batched_parents_match_the_scalar_definition() {
        for count in [2usize, 4, 30, 2 * BATCH, 2 * PAR_CHUNK + 6, 5000] {
            let bytes = pseudo_random_bytes(count * 32, count as u64);
            let children: Vec<Node> = bytes.chunks(32).map(|c| c.try_into().unwrap()).collect();
            let batched = hash_parents(&children);
            for (i, pair) in children.chunks(2).enumerate() {
                assert_eq!(
                    batched[i],
                    parent_cv(&pair[0], &pair[1]),
                    "pair {i} of {count}"
                );
            }
        }
    }

    #[test]
    fn batched_leaves_match_the_scalar_definition() {
        for leaf_len in [64usize, 128, 384, 1024] {
            for count in [1usize, 3, BATCH + 1, PAR_CHUNK + 2] {
                let data = pseudo_random_bytes(leaf_len * count, (leaf_len * count) as u64);
                let mut out = vec![[0u8; 32]; count];
                assert!(hash_leaves_batched(&data, leaf_len, &mut out));
                for (i, leaf) in data.chunks(leaf_len).enumerate() {
                    assert_eq!(out[i], leaf_cv(leaf), "leaf {i}, len {leaf_len}");
                }
            }
        }
        // unsupported lengths are refused so callers fall back to the scalar path
        assert!(!hash_leaves_batched(&[0u8; 96], 96, &mut [[0u8; 32]; 1]));
    }

    #[test]
    fn vector_leaves_match_hash_data_for_every_row_width() {
        for width in [1usize, 2, 4, 12, 33] {
            let rows: Vec<Vec<FE>> = (0..(PAR_CHUNK + 5) as u64)
                .map(|r| (0..width as u64).map(|c| FE::from(r * 1000 + c)).collect())
                .collect();
            let hashed = Blake3VectorBackend::<F>::hash_leaves(&rows);
            for (i, row) in rows.iter().enumerate() {
                assert_eq!(
                    hashed[i],
                    Blake3VectorBackend::<F>::hash_data(row),
                    "row {i} width {width}"
                );
            }
        }
    }

    #[test]
    fn parents_and_leaves_are_domain_separated() {
        let a = [7u8; 32];
        let b = [9u8; 32];
        let mut concat = [0u8; 64];
        concat[..32].copy_from_slice(&a);
        concat[32..].copy_from_slice(&b);
        assert_ne!(parent_cv(&a, &b), leaf_cv(&concat));
        assert_ne!(parent_cv(&a, &b), *blake3::hash(&concat).as_bytes());
    }

    #[test]
    fn merkle_proofs_verify_with_both_backends() {
        let rows: Vec<Vec<FE>> = (0..1000u64)
            .map(|r| (0..12u64).map(|c| FE::from(r * 12 + c)).collect())
            .collect();
        let tree = MerkleTree::<Blake3VectorBackend<F>>::build(&rows).unwrap();
        for pos in [0usize, 1, 511, 999] {
            let proof = tree.get_proof_by_pos(pos).unwrap();
            assert!(proof.verify::<Blake3VectorBackend<F>>(&tree.root, pos, &rows[pos]));
            assert!(!proof.verify::<Blake3VectorBackend<F>>(
                &tree.root,
                pos,
                &rows[(pos + 1) % 1000]
            ));
        }
        let elements: Vec<FE> = (0..777u64).map(FE::from).collect();
        let tree = MerkleTree::<Blake3ElementBackend<F>>::build(&elements).unwrap();
        let proof = tree.get_proof_by_pos(500).unwrap();
        assert!(proof.verify::<Blake3ElementBackend<F>>(&tree.root, 500, &elements[500]));
        assert!(!proof.verify::<Blake3ElementBackend<F>>(&tree.root, 501, &elements[500]));
    }
}
