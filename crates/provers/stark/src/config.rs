use lambdaworks_crypto::merkle_tree::merkle::MerkleTree;

#[cfg(feature = "blake3")]
use lambdaworks_crypto::merkle_tree::backends::blake3::{
    Blake3ElementBackend, Blake3VectorBackend,
};
#[cfg(not(feature = "blake3"))]
use lambdaworks_crypto::merkle_tree::backends::types::{BatchKeccak256Backend, Keccak256Backend};

// Merkle tree backends for FRI layers (single field element per leaf) and for the
// trace / composition polynomial (one row of field elements per leaf). Keccak-256 by
// default for Stone compatibility; the `blake3` feature switches both to BLAKE3 with
// batched SIMD hashing, which is several times faster on the prover.
#[cfg(not(feature = "blake3"))]
pub type FriMerkleTreeBackend<F> = Keccak256Backend<F>;
#[cfg(feature = "blake3")]
pub type FriMerkleTreeBackend<F> = Blake3ElementBackend<F>;
pub type FriMerkleTree<F> = MerkleTree<FriMerkleTreeBackend<F>>;

pub const COMMITMENT_SIZE: usize = 32;
pub type Commitment = [u8; COMMITMENT_SIZE];

#[cfg(not(feature = "blake3"))]
pub type BatchedMerkleTreeBackend<F> = BatchKeccak256Backend<F>;
#[cfg(feature = "blake3")]
pub type BatchedMerkleTreeBackend<F> = Blake3VectorBackend<F>;
pub type BatchedMerkleTree<F> = MerkleTree<BatchedMerkleTreeBackend<F>>;
