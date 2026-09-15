/// BLAKE3 backends with batched (SIMD) leaf and level hashing
#[cfg(feature = "blake3")]
pub mod blake3;
pub mod field_element;
pub mod field_element_vector;
pub mod poseidon2;
/// Configurations for merkle trees
/// Setting generics to some value
pub mod types;

/// Metal GPU-accelerated Merkle tree backend
#[cfg(feature = "metal")]
pub mod metal;
