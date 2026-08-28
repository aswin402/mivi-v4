//! Core tensor primitives, zero-heap memory arena, SIMD dispatch, and math functions for mivi-v4.

pub mod arena;
pub mod math;
pub mod rope;
pub mod simd;
pub mod tensor;

pub use arena::RunState;
pub use rope::RopeCache;
pub use tensor::{Tensor, TensorShape};
