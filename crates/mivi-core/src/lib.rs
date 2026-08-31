//! Core tensor primitives, zero-heap memory arena, SIMD dispatch, and math functions for mivi-v4.

pub mod arena;
pub mod brand;
pub mod math;
pub mod rope;
pub mod simd;
pub mod sys;

pub use arena::RunState;
pub use brand::{
    AGENT_RUN_ID_PREFIX, CHATCMPL_ID_PREFIX, DEFAULT_MODEL_ID, DEFAULT_SYSTEM_PROMPT, ENGINE_NAME,
    ENGINE_OWNER, ENV_API_KEY,
};
pub use math::{dot_product, rms_norm, silu, silu_scalar, softmax, swiglu, vec_add, vec_fmadd};
pub use rope::{RopeCache, RopeError};
pub use simd::{rms_norm_in_place_simd, rms_norm_simd};
pub use sys::{estimate_process_memory_mb, get_system_page_size};
