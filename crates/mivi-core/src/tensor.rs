//! Minimalist contiguous tensor abstractions.

use thiserror::Error;

#[derive(Error, Debug)]
pub enum TensorError {
    #[error("Shape mismatch: expected {expected:?}, got {actual:?}")]
    ShapeMismatch {
        expected: Vec<usize>,
        actual: Vec<usize>,
    },
    #[error("Dimension out of bounds: dim {dim} >= rank {rank}")]
    DimOutOfBounds { dim: usize, rank: usize },
    #[error("Invalid element count: shape has {expected} elements, provided buffer has {actual}")]
    ElementCountMismatch { expected: usize, actual: usize },
}

pub type Result<T> = std::result::Result<T, TensorError>;

/// Shape descriptor for multi-dimensional tensors.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorShape {
    dims: Vec<usize>,
}

impl TensorShape {
    pub fn new(dims: Vec<usize>) -> Self {
        Self { dims }
    }

    #[inline]
    pub fn rank(&self) -> usize {
        self.dims.len()
    }

    #[inline]
    pub fn dims(&self) -> &[usize] {
        &self.dims
    }

    /// Returns the dimension size at the given axis index or error if out of bounds.
    #[inline]
    pub fn dim(&self, index: usize) -> Result<usize> {
        self.dims
            .get(index)
            .copied()
            .ok_or(TensorError::DimOutOfBounds {
                dim: index,
                rank: self.dims.len(),
            })
    }

    /// Try calculating total number of elements with checked overflow arithmetic.
    #[inline]
    pub fn try_num_elements(&self) -> Result<usize> {
        if self.dims.is_empty() {
            Ok(0)
        } else {
            self.dims
                .iter()
                .try_fold(1usize, |acc, &d| acc.checked_mul(d))
                .ok_or(TensorError::ElementCountMismatch {
                    expected: usize::MAX,
                    actual: 0,
                })
        }
    }

    #[inline]
    pub fn num_elements(&self) -> usize {
        self.try_num_elements()
            .expect("Tensor shape overflow: dimensions too large")
    }
}

/// Contiguous f32 tensor owned buffer.
#[derive(Debug, Clone)]
pub struct Tensor {
    shape: TensorShape,
    data: Vec<f32>,
}

impl Tensor {
    pub fn zeros(shape: TensorShape) -> Self {
        let size = shape.num_elements();
        Self {
            shape,
            data: vec![0.0; size],
        }
    }

    pub fn from_vec(shape: TensorShape, data: Vec<f32>) -> Result<Self> {
        let expected = shape.num_elements();
        if data.len() != expected {
            return Err(TensorError::ElementCountMismatch {
                expected,
                actual: data.len(),
            });
        }
        Ok(Self { shape, data })
    }

    #[inline]
    pub fn shape(&self) -> &TensorShape {
        &self.shape
    }

    #[inline]
    pub fn as_slice(&self) -> &[f32] {
        &self.data
    }

    #[inline]
    pub fn as_mut_slice(&mut self) -> &mut [f32] {
        &mut self.data
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tensor_shape_num_elements() {
        let shape = TensorShape::new(vec![2, 3, 4]);
        assert_eq!(shape.num_elements(), 24);
        assert_eq!(shape.rank(), 3);
        assert_eq!(shape.dims(), &[2, 3, 4]);

        let empty = TensorShape::new(vec![]);
        assert_eq!(empty.num_elements(), 0);
    }

    #[test]
    #[should_panic(expected = "Tensor shape overflow")]
    fn test_tensor_shape_overflow() {
        let shape = TensorShape::new(vec![usize::MAX, 2]);
        let _ = shape.num_elements();
    }
}
