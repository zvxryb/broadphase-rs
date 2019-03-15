use std::fmt::Debug;

pub trait MaxAxis<T> {
    fn max_axis(self) -> T;
}

/// A trait for casting between types
pub trait Cast<T> {
    fn cast(self) -> Option<T>;
}

/// Used to verify that an object's bounds are fully contained within the system bounds
pub trait Containment<RHS = Self> {
    fn contains(self, other: RHS) -> bool;
}

/// The result of a "quantization" (float to normalized int) operation
pub trait QuantizeResult {
    type Quantized;
}

impl QuantizeResult for f32 {
    type Quantized = u32;
}

impl QuantizeResult for f64 {
    type Quantized = u64;
}

/// Conversion from floating-point to normalized integer representation
pub trait Quantize : QuantizeResult {
    fn quantize(self) -> Option<Self::Quantized>;
}

#[cfg(not(feature="parallel"))]
pub trait ObjectID: Copy + Clone + Eq + Ord + Debug {}

#[cfg(not(feature="parallel"))]
impl<T: Copy + Clone + Eq + Ord + Debug> ObjectID for T {}

#[cfg(feature="parallel")]
pub trait ObjectID: Copy + Clone + Eq + Ord + Send + Sync + Debug {}

#[cfg(feature="parallel")]
impl<T: Copy + Clone + Eq + Ord + Send + Sync + Debug> ObjectID for T {}