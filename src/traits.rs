// mlodato, 20190317

use std::fmt::Debug;

pub trait MaxAxis<T> {
    fn max_axis(self) -> T;
}

/// Used to verify that an object's bounds are fully contained within the system bounds
pub trait Containment<RHS = Self> {
    fn contains(self, other: RHS) -> bool;
}

/// Conversion from floating-point to normalized integer representation
pub trait Quantize {
    type Quantized;

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