// mlodato, 20190318

use std::fmt::Debug;
use std::hash::Hash;

/// Conversion from floating-point to normalized integer representation
pub trait Quantize {
    type Quantized;

    fn quantize(self) -> Option<Self::Quantized>;
}
#[cfg(not(feature="parallel"))]
pub trait ObjectID: Copy + Clone + Default + Hash + Ord + Debug {}

#[cfg(not(feature="parallel"))]
impl<T: Copy + Clone + Default + Hash + Ord + Debug> ObjectID for T {}

#[cfg(feature="parallel")]
pub trait ObjectID: Copy + Clone + Default + Hash + Ord + Send + Sync + Debug {}

#[cfg(feature="parallel")]
impl<T: Copy + Clone + Default + Hash + Ord + Send + Sync + Debug> ObjectID for T {}