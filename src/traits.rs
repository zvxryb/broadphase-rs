// mlodato, 20190318

use std::fmt::Debug;
use std::hash::Hash;

#[cfg(not(feature="parallel"))]
pub trait ObjectID: Copy + Clone + Hash + Ord + Debug {}

#[cfg(not(feature="parallel"))]
impl<T: Copy + Clone + Hash + Ord + Debug> ObjectID for T {}

#[cfg(feature="parallel")]
pub trait ObjectID: Copy + Clone + Hash + Ord + Send + Sync + Debug {}

#[cfg(feature="parallel")]
impl<T: Copy + Clone + Hash + Ord + Send + Sync + Debug> ObjectID for T {}