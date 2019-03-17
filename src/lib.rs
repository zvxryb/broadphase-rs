// mlodato, 20190221

//! # Overview
//! 
//! broadphase-rs is a broadphase collision detection library.  It transforms object bounds into a lightweight
//! spatial index representation.  These indices are integer values which are sorted directly to yield a
//! result which is a topologically-sorted Morton order, after which full-system collision detection can be
//! accomplished by a single pass over the sorted list with only a minimal auxiliary stack necessary to maintain
//! state.  Collision tests between indices are accomplished with simple bitwise shifts, masks, and XORs.
//! 
//! This method is capable of supporting objects of varying scale (unlike uniform grids), while having a
//! straightforward, non-hierarchical structure in memory (unlike quad- or oct-trees), as the entire
//! representation exists in a single vector of index/object pairs.
//! 
//! # Usage
//! 
//! [`Layer`]: struct.Layer.html
//! 
//! [`Layer`] is the "main" struct of broadphase &mdash; this is where the sorted list of
//! spatial indices is stored.
//! 
//! The usual sequence of operations on a [`Layer`] is as follows:
//! 
//! ```rust
//! extern crate broadphase;
//! # extern crate cgmath;
//! 
//! use broadphase::{Bounds, Layer, LayerBuilder, Index64_3D};
//! type ID = u64;
//! 
//! # use cgmath::Point3;
//! # fn doc_main<Iter>(system_bounds: Bounds<Point3<f32>>, objects: Iter)
//! # where
//! #     Iter: Iterator<Item = (Bounds<Point3<f32>>, ID)>
//! # {
//! let mut layer: Layer<Index64_3D, ID> = LayerBuilder::new().build();
//! 
//! // clears existing object index-ID pairs:
//! layer.clear();
//! 
//! // appends an iterator of object bounds-ID pairs to the layer:
//! layer.extend(system_bounds, objects);
//! 
//! // scans the layer for collisions:
//! let potential_collisions = layer.scan();
//! # }
//! ```

extern crate cgmath;
extern crate num_traits;

#[cfg(feature="parallel")]
extern crate rayon;

#[cfg(feature="parallel")]
extern crate thread_local;

#[cfg(test)]
extern crate rand;

#[cfg(test)]
extern crate rand_chacha;

#[macro_use]
extern crate log;

#[macro_use]
extern crate smallvec;

mod traits;
mod geom;
mod index;
mod layer;

pub use geom::{Bounds, TestGeometry, RayTestGeometry};
pub use index::{SpatialIndex, Index64_3D};
pub use layer::{Layer, LayerBuilder};