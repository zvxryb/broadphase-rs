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
//! 
//! use broadphase::{Bounds, Layer, Index64_3D};
//! type ID = u64;
//! 
//! // ...
//! 
//! let mut layer: Layer<Index64_3D, ID> = Layer::new();
//! 
//! // ...
//! 
//! // clears all internal state:
//! layer.clear();
//! 
//! // appends an iterator of object bounds-ID pairs to the layer:
//! layer.extend(system_bounds, objects);
//! 
//! // scans the layer for collisions:
//! let potential_collisions = layer.detect_collisions();
//! ```

extern crate cgmath;
extern crate num_traits;

#[cfg(feature="fnv")]
extern crate fnv;

#[cfg(feature="rustc-hash")]
extern crate rustc_hash;

#[cfg(feature="rayon")]
extern crate rayon;

#[cfg(test)]
extern crate rand;

#[cfg(test)]
extern crate rand_chacha;

#[macro_use]
extern crate log;

#[macro_use]
extern crate smallvec;

use cgmath::prelude::*;

use cgmath::{Point2, Point3, Vector2, Vector3};
use smallvec::SmallVec;

use std::collections::HashSet;

#[cfg(feature="rayon")]
use rayon::prelude::*;

#[cfg(feature="fnv")]
use fnv::FnvHasher as Hasher;

#[cfg(feature="rustc-hash")]
use rustc_hash::FxHasher as Hasher;

#[cfg(not(any(feature="fnv", feature="rustc-hash")))]
use std::collections::hash_map::DefaultHasher as Hasher;

mod index;
pub use index::{SpatialIndex, Index64_3D};

type BuildHasher = std::hash::BuildHasherDefault<Hasher>;

trait MaxAxis<T> {
    fn max_axis(self) -> T;
}

impl<T> MaxAxis<T> for Vector2<T>
where
    T: Ord
{
    fn max_axis(self) -> T {
        std::cmp::max(self.x, self.y)
    }
}

impl<T> MaxAxis<T> for Vector3<T>
where
    T: Ord
{
    fn max_axis(self) -> T {
        use std::cmp::max;
        max(max(self.x, self.y), self.z)
    }
}

/// A trait for casting between types
pub trait Cast<T> {
    fn cast(self) -> Option<T>;
}

impl<T, U> Cast<Point2<T>> for Point2<U>
where
    T: num_traits::NumCast,
    U: num_traits::NumCast + Copy
{
    fn cast(self) -> Option<Point2<T>> {
        Point2::cast(&self)
    }
}

impl<T, U> Cast<Point3<T>> for Point3<U>
where
    T: num_traits::NumCast,
    U: num_traits::NumCast + Copy
{
    fn cast(self) -> Option<Point3<T>> {
        Point3::cast(&self)
    }
}

/// A trait for truncating values to a specified level of precision
pub trait Truncate {
    fn truncate(self, bits: u32) -> Self;
}

impl<T> Truncate for T
where
    T: num_traits::PrimInt + num_traits::Unsigned + std::ops::Shl<u32, Output = Self> + Sized
{
    fn truncate(self, bits: u32) -> Self {
        let total_bits = 8 * (std::mem::size_of::<T>() as u32);
        if bits == 0 {
            self
        } else {
            self & !((Self::one() << (total_bits - bits)) - Self::one())
        }
    }
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

/// The set of indices for an object with known bounds
/// 
/// Index `depth` is chosen such that this returns no more than 4 (2D) or 8 (3D) indices

pub trait LevelIndexBounds<Index>
where
    Self::Output: IntoIterator<Item = Index>
{
    type Output;
    fn indices(self) -> Self::Output;
}

/// An axis-aligned bounding box
/// 
/// This is used in public interfaces, as a means to obtain information necessary to generate indices.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Bounds<Point> {
    pub min: Point,
    pub max: Point
}

impl<Point> Bounds<Point>
where
    Point: EuclideanSpace + Copy
{
    pub fn new(min: Point, max: Point) -> Self {
        Self{min: min, max: max}
    }

    pub fn size(self) -> Point::Diff {
        self.max - self.min
    }

    fn normalize_point(self, point: Point) -> Point
    where
        Point::Scalar: cgmath::BaseFloat,
        Point::Diff: ElementWise
    {
        EuclideanSpace::from_vec((point - self.min).div_element_wise(self.size()))
    }

    fn normalize_to_system(self, system_bounds: Bounds<Point>) -> Bounds<Point>
    where
        Point::Scalar: cgmath::BaseFloat,
        Point::Diff: ElementWise
    {
        Bounds{
            min: system_bounds.normalize_point(self.min),
            max: system_bounds.normalize_point(self.max)
        }
    }
}

impl<T> Containment for Bounds<Point2<T>>
where
    T: cgmath::BaseNum
{
    fn contains(self, other: Bounds<Point2<T>>) -> bool {
        self.min.x <= other.min.x &&
        self.min.y <= other.min.y &&
        self.max.x >  other.max.x &&
        self.max.y >  other.max.y
    }
}

impl<T> Containment for Bounds<Point3<T>>
where
    T: cgmath::BaseNum
{
    fn contains(self, other: Bounds<Point3<T>>) -> bool {
        self.min.x <= other.min.x &&
        self.min.y <= other.min.y &&
        self.min.z <= other.min.z &&
        self.max.x >  other.max.x &&
        self.max.y >  other.max.y &&
        self.max.z >  other.max.z
    }
}

impl<Scalar> QuantizeResult for Point2<Scalar>
where 
    Scalar: cgmath::BaseFloat + QuantizeResult,
    Scalar::Quantized: num_traits::int::PrimInt
{
    type Quantized = Point2<Scalar::Quantized>;
}

impl<Scalar> QuantizeResult for Point3<Scalar>
where 
    Scalar: cgmath::BaseFloat + QuantizeResult,
    Scalar::Quantized: num_traits::int::PrimInt
{
    type Quantized = Point3<Scalar::Quantized>;
}

impl<Point> QuantizeResult for Bounds<Point>
where 
    Point: QuantizeResult,
{
    type Quantized = Bounds<Point::Quantized>;
}

impl<Point, Scalar> Quantize for Bounds<Point>
where
    Point: cgmath::EuclideanSpace<Scalar = Scalar> + cgmath::ElementWise<Scalar> + QuantizeResult + Cast<<Point as QuantizeResult>::Quantized>,
    <Point as QuantizeResult>::Quantized: ElementWise<Scalar::Quantized>,
    Scalar: cgmath::BaseFloat + QuantizeResult,
    Scalar::Quantized: num_traits::int::PrimInt + num_traits::NumCast
{
    fn quantize(self) -> Option<Self::Quantized> {
        let min_value = Scalar::from(Scalar::Quantized::min_value())?;
        let max_value = Scalar::from(Scalar::Quantized::max_value() - Scalar::Quantized::one())?;
        let range = max_value - min_value;
        Some(Bounds{
            min: (self.min * range).add_element_wise(min_value).cast()?,
            max: (self.max * range).add_element_wise(min_value).cast()?.add_element_wise(Scalar::Quantized::one())
        })
    }
}

impl<Scalar, Index> LevelIndexBounds<Index> for Bounds<Point3<Scalar>>
where
    Scalar: num_traits::NumAssign + num_traits::PrimInt + Truncate + std::fmt::Debug,
    Index: SpatialIndex<Point = Point3<Scalar>>
{
    type Output = SmallVec<[Index; 8]>;
    fn indices(self) -> Self::Output {
        let max_axis = self.size().max_axis();
        let depth = Index::clamp_depth((max_axis - Scalar::one()).leading_zeros());
        let min = self.min.map(|scalar| scalar.truncate(depth));
        let max = self.max.map(|scalar| scalar.truncate(depth));

        let mask =
             ((min.x != max.x) as u32)       |
            (((min.y != max.y) as u32) << 1) |
            (((min.z != max.z) as u32) << 2);

        let mut level_bounds: SmallVec<[Point3<Scalar>; 8]> = SmallVec::new();
        if (mask & 0b000) == 0 { level_bounds.push(Point3::new(min.x, min.y, min.z)); }
        if (mask & 0b001) == 1 { level_bounds.push(Point3::new(max.x, min.y, min.z)); }
        if (mask & 0b010) == 1 { level_bounds.push(Point3::new(min.x, max.y, min.z)); }
        if (mask & 0b011) == 2 { level_bounds.push(Point3::new(max.x, max.y, min.z)); }
        if (mask & 0b100) == 1 { level_bounds.push(Point3::new(min.x, min.y, max.z)); }
        if (mask & 0b101) == 2 { level_bounds.push(Point3::new(max.x, min.y, max.z)); }
        if (mask & 0b110) == 2 { level_bounds.push(Point3::new(min.x, max.y, max.z)); }
        if (mask & 0b111) == 3 { level_bounds.push(Point3::new(max.x, max.y, max.z)); }
        level_bounds.into_iter()
            .map(|origin| Index::default().set_depth(depth).set_origin(origin))
            .collect()
    }
}

/// [`SpatialIndex`]: trait.SpatialIndex.html
/// [`Index64_3D`]: trait.Index64_3D.html

/// A group of collision data
/// 
/// `Index` be a type implmenting [`SpatialIndex`], such as [`Index64_3D`]
/// `ID` is the type representing object IDs

#[derive(Clone, Default)]
pub struct Layer<Index, ID>
where
    Index: SpatialIndex,
    ID: Ord + std::hash::Hash
{
    pub tree: (Vec<(Index, ID)>, bool),
    collisions: HashSet<(ID, ID), BuildHasher>,
    invalid: Vec<ID>
}

impl<Index, ID> Layer<Index, ID>
where
    Index: SpatialIndex,
    ID: Copy + Eq + Ord + Send + std::hash::Hash + std::fmt::Debug,
    Index::Point: Copy + EuclideanSpace,
    <Index::Point as EuclideanSpace>::Diff: cgmath::VectorSpace,
    <Index::Point as EuclideanSpace>::Scalar: std::fmt::Debug + num_traits::int::PrimInt + num_traits::NumAssignOps,
    Bounds<Index::Point>: LevelIndexBounds<Index>
{
    /// Instantiate an empty `Layer`
    pub fn new() -> Self {
        Self {
            tree: (Vec::new(), true),
            collisions: HashSet::default(),
            invalid: Vec::new()
        }
    }

    /// Clear all internal state
    pub fn clear(&mut self) {
        let (tree, sorted) = &mut self.tree;
        tree.clear();
        *sorted = true;
        self.collisions.clear();
        self.invalid.clear();
    }

    /// Append multiple objects to the `Layer`
    pub fn extend<Iter, Point_, Scalar_>(&mut self, system_bounds: Bounds<Point_>, objects: Iter)
    where
        Iter: std::iter::Iterator<Item = (Bounds<Point_>, ID)>,
        Point_: EuclideanSpace<Scalar = Scalar_>,
        Point_::Scalar: cgmath::BaseFloat,
        Point_::Diff: ElementWise,
        Scalar_: std::fmt::Debug + num_traits::NumAssignOps,
        Bounds<Point_>: Containment + Quantize + QuantizeResult<Quantized = Bounds<Index::Point>>
    {
        let (tree, sorted) = &mut self.tree;

        if let (_, Some(max_objects)) = objects.size_hint() {
            tree.reserve(max_objects);
        }

        for (bounds, id) in objects {
            if !system_bounds.contains(bounds) {
                self.invalid.push(id);
                continue
            }
            

            tree.extend(bounds
                .normalize_to_system(system_bounds)
                .quantize()
                .expect("failed to filter bounds outside system")
                .indices()
                .into_iter()
                .map(|index| (index, id)));

            *sorted = false;
        }
    }

    /// Merge another `Layer` into this `Layer`
    /// 
    /// This may be used, for example, to merge static scene `Layer` into the current
    /// frames' dynamic `Layer` without having to recalculate indices for the static data
    pub fn merge(&mut self, other: &Layer<Index, ID>) {
        let (lhs_tree, sorted) = &mut self.tree;
        let (rhs_tree, _) = &other.tree;

        lhs_tree.extend(rhs_tree.iter());
        *sorted = false;
        return;
    }

    #[cfg(feature="rayon")]
    fn sort_impl<T: Ord + Send, Slice: AsMut<[T]>>(items: &mut Slice) {
        items.as_mut().par_sort_unstable();
    }

    #[cfg(not(feature="rayon"))]
    fn sort_impl<T: Ord, Slice: AsMut<[T]>>(items: &mut Slice) {
        items.as_mut().sort_unstable();
    }

    fn sort(&mut self) {
        let (tree, sorted) = &mut self.tree;
        if !*sorted {
            Self::sort_impl(tree);
            *sorted = true;
        }
    }

    /// Detects collisions between all objects in the `Layer`
    pub fn detect_collisions<'a>(&'a mut self)
        -> &'a HashSet<(ID, ID), BuildHasher>
    {
        self.detect_collisions_filtered(|_, _| true)
    }

    /// Detects collisions between all objects in the `Layer`, returning only those which pass a user-specified test
    /// 
    /// Collisions are filtered prior to duplicate removal.  This may be faster or slower than filtering
    /// post-duplicate-removal (i.e. by `detect_collisions().iter().filter()`) depending on the complexity
    /// of the filter.
    pub fn detect_collisions_filtered<'a, F>(&'a mut self, mut filter: F)
        -> &'a HashSet<(ID, ID), BuildHasher>
    where
        F: FnMut(ID, ID) -> bool
    {
        self.sort();

        let mut stack: SmallVec<[(Index, ID); 32]> = SmallVec::new();
        let (tree, _) = &self.tree;
        for &(index, id) in tree {
            while let Some(&(index_, _)) = stack.last() {
                if index.overlaps(index_) {
                    break;
                }
                stack.pop();
            }
            for &(_, id_) in &stack {
                if id == id_ {
                    if log_enabled!(log::Level::Debug) {
                        debug!("duplicate index for entity {:?}", id);
                    }
                } else if filter(id, id_) {
                    self.collisions.insert((id, id_));
                }
            }
            stack.push((index, id))
        }

        &self.collisions
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalize_to_system() {
        let system_bounds = Bounds{
            min: Point3::new(-64f32, -64f32, -64f32),
            max: Point3::new( 64f32,  64f32,  64f32)};
        let bounds = Bounds{
            min: Point3::new(-32f32, -32f32, -32f32),
            max: Point3::new( 32f32,  32f32,  32f32)};
        let expected = Bounds{
            min: Point3::new(0.25f32, 0.25f32, 0.25f32),
            max: Point3::new(0.75f32, 0.75f32, 0.75f32)};
        assert_eq!(bounds.normalize_to_system(system_bounds), expected);
    }
}
