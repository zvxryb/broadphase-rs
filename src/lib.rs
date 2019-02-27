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
//! // clears all internal state:
//! layer.clear();
//! 
//! // appends an iterator of object bounds-ID pairs to the layer:
//! layer.extend(system_bounds, objects);
//! 
//! // scans the layer for collisions:
//! let potential_collisions = layer.detect_collisions();
//! # }
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
use num_traits::{NumAssignOps, NumCast, PrimInt, Unsigned};
use smallvec::SmallVec;

use std::collections::HashSet;
use std::fmt::Debug;
use std::ops::Shl;

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

type BuildHasherImpl = std::hash::BuildHasherDefault<Hasher>;

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
    T: NumCast,
    U: NumCast + Copy
{
    fn cast(self) -> Option<Point2<T>> {
        Point2::cast(&self)
    }
}

impl<T, U> Cast<Point3<T>> for Point3<U>
where
    T: NumCast,
    U: NumCast + Copy
{
    fn cast(self) -> Option<Point3<T>> {
        Point3::cast(&self)
    }
}

fn scale_at_depth<T>(depth: u32) -> T
where
    T: PrimInt + Unsigned + Shl<u32, Output = T> + Sized
{
    let total_bits = 8 * (std::mem::size_of::<T>() as u32);
    if depth == 0 {
        panic!("scale at zero depth would overflow integer");
    }
    T::one() << (total_bits - depth)
}

fn truncate_to_depth<T>(x: T, depth: u32) -> T
where
    T: PrimInt + Unsigned + Shl<u32, Output = T> + Sized
{
    if depth == 0 {
        x
    } else {
        x & !(scale_at_depth::<T>(depth) - T::one())
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
/// By default, index `depth` is chosen such that this returns no more than 4 (2D) or 8 (3D) indices.
/// `min_depth` provides a lower-bound which enables quick partitioning (for parallel task generation)

pub trait LevelIndexBounds<Index>
where
    Self::Output: IntoIterator<Item = Index>
{
    type Output;

    fn indices(self, min_depth: Option<u32>) -> Self::Output;
    fn indices_at_depth(self, depth: u32) -> Self::Output;
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
    Scalar::Quantized: PrimInt
{
    type Quantized = Point2<Scalar::Quantized>;
}

impl<Scalar> QuantizeResult for Point3<Scalar>
where 
    Scalar: cgmath::BaseFloat + QuantizeResult,
    Scalar::Quantized: PrimInt
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
    Scalar::Quantized: PrimInt + NumCast
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
    Scalar: NumAssignOps + PrimInt + Unsigned + Shl<u32, Output = Scalar> + Debug,
    Index: SpatialIndex<Point = Point3<Scalar>>
{
    type Output = SmallVec<[Index; 8]>;

    fn indices(self, min_depth: Option<u32>) -> Self::Output {
        let max_axis = self.size().max_axis();
        let mut depth = (max_axis - Scalar::one()).leading_zeros();
        if let Some(min_depth_) = min_depth {
            if depth < min_depth_ {
                depth = min_depth_;
            }
        }
        depth = Index::clamp_depth(depth);
        
        self.indices_at_depth(depth)
    }

    fn indices_at_depth(self, depth: u32) -> Self::Output {
        if depth == 0 {
            return smallvec![Index::default()
                .set_depth(0)
                .set_origin(Point3::new(
                    Scalar::zero(),
                    Scalar::zero(),
                    Scalar::zero()))];
        }

        let min = self.min.map(|scalar| truncate_to_depth(scalar, depth));
        let max = self.max.map(|scalar| truncate_to_depth(scalar, depth));

        let mut indices: Self::Output = Self::Output::new();

        let step = scale_at_depth::<Scalar>(depth);
        let mut z = min.z;
        loop {
            let mut y = min.y;
            loop {
                let mut x = min.x;
                loop {
                    indices.push(Index::default()
                        .set_depth(depth)
                        .set_origin(Point3::new(x, y, z)));

                    if x >= max.x {
                        break;
                    }
                    x += step;
                }

                if y >= max.y {
                    break;
                }
                y += step;
            }

            if z >= max.z {
                break;
            }
            z += step;
        }

        if indices.len() > 8 {
            warn!("indices_at_depth generated more than 8 indices; decrease min_depth or split large objects to avoid heap allocations");
        }

        indices
    }
}

/// [`SpatialIndex`]: trait.SpatialIndex.html
/// [`Index64_3D`]: trait.Index64_3D.html

/// A group of collision data
/// 
/// `Index` must be a type implmenting [`SpatialIndex`], such as [`Index64_3D`]
/// 
/// `ID` is the type representing object IDs

#[derive(Clone, Default)]
pub struct Layer<Index, ID>
where
    Index: SpatialIndex,
    ID: Ord + std::hash::Hash
{
    min_depth: u32,
    pub tree: (Vec<(Index, ID)>, bool),
    collisions: HashSet<(ID, ID), BuildHasherImpl>,
    invalid: Vec<ID>
}

impl<Index, ID> Layer<Index, ID>
where
    Index: SpatialIndex,
    ID: Copy + Eq + Ord + Send + std::hash::Hash + Debug,
    Index::Point: Copy + EuclideanSpace,
    <Index::Point as EuclideanSpace>::Diff: cgmath::VectorSpace,
    <Index::Point as EuclideanSpace>::Scalar: Debug + NumAssignOps + PrimInt,
    Bounds<Index::Point>: LevelIndexBounds<Index>
{
    /// Clear all internal state
    pub fn clear(&mut self) {
        let (tree, sorted) = &mut self.tree;
        tree.clear();
        *sorted = true;
        self.collisions.clear();
        self.invalid.clear();
    }

    /// Append multiple objects to the `Layer`
    /// 
    /// Complex geometry may provide multiple bounds for a single object ID; this usage would be common
    /// for static geometry, as it prevents extraneous self-collisions
    pub fn extend<Iter, Point_, Scalar_>(&mut self, system_bounds: Bounds<Point_>, objects: Iter)
    where
        Iter: std::iter::Iterator<Item = (Bounds<Point_>, ID)>,
        Point_: EuclideanSpace<Scalar = Scalar_>,
        Point_::Scalar: cgmath::BaseFloat,
        Point_::Diff: ElementWise,
        Scalar_: Debug + NumAssignOps,
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
                .indices(Some(self.min_depth))
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

        if other.min_depth < self.min_depth {
            warn!("merging layer of lesser min_depth (lhs: {}, rhs: {})", self.min_depth, other.min_depth);
            self.min_depth = other.min_depth;
        }

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
        -> &'a HashSet<(ID, ID), BuildHasherImpl>
    {
        self.detect_collisions_filtered(|_, _| true)
    }

    /// Detects collisions between all objects in the `Layer`, returning only those which pass a user-specified test
    /// 
    /// Collisions are filtered prior to duplicate removal.  This may be faster or slower than filtering
    /// post-duplicate-removal (i.e. by `detect_collisions().iter().filter()`) depending on the complexity
    /// of the filter.
    pub fn detect_collisions_filtered<'a, F>(&'a mut self, filter: F)
        -> &'a HashSet<(ID, ID), BuildHasherImpl>
    where
        F: FnMut(ID, ID) -> bool
    {
        self.sort();

        let (tree, _) = &self.tree;
        Self::detect_collisions_impl(tree.as_slice(), &mut self.collisions, filter);

        &self.collisions
    }

    fn detect_collisions_impl<F>(tree: &[(Index, ID)], collisions: &mut HashSet<(ID, ID), BuildHasherImpl>, mut filter: F)
    where
        F: FnMut(ID, ID) -> bool
    {
        let mut stack: SmallVec<[(Index, ID); 32]> = SmallVec::new();
        for &(index, id) in tree {
            while let Some(&(index_, _)) = stack.last() {
                if index.overlaps(index_) {
                    break;
                }
                stack.pop();
            }
            if stack.iter().any(|&(_, id_)| id == id_) {
                continue;
            }
            for &(_, id_) in &stack {
                if id != id_ && filter(id, id_) {
                    collisions.insert((id, id_));
                }
            }
            stack.push((index, id))
        }
    }
}

/// A builder for `Layer`s
pub struct LayerBuilder {
    min_depth: u32,
    index_capacity: Option<usize>,
    collision_capacity: Option<usize>
}

impl LayerBuilder {
    pub fn new() -> Self {
        Self {
            min_depth: 0,
            index_capacity: None,
            collision_capacity: None
        }
    }

    pub fn with_min_depth(&mut self, depth: u32) -> &mut Self {
        self.min_depth = depth;
        self
    }

    pub fn with_index_capacity(&mut self, capacity: usize) -> &mut Self {
        self.index_capacity = Some(capacity);
        self
    }

    pub fn with_collision_capacity(&mut self, capacity: usize) -> &mut Self {
        self.collision_capacity = Some(capacity);
        self
    }

    pub fn build<Index, ID>(&self) -> Layer<Index, ID>
    where
        Index: SpatialIndex,
        ID: Copy + Eq + Ord + Send + std::hash::Hash + Debug,
        Index::Point: Copy + EuclideanSpace,
        <Index::Point as EuclideanSpace>::Diff: cgmath::VectorSpace,
        <Index::Point as EuclideanSpace>::Scalar: Debug + NumAssignOps + PrimInt,
        Bounds<Index::Point>: LevelIndexBounds<Index>
    {
        let hasher = BuildHasherImpl::default();
        Layer::<Index, ID>{
            min_depth: self.min_depth,
            tree: (match self.index_capacity {
                    Some(capacity) => Vec::with_capacity(capacity),
                    None => Vec::new()
                }, true),
            collisions: match self.collision_capacity {
                    Some(capacity) => HashSet::with_capacity_and_hasher(capacity, hasher),
                    None => HashSet::with_hasher(hasher)
                },
            invalid: Vec::new()
        }
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
