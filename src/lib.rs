// mlodato, 20190219

extern crate cgmath;
extern crate num_traits;

#[macro_use]
extern crate log;

#[macro_use]
extern crate smallvec;

use cgmath::prelude::*;

use cgmath::{Point2, Point3, Vector2, Vector3};
use smallvec::SmallVec;

pub trait MaxAxis<T> {
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

pub trait Truncate {
    fn truncate(self, bits: u32) -> Self;
}

impl Truncate for u32 {
    fn truncate(self, bits: u32) -> Self {
        if bits == 0 {
            self
        } else {
            self & !((1u32 << (32 - bits)) - 1u32)
        }
    }
}

pub trait Containment<RHS = Self> {
    fn contains(self, other: RHS) -> bool;
}

pub trait QuantizeResult {
    type Quantized;
}

impl QuantizeResult for f32 {
    type Quantized = u32;
}

impl QuantizeResult for f64 {
    type Quantized = u64;
}

pub trait Quantize : QuantizeResult {
    fn quantize(self) -> Option<Self::Quantized>;
}

pub trait LevelIndexBounds<Index>
where
    Self::Output: IntoIterator<Item = Index>
{
    type Output;
    fn indices(self) -> Self::Output;
}

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

impl<T> Containment for Bounds<cgmath::Point2<T>>
where
    T: cgmath::BaseNum
{
    fn contains(self, other: Bounds<cgmath::Point2<T>>) -> bool {
        self.min.x <= other.min.x &&
        self.min.y <= other.min.y &&
        self.max.x >  other.max.x &&
        self.max.y >  other.max.y
    }
}

impl<T> Containment for Bounds<cgmath::Point3<T>>
where
    T: cgmath::BaseNum
{
    fn contains(self, other: Bounds<cgmath::Point3<T>>) -> bool {
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

/// Two properties are required to ensure that sorting indices produces a topological ordering:
///
/// 1. `self.origin()` should represent the lower bound for the node at `self.depth()`
/// 2. `self.depth()` should be less significant than origin for ordering
///
/// The underlying primitive type can be sorted directly if high-bits store origin and low-bits store depth

pub trait SpatialIndex<Point>: Clone + Copy + Default + Ord + std::fmt::Debug
where
    Point: Copy
{
    fn clamp_depth(u32) -> u32;
    fn origin(self) -> Point;
    fn depth(self) -> u32;
    fn set_origin(self, Point) -> Self;
    fn set_depth(self, u32) -> Self;
    fn overlaps(self, other: Self) -> bool;
}

#[derive(Clone, Copy, Debug, Default, Eq, Ord, PartialEq, PartialOrd)]
pub struct Index64_3D(u64);

impl Index64_3D {
    const DEPTH_BITS: u32 = 5;
    const DEPTH_SHIFT: u32 = 0;
    const DEPTH_MASK: u64 = ((1u64 << Self::DEPTH_BITS) - 1) << Self::DEPTH_SHIFT;
    const AXIS_BITS: u32 = 19;
    const ORIGIN_BITS: u32 = 3 * Self::AXIS_BITS;
    const ORIGIN_SHIFT: u32 = Self::DEPTH_SHIFT + Self::DEPTH_BITS;
    const ORIGIN_MASK: u64 = ((1u64 << (Self::ORIGIN_BITS)) - 1) << Self::ORIGIN_SHIFT;

    #[inline]
    fn decode_axis(origin: u64) -> u32 {
        let axis00 =  origin & 0o1_001_001_001_001_001_001_001;
        let axis01 = (origin & 0o0_010_010_010_010_010_010_010) >> 0o02;
        let axis02 = (origin & 0o0_100_100_100_100_100_100_100) >> 0o04;
        let axis0_ = axis00 | axis01 | axis02;
        let axis10 =  axis0_ & 0o0_007_000_000_007_000_000_007;
        let axis11 = (axis0_ & 0o1_000_000_007_000_000_007_000) >> 0o06;
        let axis12 = (axis0_ & 0o0_000_007_000_000_007_000_000) >> 0o14;
        let axis1_ = axis10 | axis11 | axis12;
        let axis20 =  axis1_ & 0o0_000_000_000_000_000_000_777;
        let axis21 = (axis1_ & 0o0_000_000_000_777_000_000_000) >> 0o22;
        let axis22 = (axis1_ & 0o0_777_000_000_000_000_000_000) >> 0o44;
        let axis2_ = axis20 | axis21 | axis22;
        (axis2_ as u32) << (32 - Self::AXIS_BITS)
    }

    #[inline]
    fn encode_axis(origin: u32) -> u64 {
        let axis0_ = u64::from(origin >> (32 - Self::AXIS_BITS));
        let axis00 =  axis0_          & 0o0_000_000_000_000_000_000_777;
        let axis01 = (axis0_ << 0o22) & 0o0_000_000_000_777_000_000_000;
        let axis02 = (axis0_ << 0o44) & 0o0_777_000_000_000_000_000_000;
        let axis1_ = axis00 | axis01 | axis02;
        let axis10 =  axis1_          & 0o0_007_000_000_007_000_000_007;
        let axis11 = (axis1_ << 0o06) & 0o1_000_000_007_000_000_007_000;
        let axis12 = (axis1_ << 0o14) & 0o0_000_007_000_000_007_000_000;
        let axis2_ = axis10 | axis11 | axis12;
        let axis20 =  axis2_          & 0o1_001_001_001_001_001_001_001;
        let axis21 = (axis2_ << 0o02) & 0o0_010_010_010_010_010_010_010;
        let axis22 = (axis2_ << 0o04) & 0o0_100_100_100_100_100_100_100;
        axis20 | axis21 | axis22
    }

    fn level_mask(depth: u32) -> u64 {
        !((1u64 << (Self::ORIGIN_BITS + Self::ORIGIN_SHIFT - 3 * depth)) - 1) & Self::ORIGIN_MASK
    }
}

impl SpatialIndex<Point3<u32>> for Index64_3D {
    fn clamp_depth(depth: u32) -> u32 {
        std::cmp::min(depth, Self::ORIGIN_BITS)
    }

    fn origin(self) -> Point3<u32> {
        let Self(index) = self;
        let origin = (index & Self::ORIGIN_MASK) >> Self::ORIGIN_SHIFT;
        Point3::new(
            Self::decode_axis(origin),
            Self::decode_axis(origin >> 1),
            Self::decode_axis(origin >> 2),
        )
    }

    fn depth(self) -> u32 {
        let Self(index) = self;
        ((index & Self::DEPTH_MASK) >> Self::DEPTH_SHIFT) as u32
    }

    fn set_origin(self, origin: Point3<u32>) -> Self {
        let origin = Self::encode_axis(origin.x)
                   | Self::encode_axis(origin.y) << 1
                   | Self::encode_axis(origin.z) << 2;
        let Self(mut index) = self;
        index &= !Self::ORIGIN_MASK;
        index |= Self::ORIGIN_MASK & (origin << Self::ORIGIN_SHIFT);
        Self(index)
    }

    fn set_depth(self, depth: u32) -> Self {
        let Self(mut index) = self;
        index &= !Self::DEPTH_MASK;
        index |= Self::DEPTH_MASK & (u64::from(Self::clamp_depth(depth)) << Self::DEPTH_SHIFT);
        Self(index)
    }

    fn overlaps(self, other: Self) -> bool {
        let Self(lhs) = self;
        let Self(rhs) = other;
        (lhs ^ rhs) & Self::level_mask(std::cmp::min(self.depth(), other.depth())) == 0
    }
}

impl<Scalar, Index> LevelIndexBounds<Index> for Bounds<Point3<Scalar>>
where
    Scalar: num_traits::NumAssign + num_traits::PrimInt + Truncate + std::fmt::Debug,
    Index: SpatialIndex<Point3<Scalar>>
{
    type Output = SmallVec<[Index; 8]>;
    fn indices(self) -> Self::Output {
        let max_axis = self.size().max_axis();
        let depth = Index::clamp_depth((max_axis - Scalar::one()).leading_zeros());
        let min = self.min.map(|scalar| scalar.truncate(depth));
        let max = self.max.map(|scalar| scalar.truncate(depth));

        let mut level_bounds: SmallVec<[Point3<Scalar>; 8]> = smallvec![
            Point3::new(min.x, min.y, min.z),
            Point3::new(max.x, min.y, min.z),
            Point3::new(min.x, max.y, min.z),
            Point3::new(max.x, max.y, min.z),
            Point3::new(min.x, min.y, max.z),
            Point3::new(max.x, min.y, max.z),
            Point3::new(min.x, max.y, max.z),
            Point3::new(max.x, max.y, max.z)];
        level_bounds.dedup();
        level_bounds.into_iter()
            .map(|origin| Index::default().set_depth(depth).set_origin(origin))
            .collect()
    }
}

#[derive(Clone, Default)]
pub struct Layer<Index, ID, Point>
where
    Index: SpatialIndex<Point>,
    Point: Copy
{
    phantom: std::marker::PhantomData<Point>,
    pub tree: Vec<(Index, ID)>,
    collisions: Vec<(ID, ID)>,
    invalid: Vec<ID>
}

impl<Index, ID, Point> Layer<Index, ID, Point>
where
    Index: SpatialIndex<Point>,
    ID: Copy + Ord + std::fmt::Debug,
    Point: Copy + EuclideanSpace,
    Point::Diff: cgmath::VectorSpace + MaxAxis<Point::Scalar>,
    Point::Scalar: std::fmt::Debug + num_traits::int::PrimInt + num_traits::NumAssignOps,
    Bounds<Point>: LevelIndexBounds<Index>
{
    pub fn new() -> Self {
        Self {
            phantom: std::marker::PhantomData{},
            tree: Vec::new(),
            collisions: Vec::new(),
            invalid: Vec::new()
        }
    }

    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            phantom: std::marker::PhantomData{},
            tree: Vec::with_capacity(capacity),
            collisions: Vec::new(),
            invalid: Vec::new()
        }
    }

    pub fn clear(&mut self) {
        self.tree.clear();
        self.collisions.clear();
        self.invalid.clear();
    }

    pub fn extend<Iter, Point_, Scalar_>(&mut self, system_bounds: Bounds<Point_>, objects: Iter)
    where
        Iter: std::iter::Iterator<Item = (Bounds<Point_>, ID)>,
        Point_: EuclideanSpace<Scalar = Scalar_>,
        Point_::Scalar: cgmath::BaseFloat,
        Point_::Diff: ElementWise,
        Scalar_: std::fmt::Debug + num_traits::NumAssignOps,
        Bounds<Point_>: Containment + Quantize + QuantizeResult<Quantized = Bounds<Point>>
    {
        if let (_, Some(max_objects)) = objects.size_hint() {
            self.tree.reserve(max_objects);
        }

        for (bounds, id) in objects {
            if !system_bounds.contains(bounds) {
                self.invalid.push(id);
                continue
            }

            self.tree.extend(bounds
                .normalize_to_system(system_bounds)
                .quantize()
                .expect("failed to filter bounds outside system")
                .indices()
                .into_iter()
                .map(|index| (index, id)));
        }
    }

    pub fn sort(&mut self) {
        self.tree.sort_unstable();
    }

    pub fn detect_collisions<'a>(&'a mut self) -> &'a Vec<(ID, ID)> {
        let mut stack: SmallVec<[(Index, ID); 32]> = SmallVec::new();
        for &(index, id) in self.tree.iter() {
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
                } else {
                    self.collisions.push((id, id_));
                }
            }
            stack.push((index, id))
        }
        self.collisions.sort_unstable();
        self.collisions.dedup();
        &self.collisions
    }
}

#[cfg(test)]
mod tests {
    extern crate rand;
    extern crate rand_chacha;

    use super::*;
    use tests::rand::prelude::*;

    #[test]
    fn decode() {
        assert_eq!(
            Index64_3D::decode_axis(0o0_001_111_111_111_111_111_111),
            0o1_777_777u32 << 13
        );
        assert_eq!(
            Index64_3D::decode_axis(0o0_006_666_666_666_666_666_666),
            0o0_000_000u32 << 13
        );
    }

    #[test]
    fn encode() {
        assert_eq!(
            Index64_3D::encode_axis(0o1_777_777u32 << 13),
            0o0_001_111_111_111_111_111_111
        );
        assert_eq!(
            Index64_3D::encode_axis(0o0_000_000u32 << 13),
            0o0_000_000_000_000_000_000_000
        );
    }

    #[test]
    fn round_trip_axis() {
        let mut prng = rand_chacha::ChaChaRng::seed_from_u64(0);
        for _ in 0..10000 {
            let expected = prng.gen_range(0o0_000_000, 0o2_000_000) << 13;
            let actual = Index64_3D::decode_axis(Index64_3D::encode_axis(expected));
            assert_eq!(actual, expected);
        }
    }

    #[test]
    fn normalize_to_system() {
        let system_bounds = Bounds{
            min: cgmath::Point3::new(-64f32, -64f32, -64f32),
            max: cgmath::Point3::new( 64f32,  64f32,  64f32)};
        let bounds = Bounds{
            min: cgmath::Point3::new(-32f32, -32f32, -32f32),
            max: cgmath::Point3::new( 32f32,  32f32,  32f32)};
        let expected = Bounds{
            min: cgmath::Point3::new(0.25f32, 0.25f32, 0.25f32),
            max: cgmath::Point3::new(0.75f32, 0.75f32, 0.75f32)};
        assert_eq!(bounds.normalize_to_system(system_bounds), expected);
    }
}
