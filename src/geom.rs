use super::index::SpatialIndex;
use super::traits::{Cast, Containment, MaxAxis, Quantize, QuantizeResult};

use cgmath::{BaseFloat, Point2, Vector2, Point3, Vector3};
use cgmath::prelude::*;
use num_traits::{NumAssignOps, NumCast, PrimInt, Unsigned};
use smallvec::SmallVec;

use std::fmt::Debug;
use std::ops::Shl;

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

/// The set of indices for an object with known bounds
/// 
/// By default, index `depth` is chosen such that this returns no more than 4 (2D) or 8 (3D) indices.
/// `min_depth` provides a lower-bound which enables quick partitioning (for parallel task generation)

pub trait LevelIndexBounds<Index>
where
    Index: SpatialIndex,
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
        Self{min, max}
    }

    pub fn size(self) -> Point::Diff {
        self.max - self.min
    }

    pub fn center(self) -> Point {
        self.min.midpoint(self.max)
    }

    pub fn normalize_point(self, point: Point) -> Point
    where
        Point::Scalar: BaseFloat,
        Point::Diff: ElementWise
    {
        EuclideanSpace::from_vec((point - self.min).div_element_wise(self.size()))
    }

    pub fn normalize_to_system(self, system_bounds: Bounds<Point>) -> Bounds<Point>
    where
        Point::Scalar: BaseFloat,
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
    Scalar: BaseFloat + QuantizeResult,
    Scalar::Quantized: PrimInt
{
    type Quantized = Point2<Scalar::Quantized>;
}

impl<Scalar> QuantizeResult for Point3<Scalar>
where 
    Scalar: BaseFloat + QuantizeResult,
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
    Point: EuclideanSpace<Scalar = Scalar> + ElementWise<Scalar> + QuantizeResult + Cast<<Point as QuantizeResult>::Quantized>,
    <Point as QuantizeResult>::Quantized: ElementWise<Scalar::Quantized>,
    Scalar: BaseFloat + QuantizeResult,
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
    Scalar: Debug + NumAssignOps + PrimInt + Unsigned + Shl<u32, Output = Scalar>,
    Index: SpatialIndex<Scalar = Scalar, Diff = Vector3<Scalar>, Point = Point3<Scalar>>
{
    type Output = SmallVec<[Index; 8]>;

    fn indices(self, min_depth: Option<u32>) -> Self::Output {
        let max_axis = self.size().max_axis();
        let mut depth = (max_axis - Index::Scalar::one()).leading_zeros();
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
                .set_origin(Index::Point::new(
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

/// [`Layer::test`]: struct.Layer.html#method.test
/// A trait for implementing individual geometry tests
/// 
/// See [`Layer::test`]. This is a low-level interface and should generally not be directly
/// implemented by users
pub trait TestGeometry: Sized {
    type SubdivideResult: AsRef<[Option<Self>]>;

    /// Subdivide this geometry, returning `None` for empty cells
    fn subdivide(&self) -> Self::SubdivideResult;
}

/// [`TestGeometry`]: trait.TestGeometry.html
/// A type implementing [`TestGeometry`] for rays
pub struct RayTestGeometry<Point>
where
    Point: EuclideanSpace,
    Point::Scalar: BaseFloat
{
    pub cell_bounds: Bounds<Point>,
    pub origin: Point,
    pub direction: Point::Diff,
    pub range_min: Point::Scalar,
    pub range_max: Point::Scalar
}

impl<Scalar> TestGeometry for RayTestGeometry<Point3<Scalar>>
where
    Scalar: BaseFloat
{
    type SubdivideResult = [Option<Self>; 8];

    fn subdivide(&self) -> Self::SubdivideResult {
        let center = self.cell_bounds.center();
        let distance = (self.cell_bounds.center() - self.origin).div_element_wise(self.direction);
        let mut results: [Option<Self>; 8] = [None, None, None, None, None, None, None, None];
        for cell in 0..8 {
            let mut range_min = self.range_min;
            let mut range_max = self.range_max;
            for axis in 0..3 {
                let side = cell & (1 << axis) != 0;
                let is_towards = self.direction[axis] > Scalar::zero() && !side;
                if is_towards {
                    range_max = range_max.min(distance[axis]);
                } else {
                    range_min = range_min.max(distance[axis]);
                }
            }
            if range_max > range_min {
                let mut bounds = self.cell_bounds;
                for axis in 0..3 {
                    let side = cell & (1 << axis) != 0;
                    if side {
                        bounds.min[axis] = center[axis];
                    } else {
                        bounds.max[axis] = center[axis];
                    }
                }
                results[cell] = Some(Self{
                    cell_bounds: bounds,
                    origin     : self.origin,
                    direction  : self.direction,
                    range_min  : range_min,
                    range_max  : range_max});
            }
        }
        results
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