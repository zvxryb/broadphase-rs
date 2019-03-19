// mlodato, 20190317

use super::index::SpatialIndex;
use super::traits::{Containment, MaxAxis, Quantize};

use cgmath::{Point2, Vector2, Point3, Vector3};
use cgmath::prelude::*;
use smallvec::SmallVec;

use std::fmt::{Debug, Formatter};

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

fn scale_at_depth(depth: u32) -> u32 {
    if depth == 0 {
        panic!("scale at zero depth would overflow integer");
    }
    1u32 << (32 - depth)
}

fn truncate_to_depth(x: u32, depth: u32) -> u32 {
    if depth == 0 {
        x
    } else {
        x & !(scale_at_depth(depth) - 1u32)
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
        Point::Diff: ElementWise
    {
        EuclideanSpace::from_vec((point - self.min).div_element_wise(self.size()))
    }

    pub fn normalize_to_system(self, system_bounds: Bounds<Point>) -> Bounds<Point>
    where
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

impl Quantize for Point2<f32> {
    type Quantized = Point2<u32>;

    fn quantize(self) -> Option<Self::Quantized> {
        const MIN_VALUE: f32 = std::u32::MIN as f32;
        const MAX_VALUE: f32 = (std::u32::MAX - 1u32) as f32;
        const RANGE: f32 = MAX_VALUE - MIN_VALUE;
        Point2::cast(&(self * RANGE).add_element_wise(MIN_VALUE))
    }
}

impl Quantize for Point3<f32> {
    type Quantized = Point3<u32>;

    fn quantize(self) -> Option<Self::Quantized> {
        const MIN_VALUE: f32 = std::u32::MIN as f32;
        const MAX_VALUE: f32 = (std::u32::MAX - 1u32) as f32;
        const RANGE: f32 = MAX_VALUE - MIN_VALUE;
        Point3::cast(&(self * RANGE).add_element_wise(MIN_VALUE))
    }
}

impl<Point> Quantize for Bounds<Point>
where
    Point: Quantize,
    Point::Quantized: ElementWise<u32>
{
    type Quantized = Bounds<Point::Quantized>;

    fn quantize(self) -> Option<Self::Quantized> {
        Some(Bounds{
            min: self.min.quantize()?,
            max: self.max.quantize()?.add_element_wise(1u32)
        })
    }
}

impl<Index> LevelIndexBounds<Index> for Bounds<Point3<u32>>
where
    Index: SpatialIndex<Diff = Vector3<u32>, Point = Point3<u32>>
{
    type Output = SmallVec<[Index; 8]>;

    fn indices(self, min_depth: Option<u32>) -> Self::Output {
        let max_axis = self.size().max_axis();
        let mut depth = (max_axis - 1u32).leading_zeros();
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
            return smallvec![Index::default()];
        }

        let min = self.min.map(|scalar| truncate_to_depth(scalar, depth));
        let max = self.max.map(|scalar| truncate_to_depth(scalar, depth));

        let mut indices: Self::Output = Self::Output::new();

        let step = scale_at_depth(depth);
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
pub trait TestGeometry: Sized + Debug {
    type SubdivideResult: AsRef<[Self]>;
    type TestOrder: AsRef<[usize]>;

    /// [`SpatialIndex::subdivide`]: trait.SpatialIndex.html#tymethod.subdivide
    /// [`test_order`]: #tymethod.test_order
    /// Subdivide this geometry
    /// 
    /// This is required to return results in the same order as [`SpatialIndex::subdivide`], both
    /// results will be reordered as given by [`test_order`]
    fn subdivide(&self) -> Self::SubdivideResult;

    /// The order in which to test cells
    /// 
    /// This is used to optimize tests where only the single, nearest, result should be returned
    fn test_order(&self) -> Self::TestOrder;

    /// Return whether this geometry is valid and non-empty
    /// 
    /// `nearest` may be `std::f32::INFINITY`
    fn should_test(&self, nearest: f32) -> bool;
}

/// [`TestGeometry`]: trait.TestGeometry.html
/// A type implementing [`TestGeometry`] for rays
#[derive(Clone)]
pub struct RayTestGeometry<Point>
where
    Point: EuclideanSpace<Scalar = f32>
{
    cell_bounds: Bounds<Point>,
    origin: Point,
    direction: Point::Diff,
    range_min: f32,
    range_max: f32
}

impl Debug for RayTestGeometry<Point3<f32>> {
    fn fmt(&self, f: &mut Formatter) -> std::fmt::Result {
        write!(f, "RayTestGeometry{{{{({:}, {:}, {:}) - ({:}, {:}, {:})}}, ({:}, {:}, {:}), ({:}, {:}, {:}), {{{:}-{:}}}}}",
            self.cell_bounds.min.x,
            self.cell_bounds.min.y,
            self.cell_bounds.min.z,
            self.cell_bounds.max.x,
            self.cell_bounds.max.y,
            self.cell_bounds.max.z,
            self.origin.x,
            self.origin.y,
            self.origin.z,
            self.direction.x,
            self.direction.y,
            self.direction.z,
            self.range_min,
            self.range_max)
    }
}

impl<Point> RayTestGeometry<Point>
where
    Point: EuclideanSpace<Scalar = f32>
{
    /// Construct ray test geometry
    /// 
    /// `range_min` and `range_max` may be infinity or negative infinity, in which case
    /// the ray will be clamped to system bounds
    pub fn with_system_bounds(
        system_bounds: Bounds<Point>,
        origin: Point,
        direction: Point::Diff,
        mut range_min: f32,
        mut range_max: f32) -> Self
    where
        Point: Debug,
        Point::Diff: ElementWise + std::ops::Index<usize, Output = f32> + Debug,
    {
        let distance_0 = (system_bounds.min - origin).div_element_wise(direction);
        let distance_1 = (system_bounds.max - origin).div_element_wise(direction);
        for axis in 0..3 {
            let is_forward = direction[axis] > 0f32;
            let (d0, d1) = if is_forward {
                    (distance_0[axis], distance_1[axis])
                } else {
                    (distance_1[axis], distance_0[axis])
                };
            if d0.is_finite() { range_min = range_min.max(d0); }
            if d1.is_finite() { range_max = range_max.min(d1); }
        }

        Self{
            cell_bounds: system_bounds,
            origin     : origin,
            direction  : direction,
            range_min  : range_min,
            range_max  : range_max}
    }
}

impl TestGeometry for RayTestGeometry<Point3<f32>> {
    type SubdivideResult = [Self; 8];
    type TestOrder = [usize; 8];

    fn subdivide(&self) -> Self::SubdivideResult {
        let center = self.cell_bounds.center();
        let distance = (self.cell_bounds.center() - self.origin).div_element_wise(self.direction);
        let mut results: [Self; 8] = [
            self.clone(),
            self.clone(),
            self.clone(),
            self.clone(),
            self.clone(),
            self.clone(),
            self.clone(),
            self.clone()
        ];
        for cell in 0..8 {
            let result = &mut results[cell];
            let range_min = &mut result.range_min;
            let range_max = &mut result.range_max;
            for axis in 0..3 {
                let side = cell & (1 << axis) != 0;
                if distance[axis].is_finite() {
                    let is_towards = (self.direction[axis] > 0f32) != side;
                    if is_towards {
                        *range_max = range_max.min(distance[axis]);
                    } else {
                        *range_min = range_min.max(distance[axis]);
                    }
                } else if (self.origin[axis] > center[axis]) != side {
                    *range_min = std::f32::INFINITY;
                    *range_max = std::f32::NEG_INFINITY;
                }
            }
            let bounds = &mut result.cell_bounds;
            for axis in 0..3 {
                let side = cell & (1 << axis) != 0;
                if side {
                    bounds.min[axis] = center[axis];
                } else {
                    bounds.max[axis] = center[axis];
                }
            }
        }
        results
    }

    fn test_order(&self) -> Self::TestOrder {
        let abs = self.direction.map(|x| x.abs());
        let axes = if abs.x <= abs.y && abs.x <= abs.z {
            if abs.y <= abs.z { [0, 1, 2] } else { [0, 2, 1] }
        } else if abs.y <= abs.z {
            if abs.x <= abs.z { [1, 0, 2] } else { [1, 2, 0] }
        } else {
            if abs.x <= abs.y { [2, 0, 1] } else { [2, 1, 0] }
        };

        let mut order: [usize; 8] = [0; 8];
        for i in 0..8 {
            let i0 = (i & 1 != 0) == (self.direction[axes[0]] >= 0f32);
            let i1 = (i & 2 != 0) == (self.direction[axes[1]] >= 0f32);
            let i2 = (i & 4 != 0) == (self.direction[axes[2]] >= 0f32);
            order[i] =
                ((i0 as usize) << axes[0]) |
                ((i1 as usize) << axes[1]) |
                ((i2 as usize) << axes[2])
        }

        order
    }

    fn should_test(&self, nearest: f32) -> bool {
        return self.range_min < self.range_max && self.range_min < nearest;
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