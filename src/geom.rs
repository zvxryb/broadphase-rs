// mlodato, 20190317

use crate::index::SpatialIndex;

use cgmath::{Point3, Vector3};
use cgmath::prelude::*;
use num_traits::{Float, One, PrimInt};
use smallvec::SmallVec;

use std::fmt::{Debug, Formatter};

fn fold_arr<Arr, State, F>(arr: Arr, init: State, f: F) -> State
where
    Arr: Array,
    F: FnMut(State, Arr::Element) -> State
{
    (0..Arr::len()).map(|i| arr[i]).fold(init, f)
}

fn init_arr<Arr, F>(arr: &mut Arr, mut f: F)
where
    Arr: Array,
    F: FnMut(usize) -> Arr::Element
{
    for i in 0..Arr::len() {
        arr[i] = f(i);
    }
}

fn max_axis<Arr>(arr: Arr) -> Arr::Element
where
    Arr: Array,
    Arr::Element: Bounded + Ord
{
    fold_arr(arr, Arr::Element::min_value(), std::cmp::max)
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

pub trait IndexGenerator<Index>
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
/// Min and max values are _inclusive_
#[derive(Copy, Clone, Debug, PartialEq)]
#[cfg_attr(any(test, feature="serde"), derive(Deserialize, Serialize))]
pub struct Bounds<Point> {
    pub min: Point,
    pub max: Point
}

impl<Point> Bounds<Point>
where
    Point: EuclideanSpace + Array<Element = <Point as EuclideanSpace>::Scalar> + Copy
{
    pub fn new(min: Point, max: Point) -> Self {
        Self{min, max}
    }

    pub fn sizef(self) -> Point::Diff
    where
        Point::Scalar: Float
    {
        self.max - self.min
    }

    pub fn sizei(self) -> Point::Diff
    where
        Point::Scalar: PrimInt + One,
        Point::Diff: ElementWise<Point::Scalar>
    {
        (self.max - self.min).add_element_wise(Point::Scalar::one())
    }

    pub fn overlaps(self, other: Bounds<Point>) -> bool {
        for i in 0..Point::len() {
            if self.min[i] > other.max[i] || self.max[i] < other.min[i] {
                return false;
            }
        }
        true
    }

    pub fn contains(self, other: Bounds<Point>) -> bool {
        for i in 0..Point::len() {
            if self.min[i] > other.min[i] || self.max[i] < other.max[i] {
                return false;
            }
        }
        true
    }

    pub fn center(self) -> Point {
        self.min.midpoint(self.max)
    }
}

/// System bounds supporting conversions between local and global coordinates
pub trait SystemBounds<PointGlobal, PointLocal> {
    fn to_local(&self, global: Bounds<PointGlobal>) -> Bounds<PointLocal>;
    fn to_global(&self, local: Bounds<PointLocal>) -> Bounds<PointGlobal>;
}

impl<PointGlobal, PointLocal> SystemBounds<PointGlobal, PointLocal> for Bounds<PointGlobal>
where
    PointGlobal: EuclideanSpace<Scalar = f32>,
    PointGlobal::Diff: Array<Element = f32>,
    PointLocal: EuclideanSpace<Scalar = u32>,
    PointLocal::Diff: Array<Element = u32>
{
    fn to_local(&self, global: Bounds<PointGlobal>) -> Bounds<PointLocal> {
        let size = self.sizef();
        let to_local = |global: PointGlobal, i| {
            // MAX_VALUE has 24 bits set because IEEE floats have 23 explicit + 1 implicit fractional bits
            const MIN_VALUE: f32 = std::u32::MIN as f32;
            const MAX_VALUE: f32 = 0xffff_ff00u32 as f32;
            const RANGE: f32 = MAX_VALUE - MIN_VALUE;
            ((global[i] - self.min[i]) / size[i] * RANGE + MIN_VALUE) as u32
        };
        let mut local = Bounds::new(
            PointLocal::from_vec(PointLocal::Diff::zero()),
            PointLocal::from_vec(PointLocal::Diff::zero()));
        init_arr(&mut local.min, |i| to_local(global.min, i));
        init_arr(&mut local.max, |i| to_local(global.max, i));
        local
    }

    fn to_global(&self, local: Bounds<PointLocal>) -> Bounds<PointGlobal> {
        let size = self.sizef();
        let to_global = |local: PointLocal, i| {
            // MAX_VALUE has 24 bits set because IEEE floats have 23 explicit + 1 implicit fractional bits
            const MIN_VALUE: f32 = std::u32::MIN as f32;
            const MAX_VALUE: f32 = 0xffff_ff00u32 as f32;
            const RANGE: f32 = MAX_VALUE - MIN_VALUE;
            self.min[i] + (local[i] as f32 - MIN_VALUE) / RANGE * size[i]
        };
        let mut global = Bounds::new(
            PointGlobal::from_vec(PointGlobal::Diff::zero()),
            PointGlobal::from_vec(PointGlobal::Diff::zero()));
        init_arr(&mut global.min, |i| to_global(local.min, i));
        init_arr(&mut global.max, |i| to_global(local.max, i));
        global
    }
}

impl<Index> IndexGenerator<Index> for Bounds<Point3<u32>>
where
    Index: SpatialIndex<Diff = Vector3<u32>, Point = Point3<u32>>
{
    type Output = SmallVec<[Index; 8]>;

    fn indices(self, min_depth: Option<u32>) -> Self::Output {
        let max_axis = max_axis(self.sizei());
        let mut depth = (max_axis - 1u32).leading_zeros();
        if let Some(min_depth) = min_depth {
            if depth < min_depth {
                depth = min_depth;
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

impl<Index, Point> From<Index> for Bounds<Point>
where
    Index: SpatialIndex<Diff = Point::Diff, Point = Point>,
    Point: EuclideanSpace<Scalar = u32> + ElementWise<u32>
{
    fn from(index: Index) -> Self {
        let origin = index.origin();
        let scale = scale_at_depth(index.depth());
        Self{
            min: origin,
            max: origin.add_element_wise(scale-1)
        }
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
            origin,
            direction,
            range_min,
            range_max}
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
        for (cell, result) in results.iter_mut().enumerate() {
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
            #[allow(clippy::needless_range_loop)]
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
        #[allow(clippy::collapsible_if)]
        let axes = if abs.x <= abs.y && abs.x <= abs.z {
            if abs.y <= abs.z { [0, 1, 2] } else { [0, 2, 1] }
        } else if abs.y <= abs.z {
            if abs.x <= abs.z { [1, 0, 2] } else { [1, 2, 0] }
        } else {
            if abs.x <= abs.y { [2, 0, 1] } else { [2, 1, 0] }
        };

        let mut order: [usize; 8] = [0; 8];
        for (cell_src, cell_dst) in order.iter_mut().enumerate() {
            let i0 = (cell_src & 1 != 0) == (self.direction[axes[0]] >= 0f32);
            let i1 = (cell_src & 2 != 0) == (self.direction[axes[1]] >= 0f32);
            let i2 = (cell_src & 4 != 0) == (self.direction[axes[2]] >= 0f32);
            *cell_dst =
                ((i0 as usize) << axes[0]) |
                ((i1 as usize) << axes[1]) |
                ((i2 as usize) << axes[2])
        }

        order
    }

    fn should_test(&self, nearest: f32) -> bool {
        self.range_min < self.range_max && self.range_min < nearest
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn system_bounds() {
        let system_bounds = Bounds{
            min: Point3::new(-64f32, -64f32, -64f32),
            max: Point3::new( 64f32,  64f32,  64f32)};
        let global = Bounds{
            min: Point3::new(-32f32, -32f32, -32f32),
            max: Point3::new( 32f32,  32f32,  32f32)};
        let local: Bounds<Point3<u32>> = system_bounds.to_local(global);
        let expected = global;
        let actual = system_bounds.to_global(local);
        assert_eq!(actual, expected);
    }
}