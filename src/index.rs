// mlodato, 20190317

use cgmath::{Point3, Vector3};
use cgmath::prelude::*;
use std::fmt::{Debug, Formatter};

/// An index representing an object's position and scale
/// 
/// The `Ord` trait must be implemented such that sorting produces a topological ordering.
/// 
/// This may be accomplished using trivial comparison operators for a primitive integer type by:
///
/// 1. Packing bits such that `origin` is higher-significance than `depth`
/// 2. Storing the value of `origin` as a Morton code
/// 3. Truncating `origin` bits to the level specified by `depth`, such that it represents the _minimum bound_
///    of the cell at the given scale
/// 
/// Currently, requirement #3 (truncating origin) is not the responsibility of the particular `SpatialIndex`
/// implementation &mdash; an appropriately-truncated value must be passed as an argument to `set_origin`
/// 
/// `Ord` should be implemented such that an X-bit is lower significance (changes more rapidly) than the
/// corresponding a Y-bit (which should, likewise, be lower significance than the corresponding Z-bit for
/// 3D indices)
/// 
/// `<SpatialIndex as Default>::default()` is required to return an index which encompasses the entire system
/// bounds (i.e. zero origin and zero depth)

pub trait SpatialIndex: Clone + Copy + Default + Ord + Send + std::fmt::Debug {
    type Diff: cgmath::VectorSpace<Scalar = u32>;
    type Point: Copy + EuclideanSpace<Diff = Self::Diff, Scalar = u32>;

    /// clamps a depth value to the representable range
    fn clamp_depth(u32) -> u32;

    fn origin(self) -> Self::Point;
    fn depth(self) -> u32;

    fn set_origin(self, Self::Point) -> Self;
    fn set_depth(self, u32) -> Self;

    type SubdivideResult: AsRef<[Self]>;

    /// Subdivide the cell represented by this index into cells of `depth + 1`
    /// 
    /// This is required to return results in sorted order.  Returns `None` if depth limit has been reached.
    fn subdivide(self) -> Option<Self::SubdivideResult>;

    /// Check if two indices represent overlapping regions of space
    fn overlaps(self, other: Self) -> bool;

    /// Check if two indices would fall into the same cell at a given (truncated) depth
    fn same_cell_at_depth(lhs: Self, rhs: Self, depth: u32) -> bool;
}

/// A 64-bit 3D index which provides 19 bits' precision per axis

#[derive(Clone, Copy, Default, Eq, Ord, PartialEq, PartialOrd)]
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
        let axis0_ = <u64 as From<u32>>::from(origin >> (32 - Self::AXIS_BITS));
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

impl Debug for Index64_3D {
    fn fmt(&self, f: &mut Formatter) -> std::fmt::Result {
        let Self(index) = self;
        let origin_bits = (index & Self::ORIGIN_MASK) >> Self::ORIGIN_SHIFT;
        let origin = self.origin();
        write!(f, "Index64_3D{{origin={{0o{:019o}, <0x{:08x}, 0x{:08x}, 0x{:08x}>}}, depth={:}}}",
            origin_bits,
            origin.x,
            origin.y,
            origin.z,
            self.depth())
    }
}

impl SpatialIndex for Index64_3D {
    type Diff = Vector3<u32>;
    type Point = Point3<u32>;

    fn clamp_depth(depth: u32) -> u32 {
        std::cmp::min(depth, Self::AXIS_BITS)
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
        index |= Self::DEPTH_MASK & (
            <u64 as From<u32>>::from(Self::clamp_depth(depth)) << Self::DEPTH_SHIFT);
        Self(index)
    }

    type SubdivideResult = [Self; 8];
    fn subdivide(self) -> Option<[Self; 8]> {
        let depth = self.depth();
        if depth < Self::AXIS_BITS {
            let Self(index) = self;
            let shift = Self::ORIGIN_BITS + Self::ORIGIN_SHIFT - (3 * (depth + 1));
            Some([
                Self(index | (0b000u64 << shift)).set_depth(depth + 1),
                Self(index | (0b001u64 << shift)).set_depth(depth + 1),
                Self(index | (0b010u64 << shift)).set_depth(depth + 1),
                Self(index | (0b011u64 << shift)).set_depth(depth + 1),
                Self(index | (0b100u64 << shift)).set_depth(depth + 1),
                Self(index | (0b101u64 << shift)).set_depth(depth + 1),
                Self(index | (0b110u64 << shift)).set_depth(depth + 1),
                Self(index | (0b111u64 << shift)).set_depth(depth + 1)
            ])
        } else {
            None
        }
    }

    fn overlaps(self, other: Self) -> bool {
        Self::same_cell_at_depth(self, other, std::cmp::min(self.depth(), other.depth()))
    }

    fn same_cell_at_depth(Self(lhs): Self, Self(rhs): Self, depth: u32) -> bool {
        (lhs ^ rhs) & Self::level_mask(depth) == 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::prelude::*;

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
}
