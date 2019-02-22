// mlodato, 20190221

use cgmath::Point3;

/// Two properties are required to ensure that sorting indices produces a topological ordering:
///
/// 1. `self.origin()` should represent the lower bound for the node at `self.depth()`
/// 2. `self.depth()` should be less significant than origin for ordering
///
/// The underlying primitive type can be sorted directly if high-bits store origin and low-bits store depth

pub trait SpatialIndex: Clone + Copy + Default + Ord + Send + std::fmt::Debug {
    type Point;

    fn clamp_depth(u32) -> u32;
    fn origin(self) -> Self::Point;
    fn depth(self) -> u32;
    fn set_origin(self, Self::Point) -> Self;
    fn set_depth(self, u32) -> Self;
    fn overlaps(self, other: Self) -> bool;
}

/// 64-bit 3D index; provides 19 bits' precision per axis

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

impl SpatialIndex for Index64_3D {
    type Point = Point3<u32>;

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
