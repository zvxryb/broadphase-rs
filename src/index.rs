// mlodato, 2020

use cgmath::{Point2, Point3, Vector2, Vector3};
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
/// 
/// The following index types are provided:
/// 
/// [`Index32_2D`]: struct.Index32_2D.html
/// [`Index64_2D`]: struct.Index64_2D.html
/// [`Index64_3D`]: struct.Index64_3D.html
/// 
/// * [`Index32_2D`]: A 32-bit 2D index type providing 14 bits' precision per axis
/// * [`Index64_2D`]: A 64-bit 2D index type providing 29 bits' precision per axis
/// * [`Index64_3D`]: A 64-bit 3D index type providing 19 bits' precision per axis

pub trait SpatialIndex: Clone + Copy + Default + Ord + Send + std::fmt::Debug {
    type Diff: cgmath::VectorSpace<Scalar = u32>;
    type Point: Copy + EuclideanSpace<Diff = Self::Diff, Scalar = u32>;

    /// clamps a depth value to the representable range
    fn clamp_depth(_: u32) -> u32;

    fn origin(self) -> Self::Point;
    fn depth(self) -> u32;

    fn set_origin(self, _: Self::Point) -> Self;
    fn set_depth(self, _: u32) -> Self;

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

macro_rules! index_impl {
    (index: $name:ident, $dim:tt, $bits:tt, $depth_bits:tt, $axis_bits:tt) => {
        #[derive(Clone, Copy, Default, Eq, Ord, PartialEq, PartialOrd)]
        #[cfg_attr(any(test, feature="serde"), derive(Deserialize, Serialize))]
        pub struct $name(index_impl!{primitive_type: $bits});

        impl $name {
            const DEPTH_BITS: u32 = $depth_bits;
            const DEPTH_SHIFT: u32 = 0;
            const DEPTH_MASK: index_impl!{primitive_type: $bits} = (((1 as index_impl!{primitive_type: $bits}) << Self::DEPTH_BITS) - 1) << Self::DEPTH_SHIFT;
            const AXIS_BITS: u32 = $axis_bits;
            const ORIGIN_BITS: u32 = $dim * Self::AXIS_BITS;
            const ORIGIN_SHIFT: u32 = Self::DEPTH_SHIFT + Self::DEPTH_BITS;
            const ORIGIN_MASK: index_impl!{primitive_type: $bits} = (((1 as index_impl!{primitive_type: $bits}) << (Self::ORIGIN_BITS)) - 1) << Self::ORIGIN_SHIFT;

            index_impl!{codec: $dim, $bits}
        
            fn level_mask(depth: u32) -> index_impl!{primitive_type: $bits} {
                if depth <= 0 { 0 } else {
                    (((1 as index_impl!{primitive_type: $bits}) << ($dim * depth)) - 1) << (Self::ORIGIN_BITS + Self::ORIGIN_SHIFT - $dim * depth)
                }
            }
        }

        impl SpatialIndex for $name {
            type Diff  = index_impl!{vector_type: $dim};
            type Point = index_impl!{point_type: $dim};

            fn clamp_depth(depth: u32) -> u32 {
                std::cmp::min(depth, Self::AXIS_BITS)
            }

            index_impl!{origin: $dim}

            fn depth(self) -> u32 {
                let Self(index) = self;
                ((index & Self::DEPTH_MASK) >> Self::DEPTH_SHIFT) as u32
            }

            index_impl!{set_origin: $dim}

            fn set_depth(self, depth: u32) -> Self {
                let Self(mut index) = self;
                index &= !Self::DEPTH_MASK;
                index |= Self::DEPTH_MASK & (
                    <index_impl!{primitive_type: $bits} as From<u32>>::from(Self::clamp_depth(depth)) << Self::DEPTH_SHIFT);
                Self(index)
            }

            index_impl!{subdivide: $dim, $bits}

            fn overlaps(self, other: Self) -> bool {
                Self::same_cell_at_depth(self, other, std::cmp::min(self.depth(), other.depth()))
            }

            fn same_cell_at_depth(Self(lhs): Self, Self(rhs): Self, depth: u32) -> bool {
                (lhs ^ rhs) & Self::level_mask(depth) == 0
            }
        }
    };
    (primitive_type: 32) => {u32};
    (primitive_type: 64) => {u64};
    (vector_type: 2) => {Vector2<u32>};
    (vector_type: 3) => {Vector3<u32>};
    (point_type: 2) => {Point2<u32>};
    (point_type: 3) => {Point3::<u32>};
    (codec: 2, $bits:tt) => {
        #[allow(overflowing_literals)] // allow (intentional) truncating casts
        #[inline]
        fn decode_axis(origin: index_impl!{primitive_type: $bits}) -> u32 {
            let axis00 =  origin & 0x1111_1111_1111_1111 as index_impl!{primitive_type: $bits};
            let axis01 = (origin & 0x4444_4444_4444_4444 as index_impl!{primitive_type: $bits}) >> 0x01;
            let axis0_ = axis00 | axis01;
            let axis10 =  axis0_ & 0x0303_0303_0303_0303 as index_impl!{primitive_type: $bits};
            let axis11 = (axis0_ & 0x3030_3030_3030_3030 as index_impl!{primitive_type: $bits}) >> 0x02;
            let axis1_ = axis10 | axis11;
            let axis20 =  axis1_ & 0x000f_000f_000f_000f as index_impl!{primitive_type: $bits};
            let axis21 = (axis1_ & 0x0f00_0f00_0f00_0f00 as index_impl!{primitive_type: $bits}) >> 0x04;
            let axis2_ = axis20 | axis21;
            let axis30 =  axis2_ & 0x0000_00ff_0000_00ff as index_impl!{primitive_type: $bits};
            let axis31 = (axis2_ & 0x00ff_0000_00ff_0000 as index_impl!{primitive_type: $bits}) >> 0x08;
            let axis3_ = axis30 | axis31;
            let axis40 =  axis3_ & 0x0000_0000_0000_ffff as index_impl!{primitive_type: $bits};
            let axis41 = (axis3_ & 0x0000_ffff_0000_0000 as index_impl!{primitive_type: $bits}) >> 0x10;
            let axis4_ = axis40 | axis41;
            (axis4_ as u32) << (32 - Self::AXIS_BITS)
        }

        #[allow(overflowing_literals)] // allow (intentional) truncating casts
        #[inline]
        fn encode_axis(origin: u32) -> index_impl!{primitive_type: $bits} {
            let axis0_ = <index_impl!{primitive_type: $bits} as From<u32>>::from(origin >> (32 - Self::AXIS_BITS));
            let axis00 =  axis0_          & 0x0000_0000_0000_ffff as index_impl!{primitive_type: $bits};
            let axis01 = (axis0_ << 0x10) & 0x0000_ffff_0000_0000 as index_impl!{primitive_type: $bits};
            let axis1_ = axis00 | axis01;
            let axis10 =  axis1_          & 0x0000_00ff_0000_00ff as index_impl!{primitive_type: $bits};
            let axis11 = (axis1_ << 0x08) & 0x00ff_0000_00ff_0000 as index_impl!{primitive_type: $bits};
            let axis2_ = axis10 | axis11;
            let axis20 =  axis2_          & 0x000f_000f_000f_000f as index_impl!{primitive_type: $bits};
            let axis21 = (axis2_ << 0x04) & 0x0f00_0f00_0f00_0f00 as index_impl!{primitive_type: $bits};
            let axis3_ = axis20 | axis21;
            let axis30 =  axis3_          & 0x0303_0303_0303_0303 as index_impl!{primitive_type: $bits};
            let axis31 = (axis3_ << 0x02) & 0x3030_3030_3030_3030 as index_impl!{primitive_type: $bits};
            let axis4_ = axis30 | axis31;
            let axis40 =  axis4_          & 0x1111_1111_1111_1111 as index_impl!{primitive_type: $bits};
            let axis41 = (axis4_ << 0x01) & 0x4444_4444_4444_4444 as index_impl!{primitive_type: $bits};
            axis40 | axis41
        }
    };
    (codec: 3, $bits:tt) => {
        #[inline]
        fn decode_axis(origin: index_impl!{primitive_type: $bits}) -> u32 {
            let axis00 =  origin & 0o1_001_001_001_001_001_001_001 as index_impl!{primitive_type: $bits};
            let axis01 = (origin & 0o0_010_010_010_010_010_010_010 as index_impl!{primitive_type: $bits}) >> 0o02;
            let axis02 = (origin & 0o0_100_100_100_100_100_100_100 as index_impl!{primitive_type: $bits}) >> 0o04;
            let axis0_ = axis00 | axis01 | axis02;
            let axis10 =  axis0_ & 0o0_007_000_000_007_000_000_007 as index_impl!{primitive_type: $bits};
            let axis11 = (axis0_ & 0o1_000_000_007_000_000_007_000 as index_impl!{primitive_type: $bits}) >> 0o06;
            let axis12 = (axis0_ & 0o0_000_007_000_000_007_000_000 as index_impl!{primitive_type: $bits}) >> 0o14;
            let axis1_ = axis10 | axis11 | axis12;
            let axis20 =  axis1_ & 0o0_000_000_000_000_000_000_777 as index_impl!{primitive_type: $bits};
            let axis21 = (axis1_ & 0o0_000_000_000_777_000_000_000 as index_impl!{primitive_type: $bits}) >> 0o22;
            let axis22 = (axis1_ & 0o0_777_000_000_000_000_000_000 as index_impl!{primitive_type: $bits}) >> 0o44;
            let axis2_ = axis20 | axis21 | axis22;
            (axis2_ as u32) << (32 - Self::AXIS_BITS)
        }

        #[inline]
        fn encode_axis(origin: u32) -> index_impl!{primitive_type: $bits} {
            let axis0_ = <index_impl!{primitive_type: $bits} as From<u32>>::from(origin >> (32 - Self::AXIS_BITS));
            let axis00 =  axis0_          & 0o0_000_000_000_000_000_000_777 as index_impl!{primitive_type: $bits};
            let axis01 = (axis0_ << 0o22) & 0o0_000_000_000_777_000_000_000 as index_impl!{primitive_type: $bits};
            let axis02 = (axis0_ << 0o44) & 0o0_777_000_000_000_000_000_000 as index_impl!{primitive_type: $bits};
            let axis1_ = axis00 | axis01 | axis02;
            let axis10 =  axis1_          & 0o0_007_000_000_007_000_000_007 as index_impl!{primitive_type: $bits};
            let axis11 = (axis1_ << 0o06) & 0o1_000_000_007_000_000_007_000 as index_impl!{primitive_type: $bits};
            let axis12 = (axis1_ << 0o14) & 0o0_000_007_000_000_007_000_000 as index_impl!{primitive_type: $bits};
            let axis2_ = axis10 | axis11 | axis12;
            let axis20 =  axis2_          & 0o1_001_001_001_001_001_001_001 as index_impl!{primitive_type: $bits};
            let axis21 = (axis2_ << 0o02) & 0o0_010_010_010_010_010_010_010 as index_impl!{primitive_type: $bits};
            let axis22 = (axis2_ << 0o04) & 0o0_100_100_100_100_100_100_100 as index_impl!{primitive_type: $bits};
            axis20 | axis21 | axis22
        }
    };
    (origin: 2) => {
        fn origin(self) -> index_impl!{point_type: 2} {
            let Self(index) = self;
            let origin = (index & Self::ORIGIN_MASK) >> Self::ORIGIN_SHIFT;
            Point2::new(
                Self::decode_axis(origin),
                Self::decode_axis(origin >> 1),
            )
        }
    };
    (origin: 3) => {
        fn origin(self) -> index_impl!{point_type: 3} {
            let Self(index) = self;
            let origin = (index & Self::ORIGIN_MASK) >> Self::ORIGIN_SHIFT;
            Point3::new(
                Self::decode_axis(origin),
                Self::decode_axis(origin >> 1),
                Self::decode_axis(origin >> 2),
            )
        }
    };
    (set_origin: 2) => {
        fn set_origin(self, origin: index_impl!{point_type: 2}) -> Self {
            let origin = Self::encode_axis(origin.x)
                       | Self::encode_axis(origin.y) << 1;
            let Self(mut index) = self;
            index &= !Self::ORIGIN_MASK;
            index |= Self::ORIGIN_MASK & (origin << Self::ORIGIN_SHIFT);
            Self(index)
        }
    };
    (set_origin: 3) => {
        fn set_origin(self, origin: index_impl!{point_type: 3}) -> Self {
            let origin = Self::encode_axis(origin.x)
                       | Self::encode_axis(origin.y) << 1
                       | Self::encode_axis(origin.z) << 2;
            let Self(mut index) = self;
            index &= !Self::ORIGIN_MASK;
            index |= Self::ORIGIN_MASK & (origin << Self::ORIGIN_SHIFT);
            Self(index)
        }
    };
    (subdivide: 2, $bits:tt) => {
        type SubdivideResult = [Self; 4];
        fn subdivide(self) -> Option<[Self; 4]> {
            let depth = self.depth();
            if depth < Self::AXIS_BITS {
                let Self(index) = self;
                let shift = Self::ORIGIN_BITS + Self::ORIGIN_SHIFT - (2 * (depth + 1));
                Some([
                    Self(index | (0b00 as index_impl!{primitive_type: $bits} << shift)).set_depth(depth + 1),
                    Self(index | (0b01 as index_impl!{primitive_type: $bits} << shift)).set_depth(depth + 1),
                    Self(index | (0b10 as index_impl!{primitive_type: $bits} << shift)).set_depth(depth + 1),
                    Self(index | (0b11 as index_impl!{primitive_type: $bits} << shift)).set_depth(depth + 1)
                ])
            } else {
                None
            }
        }
    };
    (subdivide: 3, $bits:tt) => {
        type SubdivideResult = [Self; 8];
        fn subdivide(self) -> Option<[Self; 8]> {
            let depth = self.depth();
            if depth < Self::AXIS_BITS {
                let Self(index) = self;
                let shift = Self::ORIGIN_BITS + Self::ORIGIN_SHIFT - (3 * (depth + 1));
                Some([
                    Self(index | (0b000 as index_impl!{primitive_type: $bits} << shift)).set_depth(depth + 1),
                    Self(index | (0b001 as index_impl!{primitive_type: $bits} << shift)).set_depth(depth + 1),
                    Self(index | (0b010 as index_impl!{primitive_type: $bits} << shift)).set_depth(depth + 1),
                    Self(index | (0b011 as index_impl!{primitive_type: $bits} << shift)).set_depth(depth + 1),
                    Self(index | (0b100 as index_impl!{primitive_type: $bits} << shift)).set_depth(depth + 1),
                    Self(index | (0b101 as index_impl!{primitive_type: $bits} << shift)).set_depth(depth + 1),
                    Self(index | (0b110 as index_impl!{primitive_type: $bits} << shift)).set_depth(depth + 1),
                    Self(index | (0b111 as index_impl!{primitive_type: $bits} << shift)).set_depth(depth + 1)
                ])
            } else {
                None
            }
        }
    };
}

index_impl!{index: Index32_2D, 2, 32, 4, 14}
index_impl!{index: Index64_2D, 2, 64, 5, 29}
index_impl!{index: Index64_3D, 3, 64, 5, 19}

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

impl Debug for Index32_2D {
    fn fmt(&self, f: &mut Formatter) -> std::fmt::Result {
        let Self(index) = self;
        let origin_bits = (index & Self::ORIGIN_MASK) >> Self::ORIGIN_SHIFT;
        let origin = self.origin();
        write!(f, "Index32_2D{{origin={{0x{:011x}, <0x{:04x}, 0x{:04x}>}}, depth={:}}}",
            origin_bits,
            origin.x,
            origin.y,
            self.depth())
    }
}

impl Debug for Index64_2D {
    fn fmt(&self, f: &mut Formatter) -> std::fmt::Result {
        let Self(index) = self;
        let origin_bits = (index & Self::ORIGIN_MASK) >> Self::ORIGIN_SHIFT;
        let origin = self.origin();
        write!(f, "Index64_2D{{origin={{0x{:022x}, <0x{:08x}, 0x{:08x}>}}, depth={:}}}",
            origin_bits,
            origin.x,
            origin.y,
            self.depth())
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
