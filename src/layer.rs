use super::geom::{Bounds, LevelIndexBounds};
use super::index::SpatialIndex;
use super::traits::{Containment, ObjectID, Quantize, QuantizeResult};

use cgmath::prelude::*;
use num_traits::NumAssignOps;
use smallvec::SmallVec;

use std::fmt::Debug;
use std::ops::DerefMut;

#[cfg(feature="parallel")]
use rayon::prelude::*;

#[cfg(feature="parallel")]
use std::cell::{RefMut, RefCell};

#[cfg(feature="parallel")]
use thread_local::CachedThreadLocal;

/// [`SpatialIndex`]: trait.SpatialIndex.html
/// [`Index64_3D`]: trait.Index64_3D.html

/// A group of collision data
/// 
/// `Index` must be a type implmenting [`SpatialIndex`], such as [`Index64_3D`]
/// 
/// `ID` is the type representing object IDs

pub struct Layer<Index, ID>
where
    Index: SpatialIndex,
    ID: ObjectID,
    Bounds<Index::Point>: LevelIndexBounds<Index>
{
    min_depth: u32,
    tree: (Vec<(Index, ID)>, bool),
    collisions: Vec<(ID, ID)>,
    invalid: Vec<ID>,

    #[cfg(feature="parallel")]
    collisions_tls: CachedThreadLocal<RefCell<Vec<(ID, ID)>>>,
}

impl<Index, ID> Layer<Index, ID>
where
    Index: SpatialIndex,
    ID: ObjectID,
    Bounds<Index::Point>: LevelIndexBounds<Index>
{
    /// Iterate over all indices in the `Layer`
    /// 
    /// This is primarily intended for visualization + debugging
    pub fn iter<'a>(&'a self) -> std::slice::Iter<'a, (Index, ID)> {
        self.tree.0.iter()
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

    /// [`par_scan_filtered`]: struct.Layer.html#method.par_scan_filtered
    /// [`par_scan`]: struct.Layer.html#method.par_scan
    /// Sort indices to ready data for detection (parallel)
    /// 
    /// This will be called implicitly when necessary (i.e. by [`par_scan_filtered`], [`par_scan`], etc.)
    #[cfg(feature="parallel")]
    pub fn par_sort(&mut self) {
        let (tree, sorted) = &mut self.tree;
        if !*sorted {
            tree.par_sort_unstable();
            *sorted = true;
        }
    }

    /// [`scan_filtered`]: struct.Layer.html#method.scan_filtered
    /// [`scan`]: struct.Layer.html#method.scan
    /// Sort indices to ready data for detection
    /// 
    /// This will be called implicitly when necessary (i.e. by [`scan_filtered`], [`scan`], etc.)
    pub fn sort(&mut self) {
        let (tree, sorted) = &mut self.tree;
        if !*sorted {
            tree.sort_unstable();
            *sorted = true;
        }
    }

    /// Detects collisions between all objects in the `Layer`
    pub fn scan<'a>(&'a mut self)
        -> &'a Vec<(ID, ID)>
    {
        self.scan_filtered(|_, _| true)
    }

    /// Detects collisions between all objects in the `Layer`, returning only those which pass a user-specified test
    /// 
    /// Collisions are filtered prior to duplicate removal.  This may be faster or slower than filtering
    /// post-duplicate-removal (i.e. by `scan().iter().filter()`) depending on the complexity
    /// of the filter.
    pub fn scan_filtered<'a, F>(&'a mut self, filter: F)
        -> &'a Vec<(ID, ID)>
    where
        F: FnMut(ID, ID) -> bool
    {
        self.sort();

        let (tree, _) = &self.tree;
        Self::scan_impl(tree.as_slice(), &mut self.collisions, filter);

        self.collisions.sort_unstable();
        self.collisions.dedup();

        &self.collisions
    }

    /// [`scan`]: struct.Layer.html#method.scan
    /// Parallel version of [`scan`]
    #[cfg(feature="parallel")]
    pub fn par_scan<'a>(&'a mut self)
        -> &'a Vec<(ID, ID)>
    where
        Index: Send + Sync
    {
        self.par_scan_filtered(|_, _| true)
    }

    /// [`scan_filtered`]: struct.Layer.html#method.scan_filtered
    /// Parallel version of [`scan_filtered`]
    #[cfg(feature="parallel")]
    pub fn par_scan_filtered<'a, F>(&'a mut self, filter: F)
        -> &'a Vec<(ID, ID)>
    where
        Index: Send + Sync,
        F: Copy + Send + Sync + FnMut(ID, ID) -> bool
    {
        self.par_sort();

        for set in self.collisions_tls.iter_mut() {
            set.borrow_mut().clear();
        }

        self.par_scan_impl(rayon::current_num_threads(), self.tree.0.as_slice(), filter);

        for set in self.collisions_tls.iter_mut() {
            use std::borrow::Borrow;
            let set_: RefMut<Vec<(ID, ID)>> = set.borrow_mut();
            let set__: &Vec<(ID, ID)> = set_.borrow();
            self.collisions.extend(set__.iter());
        }

        self.collisions.par_sort_unstable();
        self.collisions.dedup();

        &self.collisions
    }

    #[cfg(feature="parallel")]
    fn par_scan_impl<F>(&self, threads: usize, tree: &[(Index, ID)], filter: F)
    where
        Index: Send + Sync,
        F: Copy + Send + Sync + FnMut(ID, ID) -> bool
    {
        const SPLIT_THRESHOLD: usize = 64;
        if threads <= 1 || tree.len() <= SPLIT_THRESHOLD {
            let collisions = self.collisions_tls.get_or(|| Box::new(RefCell::new(Vec::new())));
            Self::scan_impl(tree, collisions.borrow_mut(), filter);
        } else {
            let n = tree.len();
            let mut i = n / 2;
            while i < n {
                let (last, _) = tree[i-1];
                let (next, _) = tree[i];
                if !Index::same_cell_at_depth(last, next, self.min_depth) {
                    break;
                }
                i += 1;
            }
            let (head, tail) = tree.split_at(i);
            rayon::join(
                || self.par_scan_impl(threads >> 1, head, filter),
                || self.par_scan_impl(threads >> 1, tail, filter));
        }
    }

    fn scan_impl<C, F>(tree: &[(Index, ID)], mut collisions: C, mut filter: F)
    where
        C: DerefMut<Target = Vec<(ID, ID)>>,
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
                    collisions.push((id, id_));
                }
            }
            stack.push((index, id))
        }
    }
}

/// A builder for `Layer`s
#[derive(Default)]
pub struct LayerBuilder {
    min_depth: u32,
    index_capacity: Option<usize>,
    collision_capacity: Option<usize>
}

impl LayerBuilder {
    pub fn new() -> Self {
        Self::default()
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
        ID: ObjectID,
        Bounds<Index::Point>: LevelIndexBounds<Index>
    {
        Layer::<Index, ID>{
            min_depth: self.min_depth,
            tree: (match self.index_capacity {
                    Some(capacity) => Vec::with_capacity(capacity),
                    None => Vec::new()
                }, true),
            collisions: match self.collision_capacity {
                    Some(capacity) => Vec::with_capacity(capacity),
                    None => Vec::new()
                },
            invalid: Vec::new(),
            #[cfg(feature="parallel")]
            collisions_tls: CachedThreadLocal::new()
        }
    }
}