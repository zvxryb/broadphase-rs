# broadphase-rs [![Crate](https://img.shields.io/crates/v/zvxryb-broadphase)](https://crates.io/crates/zvxryb-broadphase) [![Docs](https://docs.rs/zvxryb-broadphase/badge.svg)](https://docs.rs/zvxryb-broadphase) [![Build](https://github.com/zvxryb/broadphase-rs/workflows/Build/badge.svg)](https://github.com/zvxryb/broadphase-rs/actions)

![Collision Grid](docs/images/example_with_grid.png)

## Overview

broadphase-rs is a, creatively named, broadphase collision detection library in Rust.  It transforms object bounds
into a lightweight spatial index representation, a single integer.  A vector of such indices is sorted directly to
yield a result which is a topologically-sorted Morton order, after which full-system collision detection can be
accomplished by a single pass over the sorted list with only a minimal auxiliary stack necessary to maintain state.
Collision tests between indices are accomplished with simple bitwise shifts, masks, and XORs.

This method is capable of supporting objects of varying scale (unlike uniform grids), while having a straightforward,
non-hierarchical structure in memory (unlike quad- or oct-trees), as the entire representation exists in a single
vector of index/object pairs.

broadphase-rs is capable of supporting many thousands of objects at interactive speeds.  While it has not (yet) been
benchmarked against alternatives, the example's collision detection routine takes roughly ~6ms for 10,000 dynamic
objects (stable-x86_64-pc-windows-msvc, rustc 1.32.0, Intel Core i5 6600K, release mode, using `par_scan()`)

![Visualization](docs/images/visualization.apng)

## Features

* Support for both 2D and 3D systems
* Full-system collision checking with `Layer::scan`
* User-defined collision filters with `Layer::scan_filtered`
* Layers can be pre-computed and merged (using `Layer::merge`) to avoid recalculation of static data
* Optional multi-threaded operations using Rayon (`Layer::par_sort` and `Layer::par_scan`)
* Individual queries for boxes (`Layer::test_box`), rays (`Layer::test_ray`), or user-specified tests (`Layer::test`)
* Picking first element along a ray (`Layer::pick_ray`) or user-specified picker (`Layer::pick`)

## Usage

1. Instantiate a `Layer<Index, ID>`
    * `Index` must be an instance of `SpatialIndex`; one of `Index32_2D`, `Index64_2D`, or `Index64_3D`
    * `ID` may be any user-specified type which satisfies the `ObjectID` trait (blanket implementation; includes primitive integral types)
2. Clear old data, if necessary, using `Layer::clear`
3. Append object bounds-ID pairs using `Layer::extend`
4. Retrieve potential collisions using `Layer::scan`

## Example

From the included sample program:

```rust
struct Collisions {
    system: broadphase::Layer<broadphase::Index32_2D, specs::world::Index>,
    collisions: Vec<(specs::Entity, specs::Entity, f32, Vector2<f32>)>,
}

// ...

self.system.clear();
self.system.extend(collision_config.bounds,
    (&entities, &positions, &radii).join()
        .map(|(ent, &pos, &Radius(r))| {
            let bounds = Bounds{
                min: Point2::new(pos.1.x - r, pos.1.y - r),
                max: Point2::new(pos.1.x + r, pos.1.y + r)};
            (bounds, ent.id())}));

// ...

self.collisions.clear();
self.collisions.extend(self.system.par_scan()
    .iter()
    .filter_map(|&(id0, id1)| {
        // ...narrow-phase...
    }));
```

## Development

### Generating Test Data

A program, `gen_test_data`, is included in the `utils/` directory to generate and visualize test data.  Two scripts are included to automate the test generation process (`tests/gen_test_scenes.py` and `tests/gen_validation_data.py`)

### Visualizing Test Data

To show test data, execute `cargo run show -i <test_data_path> --gui` from the `utils/` subdirectory.  Move with W/A/S/D + Mouse (holding LMB) and cycle through objects left/right square bracket keys (\[/\]).  Alternatively, specify a particular ID as a command-line parameter using `--select-id`.  All objects can be selected simultaneously by passing the `--select-all` command-line parameter or pressing the spacebar while the application is running.