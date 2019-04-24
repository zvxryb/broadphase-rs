extern crate broadphase;
extern crate broadphase_data;

#[macro_use]
extern crate lazy_static;

use broadphase_data::Scene;

use std::path::{Path, PathBuf};

lazy_static! {
    static ref DATA_PATH: PathBuf = {
        let mut path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        path.push("tests/data");
        path
    };
}

fn load_scene<P: AsRef<Path>>(rel_path: P) -> Scene {
    let mut path = DATA_PATH.clone();
    path.push(rel_path);
    Scene::load(path).expect("failed to load test scene")
}

#[test]
fn extend() {
    let mut input = load_scene("inputs/boxes-seed_0-d_1_1000-s_1_10-n_010000.br_scene");
    let validation = load_scene("validation/0_layer_unsorted.br_scene");

    input.layer.extend(
        input.system_bounds,
        input.object_bounds.iter().cloned());

    let actual   = input     .layer;
    let expected = validation.layer;

    if actual != expected {
        panic!("Layer::extend() produced unexpected results");
    }
}

fn is_sorted<Item: Ord, Iter: Iterator<Item = Item>>(mut iter: Iter) -> bool {
    iter.try_fold(None, |old, new| {
        if let Some(old) = old {
            if old <= new { Some(Some(new)) } else { None }
        } else {
            Some(Some(new))
        }
    }).is_some()
}

#[test]
fn sort() {
    let input = load_scene("validation/0_layer_unsorted.br_scene");
    let validation = load_scene("validation/1_layer_sorted.br_scene");

    let mut actual = input.layer;
    actual.sort();
    if !is_sorted(actual.iter()) {
        panic!("Layer::sort() produced unsorted output");
    }

    let expected = validation.layer;

    if actual != expected {
        panic!("Layer::sort() produced unexpected results");
    }
}

#[test]
fn par_sort() {
    let input = load_scene("validation/0_layer_unsorted.br_scene");
    let validation = load_scene("validation/1_layer_sorted.br_scene");

    let mut actual = input.layer;
    actual.par_sort();
    if !is_sorted(actual.iter()) {
        panic!("Layer::par_sort() produced unsorted output");
    }

    let expected = validation.layer;

    if actual != expected {
        panic!("Layer::sort() produced unexpected results");
    }
}