extern crate broadphase;

extern crate bincode;
extern crate cgmath;
extern crate rand;
extern crate rand_chacha;

#[macro_use]
extern crate clap;

#[macro_use]
extern crate serde;

use broadphase::Bounds;
use cgmath::{Point3, Vector3};

use rand::prelude::*;

use std::fs::File;

type ID = u32;

#[derive(Deserialize, Serialize)]
struct Scene {
    system_bounds: Bounds<Point3<f32>>,
    object_bounds: Vec<(Bounds<Point3<f32>>, ID)>
}

trait Command {
    fn name() -> &'static str;
    fn init() -> clap::App<'static, 'static>;
    fn exec(args: &clap::ArgMatches);
}

struct GenBoxes {}
impl Command for GenBoxes {
    fn name() -> &'static str { "gen_boxes" }
    fn init() -> clap::App<'static, 'static> {
        use clap::Arg;
        clap::SubCommand::with_name(Self::name())
            .about("generate a scene with multiple AABBs")
            .arg(Arg::with_name("seed")
                .long("seed")
                .value_name("NUMBER")
                .help("initial state for the random number generator"))
            .arg(Arg::with_name("count")
                .short("n")
                .long("count")
                .value_name("NUMBER")
                .required(true)
                .help("number of objects in the scene"))
            .arg(Arg::with_name("size_range")
                .short("s")
                .long("size_range")
                .value_names(&["MIN", "MAX"])
                .required(true)
                .help("size range for objects"))
            .arg(Arg::with_name("bounds")
                .short("b")
                .long("bounds")
                .value_names(&["X0", "Y0", "Z0", "X1", "Y1", "Z1"])
                .required(true)
                .help("system bounds"))
            .arg(Arg::with_name("out_path")
                .short("o")
                .long("out")
                .value_name("PATH")
                .required(true)
                .help("where to write output"))
    }

    fn exec(args: &clap::ArgMatches) {
        let n = value_t!(args, "count", usize)
            .expect("failed to get count");
        let size_range = values_t!(args, "size_range", f32)
            .expect("failed to get size_range");
        let system_bounds = values_t!(args, "bounds", f32)
            .expect("failed to get bounds");

        let system_bounds = Bounds{
            min: Point3::new(
                system_bounds[0],
                system_bounds[1],
                system_bounds[2]),
            max: Point3::new(
                system_bounds[3],
                system_bounds[4],
                system_bounds[5])};

        let mut rng = rand_chacha::ChaChaRng::seed_from_u64(
            value_t!(args, "seed", u64).unwrap_or(0));
        let mut bounds: Vec<(Bounds<Point3<f32>>, ID)> = Vec::with_capacity(n);
        bounds.extend((0..n)
            .map(|id| {
                let size = Vector3::new(0f32, 0f32, 0f32)
                    .map(|_| rng.gen_range(size_range[0], size_range[1]));

                let mut min = Point3::new(0f32, 0f32, 0f32);
                for i in 0..3 {
                    min[i] = rng.gen_range(
                        system_bounds.min[i],
                        system_bounds.max[i] - size[i])
                }

                let max = min + size;

                (Bounds{ min: min, max: max }, id as u32)
            }));

        let f = File::create(args.value_of("out_path")
            .expect("no output path specified"))
            .expect("failed to open output for writing");

        bincode::serialize_into(f, &Scene{
            system_bounds: system_bounds,
            object_bounds: bounds
        }).expect("failed to write output");
    }
}

struct ShowBoxes {}
impl Command for ShowBoxes {
    fn name() -> &'static str { "show_boxes" }
    fn init() -> clap::App<'static, 'static> {
        use clap::Arg;
        clap::SubCommand::with_name(Self::name())
            .about("show a scene with multiple AABBs")
            .arg(Arg::with_name("in_path")
                .short("i")
                .long("in")
                .value_name("PATH")
                .required(true)
                .help("path to a scene generated with gen_boxes"))
    }
    fn exec(args: &clap::ArgMatches) {
        let f = File::open(args.value_of("in_path")
            .expect("no input path specified"))
            .expect("failed to open input for reading");

        let scene: Scene = bincode::deserialize_from(f)
            .expect("failed to read input");

        println!("system_bounds: {:?}", scene.system_bounds);
        println!("object_bounds:");
        for (bounds, id) in scene.object_bounds {
            println!("\tid: {:5}, bounds: <{:6.3}, {:6.3}, {:6.3}> <{:6.3}, {:6.3}, {:6.3}>",
                id,
                bounds.min.x,
                bounds.min.y,
                bounds.min.z,
                bounds.max.x,
                bounds.max.y,
                bounds.max.z);
        }
    }
}

macro_rules! app_cmds {
    (app $app: expr; $(cmd $cmd: ident)*) => {
        {
            let mut app = $app as clap::App;
            $(
                app = app.subcommand(<$cmd as Command>::init());
            )*
            let matches = app.get_matches();
            $(
                if let Some(matches) = matches.subcommand_matches(<$cmd as Command>::name()) {
                    <$cmd as Command>::exec(matches);
                }
            )*
        }
    };
}

fn main() {
    app_cmds!{
        app clap::App::new("gen_test_data")
            .version("0.1.0");
        cmd GenBoxes
        cmd ShowBoxes
    };
}