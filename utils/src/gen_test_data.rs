extern crate broadphase;
extern crate broadphase_data;

extern crate cgmath;
extern crate rand;
extern crate rand_chacha;

#[macro_use]
extern crate clap;

#[macro_use]
extern crate glium;

use broadphase::Bounds;
use broadphase_data::{ID, Scene};
use cgmath::{Deg, Matrix4, Point3, Quaternion, Rad, Vector3};
use glium::glutin;

use cgmath::prelude::*;
use rand::prelude::*;

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

                (Bounds{ min, max }, id as u32)
            }));

        let scene = Scene{
            system_bounds,
            object_bounds: bounds
        };

        scene.save(args.value_of("out_path").expect("no output path specified"))
            .expect("failed to write output");
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
            .arg(Arg::with_name("cli")
                .long("cli")
                .help("dump output to terminal (default)"))
            .arg(Arg::with_name("gui")
                .long("gui")
                .conflicts_with("cli")
                .help("show 3D visualization"))
    }
    fn exec(args: &clap::ArgMatches) {
        let scene = Scene::load(args.value_of("in_path").expect("no input path specified"))
            .expect("failed to read input");

        if args.is_present("gui") {
            let mut events_loop = glutin::EventsLoop::new();
            let window_builder = glutin::WindowBuilder::new()
                .with_title("Broadphase Util: Show Boxes")
                .with_resizable(true);
            let context = glutin::ContextBuilder::new()
                .with_gl(glutin::GlRequest::Specific(glutin::Api::OpenGl, (4, 5)))
                .with_gl_profile(glutin::GlProfile::Core)
                .with_multisampling(4);
            let display = glium::Display::new(window_builder, context, &events_loop)
                .expect("failed to create display");

            let box_outline_vbo = {
                #[derive(Copy, Clone)]
                struct Vertex {
                    position: [f32; 3]
                }

                implement_vertex!(Vertex, position);

                glium::VertexBuffer::immutable(&display, &[
                    Vertex{ position: [0f32, 0f32, 0f32] }, Vertex{ position: [1f32, 0f32, 0f32] },
                    Vertex{ position: [0f32, 0f32, 1f32] }, Vertex{ position: [1f32, 0f32, 1f32] },
                    Vertex{ position: [0f32, 1f32, 0f32] }, Vertex{ position: [1f32, 1f32, 0f32] },
                    Vertex{ position: [0f32, 1f32, 1f32] }, Vertex{ position: [1f32, 1f32, 1f32] },
                    Vertex{ position: [0f32, 0f32, 0f32] }, Vertex{ position: [0f32, 1f32, 0f32] },
                    Vertex{ position: [0f32, 0f32, 1f32] }, Vertex{ position: [0f32, 1f32, 1f32] },
                    Vertex{ position: [1f32, 0f32, 0f32] }, Vertex{ position: [1f32, 1f32, 0f32] },
                    Vertex{ position: [1f32, 0f32, 1f32] }, Vertex{ position: [1f32, 1f32, 1f32] },
                    Vertex{ position: [0f32, 0f32, 0f32] }, Vertex{ position: [0f32, 0f32, 1f32] },
                    Vertex{ position: [1f32, 0f32, 0f32] }, Vertex{ position: [1f32, 0f32, 1f32] },
                    Vertex{ position: [0f32, 1f32, 0f32] }, Vertex{ position: [0f32, 1f32, 1f32] },
                    Vertex{ position: [1f32, 1f32, 0f32] }, Vertex{ position: [1f32, 1f32, 1f32] }
                ]).expect("failed to create vbo")
            };

            let box_solid_vbo = {
                #[derive(Copy, Clone)]
                struct Vertex {
                    position: [f32; 3],
                    normal  : [f32; 3]
                }

                implement_vertex!(Vertex, position, normal);

                glium::VertexBuffer::immutable(&display, &[
                    Vertex{ position: [0f32, 0f32, 0f32], normal: [ 0f32,  0f32, -1f32] }, Vertex{ position: [0f32, 1f32, 0f32], normal: [ 0f32,  0f32, -1f32] }, Vertex{ position: [1f32, 1f32, 0f32], normal: [ 0f32,  0f32, -1f32] },
                    Vertex{ position: [0f32, 0f32, 0f32], normal: [ 0f32,  0f32, -1f32] }, Vertex{ position: [1f32, 1f32, 0f32], normal: [ 0f32,  0f32, -1f32] }, Vertex{ position: [1f32, 0f32, 0f32], normal: [ 0f32,  0f32, -1f32] },
                    Vertex{ position: [0f32, 0f32, 1f32], normal: [ 0f32,  0f32,  1f32] }, Vertex{ position: [1f32, 1f32, 1f32], normal: [ 0f32,  0f32,  1f32] }, Vertex{ position: [0f32, 1f32, 1f32], normal: [ 0f32,  0f32,  1f32] },
                    Vertex{ position: [0f32, 0f32, 1f32], normal: [ 0f32,  0f32,  1f32] }, Vertex{ position: [1f32, 0f32, 1f32], normal: [ 0f32,  0f32,  1f32] }, Vertex{ position: [1f32, 1f32, 1f32], normal: [ 0f32,  0f32,  1f32] },
                    Vertex{ position: [0f32, 0f32, 0f32], normal: [ 0f32, -1f32,  0f32] }, Vertex{ position: [1f32, 0f32, 1f32], normal: [ 0f32, -1f32,  0f32] }, Vertex{ position: [0f32, 0f32, 1f32], normal: [ 0f32, -1f32,  0f32] },
                    Vertex{ position: [0f32, 0f32, 0f32], normal: [ 0f32, -1f32,  0f32] }, Vertex{ position: [1f32, 0f32, 0f32], normal: [ 0f32, -1f32,  0f32] }, Vertex{ position: [1f32, 0f32, 1f32], normal: [ 0f32, -1f32,  0f32] },
                    Vertex{ position: [0f32, 1f32, 0f32], normal: [ 0f32,  1f32,  0f32] }, Vertex{ position: [0f32, 1f32, 1f32], normal: [ 0f32,  1f32,  0f32] }, Vertex{ position: [1f32, 1f32, 1f32], normal: [ 0f32,  1f32,  0f32] },
                    Vertex{ position: [0f32, 1f32, 0f32], normal: [ 0f32,  1f32,  0f32] }, Vertex{ position: [1f32, 1f32, 1f32], normal: [ 0f32,  1f32,  0f32] }, Vertex{ position: [1f32, 1f32, 0f32], normal: [ 0f32,  1f32,  0f32] },
                    Vertex{ position: [0f32, 0f32, 0f32], normal: [-1f32,  0f32,  0f32] }, Vertex{ position: [0f32, 0f32, 1f32], normal: [-1f32,  0f32,  0f32] }, Vertex{ position: [0f32, 1f32, 1f32], normal: [-1f32,  0f32,  0f32] },
                    Vertex{ position: [0f32, 0f32, 0f32], normal: [-1f32,  0f32,  0f32] }, Vertex{ position: [0f32, 1f32, 1f32], normal: [-1f32,  0f32,  0f32] }, Vertex{ position: [0f32, 1f32, 0f32], normal: [-1f32,  0f32,  0f32] },
                    Vertex{ position: [1f32, 0f32, 0f32], normal: [ 1f32,  0f32,  0f32] }, Vertex{ position: [1f32, 1f32, 1f32], normal: [ 1f32,  0f32,  0f32] }, Vertex{ position: [1f32, 0f32, 1f32], normal: [ 1f32,  0f32,  0f32] },
                    Vertex{ position: [1f32, 0f32, 0f32], normal: [ 1f32,  0f32,  0f32] }, Vertex{ position: [1f32, 1f32, 0f32], normal: [ 1f32,  0f32,  0f32] }, Vertex{ position: [1f32, 1f32, 1f32], normal: [ 1f32,  0f32,  0f32] }
                ]).expect("failed to create vbo")
            };

            let box_instances_vbo = {
                #[derive(Copy, Clone)]
                struct Vertex {
                    aabb_min: [f32; 3],
                    aabb_max: [f32; 3],
                    color: [f32; 3]
                }

                implement_vertex!(Vertex, aabb_min, aabb_max, color);

                glium::VertexBuffer::immutable(&display,
                    scene.object_bounds.iter()
                        .map(|&(bounds, _)| {
                            Vertex{
                                aabb_min: [bounds.min.x, bounds.min.y, bounds.min.z],
                                aabb_max: [bounds.max.x, bounds.max.y, bounds.max.z],
                                color: [1f32; 3]}})
                        .collect::<Vec<Vertex>>()
                        .as_slice())
                    .expect("failed to create vbo")
            };

            #[derive(Debug)]
            struct Camera {
                position: Point3<f32>,
                orientation: Quaternion<f32>,
                fov_y: Rad<f32>,
                aspect_ratio: f32,
                near: f32,
                far: f32
            }

            let mut camera = {
                let size = scene.system_bounds.sizef();
                let near = 0.0001f32 * size.magnitude();
                let far = near + size.magnitude();
                Camera{
                    position: scene.system_bounds.max,
                    orientation: Quaternion::from_arc(-Vector3::unit_z(), -size.normalize(), None),
                    fov_y: Deg(90f32).into(),
                    aspect_ratio: 1f32,
                    near,
                    far
                }
            };

            #[derive(Copy, Clone)]
            struct Transforms {
                view_proj: [[f32; 4]; 4]
            }

            implement_uniform_block!(Transforms, view_proj);

            let mut transforms = Transforms{ view_proj: [[0f32; 4]; 4] };
            let transforms_ubo = glium::uniforms::UniformBuffer::<Transforms>::empty_persistent(&display).unwrap();

            let program_solid = glium::Program::from_source(&display,
                r#"
                    #version 450 core

                    in vec3 position;
                    in vec3 normal;
                    in vec3 aabb_min;
                    in vec3 aabb_max;
                    in vec3 color;

                    out vec3 v_color;

                    uniform transforms {
                        mat4 view_proj;
                    };

                    const vec3 light_dir = vec3(0.802, 0.535, 0.267);

                    void main() {
                        vec4 global = vec4(mix(aabb_min, aabb_max, position), 1.0);
                        v_color = color * (0.4 * dot(light_dir, normal) + 0.6);
                        gl_Position = view_proj * global;
                    }
                "#,
                r#"
                    #version 450 core

                    in vec3 v_color;

                    out vec4 f_color;

                    void main() {
                        f_color = vec4(v_color, 1.0);
                    }
                "#,
                None)
                .expect("failed to compile shader");

            let program_outline = glium::Program::from_source(&display,
                r#"
                    #version 450 core

                    in vec3 position;
                    in vec3 normal;
                    in vec3 aabb_min;
                    in vec3 aabb_max;

                    out vec3 v_color;

                    uniform transforms {
                        mat4 view_proj;
                    };

                    uniform vec3 color;

                    void main() {
                        vec4 global = vec4(mix(aabb_min, aabb_max, position), 1.0);
                        v_color = vec3(0.0, 0.0, 0.0);
                        gl_Position = view_proj * global;
                    }
                "#,
                r#"
                    #version 450 core

                    in vec3 v_color;

                    out vec4 f_color;

                    void main() {
                        f_color = vec4(v_color, 1.0);
                    }
                "#,
                None)
                .expect("failed to compile shader");

            let mut draw = |display: &glium::Display, camera: &Camera| {

                let rot: Matrix4<f32> = camera.orientation.invert().into();
                let offs = Matrix4::from_translation(Point3::new(0f32, 0f32, 0f32) - camera.position);
                let view = rot * offs;
                let proj = cgmath::perspective(
                    camera.fov_y,
                    camera.aspect_ratio,
                    camera.near,
                    camera.far);

                transforms.view_proj = (proj * view).into();
                transforms_ubo.write(&transforms);

                use glium::Surface;

                let mut frame = display.draw();
                frame.clear_color_and_depth((0f32, 0.1f32, 0.2f32, 1f32), 1f32);

                {
                    let params = glium::DrawParameters{
                        backface_culling: glium::BackfaceCullingMode::CullClockwise,
                        depth: glium::Depth{
                            test: glium::DepthTest::IfLess,
                            write: true,
                            .. Default::default()
                        },
                        .. Default::default()
                    };
                    let uniforms = uniform!{ transforms: &transforms_ubo };
                    frame.draw(
                        (&box_solid_vbo, box_instances_vbo.per_instance().unwrap()),
                        glium::index::NoIndices(glium::index::PrimitiveType::TrianglesList),
                        &program_solid, &uniforms, &params)
                        .expect("failed to draw frame");
                }

                {
                    let params = glium::DrawParameters{
                        depth: glium::Depth{
                            test: glium::DepthTest::IfLessOrEqual,
                            write: true,
                            .. Default::default()
                        },
                        .. Default::default()
                    };
                    let uniforms = uniform!{
                        transforms: &transforms_ubo,
                        color: [1f32, 0f32, 0f32]
                    };
                    frame.draw(
                        (&box_outline_vbo, box_instances_vbo.per_instance().unwrap()),
                        glium::index::NoIndices(glium::index::PrimitiveType::LinesList),
                        &program_outline, &uniforms, &params)
                        .expect("failed to draw frame");
                }

                frame.finish().unwrap();
            };

            let mut running = true;
            let mut time = std::time::Instant::now();
            let mut mouse_grab: Option<glutin::MouseButton> = None;
            let mut move_forward = false;
            let mut move_back = false;
            let mut move_left = false;
            let mut move_right = false;
            while running {
                let mut redraw = false;
                events_loop.poll_events(|event| {
                    use glutin::{DeviceEvent, Event, WindowEvent};
                    match event {
                        Event::DeviceEvent{event, ..} => {
                            match event {
                                DeviceEvent::Key(glutin::KeyboardInput{state, virtual_keycode, ..}) => {
                                    let move_dir = match virtual_keycode {
                                        Some(glutin::VirtualKeyCode::W) => Some(&mut move_forward),
                                        Some(glutin::VirtualKeyCode::A) => Some(&mut move_left),
                                        Some(glutin::VirtualKeyCode::S) => Some(&mut move_back),
                                        Some(glutin::VirtualKeyCode::D) => Some(&mut move_right),
                                        _ => None
                                    };

                                    if let Some(move_dir) = move_dir {
                                        *move_dir = state == glutin::ElementState::Pressed;
                                    }
                                }
                                DeviceEvent::MouseMotion{delta: (dx, dy)} => {
                                    if mouse_grab.is_some() {
                                        const DEG_PER_PX: f32 = 0.2;
                                        let rot_x = -(dy as f32) * DEG_PER_PX / 180f32;
                                        let rot_y = -(dx as f32) * DEG_PER_PX / 180f32;
                                        let rot_w = 1f32 - (rot_x.powi(2) + rot_y.powi(2)).sqrt();
                                        camera.orientation = (camera.orientation * Quaternion::new(rot_w, rot_x, rot_y, 0f32)).normalize();
                                        redraw = true;
                                    }
                                }
                                _ => {}
                            }
                        }
                        Event::WindowEvent{event, ..} => {
                            match event {
                                WindowEvent::CloseRequested => {
                                    running = false
                                }
                                WindowEvent::MouseInput{state, button, ..} => {
                                    if mouse_grab.map_or(true, |b| b == button) {
                                        let is_pressed = match state {
                                            glutin::ElementState::Pressed => true,
                                            glutin::ElementState::Released => false
                                        };
                                        if mouse_grab.is_some() != is_pressed {
                                            let window = display.gl_window();
                                            window.grab_cursor(is_pressed).unwrap();
                                            window.hide_cursor(is_pressed);
                                        }
                                        mouse_grab = if is_pressed { Some(button) } else { None };
                                    }
                                }
                                WindowEvent::Resized(size) => {
                                    camera.aspect_ratio = (size.width / size.height) as f32;
                                    redraw = true;
                                }
                                _ => {}
                            }
                        }
                        _ => {}
                    }
                });

                const MAX_FPS: i32 = 200;
                const MIN_DURATION_NS: i32 = 1_000_000_000 / MAX_FPS;
                let remaining_duration_ns = MIN_DURATION_NS - (time.elapsed().subsec_nanos() as i32);
                if remaining_duration_ns > 0 {
                    std::thread::sleep(std::time::Duration::from_nanos(remaining_duration_ns as u64));
                }

                if move_forward || move_back || move_left || move_right {
                    const SPEED: f32 = 0.1f32;

                    let mut move_local = Vector3::new(0f32, 0f32, 0f32);
                    if move_forward { move_local += Vector3::new( 0f32, 0f32, -1f32); }
                    if move_back    { move_local += Vector3::new( 0f32, 0f32,  1f32); }
                    if move_left    { move_local += Vector3::new(-1f32, 0f32,  0f32); }
                    if move_right   { move_local += Vector3::new( 1f32, 0f32,  0f32); }

                    move_local *= (time.elapsed().subsec_nanos() as f32 / 1_000_000_000f32)
                        * SPEED * scene.system_bounds.sizef().magnitude();

                    let move_global = camera.orientation.rotate_vector(move_local);
                    camera.position += move_global;

                    redraw = true;
                }

                time = std::time::Instant::now();

                if redraw {
                    draw(&display, &camera);
                }
            }
        } else {
            println!("system_bounds: {:?}", scene.system_bounds);
            println!("object_bounds:");
            for &(bounds, id) in &scene.object_bounds {
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