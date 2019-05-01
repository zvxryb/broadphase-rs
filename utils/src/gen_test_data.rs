extern crate broadphase;
extern crate broadphase_data;

extern crate cgmath;
extern crate rand;
extern crate rand_chacha;

#[macro_use]
extern crate clap;

#[macro_use]
extern crate glium;

use broadphase::{Bounds, Layer};
use broadphase_data::{Index, ID, Scene};
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
                .help("system bounds"))
            .arg(Arg::with_name("density")
                .short("d")
                .long("density")
                .value_name("DENSITY")
                .help("number of boxes per unit^3"))
            .arg(Arg::with_name("out_path")
                .short("o")
                .long("out")
                .value_name("PATH")
                .required(true)
                .help("where to write output"))
    }

    fn exec(args: &clap::ArgMatches) {
        let size_range = values_t!(args, "size_range", f32)
            .expect("failed to get size_range");

        let count = args.value_of("count").map(|s|
            s.parse::<usize>()
                .expect("failed to parse count"));

        let density = args.value_of("density").map(|s|
            s.parse::<f32>()
                .expect("failed to parse density"));

        let system_bounds = args.values_of("bounds").map(|mut values| {
            let mut next = || {
                values.next()
                    .expect("too few arguments for system bounds")
                    .parse::<f32>()
                    .expect("failed to parse bounds")
            };
            Bounds{
                min: Point3::new(
                    next(),
                    next(),
                    next()),
                max: Point3::new(
                    next(),
                    next(),
                    next())}
        });

        let avg_box_size = (size_range[0] + size_range[1]) / 2f32;

        let count = if let Some(count) = count {
            count
        } else {
            let density = density
                .expect("calculation of count requires density to be known");
            let system_size = system_bounds
                .expect("calculation of count requires bounds to be known")
                .sizef();
            let volume = (system_size.sub_element_wise(avg_box_size)).product();
            (density * volume) as usize
        };

        let system_bounds = if let Some(bounds) = system_bounds {
            bounds
        } else {
            let density = density
                .expect("calculation of count requires density to be known");
            let volume = count as f32 / density;
            let linear_size = volume.cbrt() + avg_box_size;
            Bounds{
                min: Point3::new(0f32, 0f32, 0f32),
                max: Point3::new(
                    linear_size,
                    linear_size,
                    linear_size)}
        };

        {
            let size = system_bounds.sizef();
            if size.x < size_range[1] ||
               size.y < size_range[1] ||
               size.z < size_range[1]
            {
                panic!("object size larger than system bounds; reduce object size, increase system bounds, or reduce density");
            }
        }

        let mut rng = rand_chacha::ChaChaRng::seed_from_u64(
            value_t!(args, "seed", u64).unwrap_or(0));
        let mut bounds: Vec<(Bounds<Point3<f32>>, ID)> = Vec::with_capacity(count);
        bounds.extend((0..count)
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
            object_bounds: bounds,
            layer: Default::default(),
            collisions: Default::default(),
            hits: Default::default(),
            nearest: Default::default()
        };

        scene.save(args.value_of("out_path").expect("no output path specified"))
            .expect("failed to write output");
    }
}

struct ShowBoxes {}
impl Command for ShowBoxes {
    fn name() -> &'static str { "show" }
    fn init() -> clap::App<'static, 'static> {
        use clap::Arg;
        clap::SubCommand::with_name(Self::name())
            .about("show a scene")
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
            .arg(Arg::with_name("select_id")
                .long("select-id")
                .requires("gui")
                .value_name("ID")
                .help("select by ID"))
            .arg(Arg::with_name("select_all")
                .long("select-all")
                .requires("gui")
                .conflicts_with("select_id")
                .help("select all"))
    }
    fn exec(args: &clap::ArgMatches) {
        let scene = Scene::load(args.value_of("in_path").expect("no input path specified"))
            .expect("failed to read input");

        let ids = {
            let mut ids: Vec<ID> = scene.object_bounds.iter()
                .map(|&(_, id)| id)
                .collect();
            ids.sort();
            ids
        };

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

            #[derive(Copy, Clone)]
            enum Selection {
                None,
                All,
                ID(ID)
            }

            impl Selection {
                fn cycle(&mut self, ids: &Vec<ID>, offset: i32) {
                    let mut id = if let &mut Selection::ID(id) = self { id } else { 0 };
                    let mut i = ids.binary_search(&id).unwrap();
                    let n = ids.len();
                    if offset < 0 {
                        let offset = -offset as usize;
                        i = (n + i - offset) % n;
                    } else {
                        i = (i + (offset as usize)) % n;
                    }
                    id = ids[i];
                    *self = Selection::ID(id);
                }
            }

            let mut selection = if let Some(id) = args.value_of("select_id") {
                Selection::ID(id.parse::<ID>().unwrap())
            } else if args.is_present("select_all") {
                Selection::All
            } else {
                Selection::None
            };

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

            #[derive(Copy, Clone)]
            struct InstanceData {
                aabb_min: [f32; 3],
                aabb_max: [f32; 3],
                fill_color: [f32; 4],
                edge_color: [f32; 4]
            }
            implement_vertex!(InstanceData, aabb_min, aabb_max, fill_color, edge_color);

            struct InstanceBuffer {
                vbo: glium::VertexBuffer<InstanceData>,
                count: usize
            }

            impl InstanceBuffer {
                fn resize(&mut self, display: &glium::Display, count: usize) {
                    if count > self.vbo.len() {
                        self.vbo = glium::VertexBuffer::<InstanceData>::empty_persistent(display, count)
                            .expect("failed to create vbo");
                    }
                    self.count = count;
                }

                fn slice(&self) -> glium::vertex::VertexBufferSlice<'_, InstanceData> {
                    self.vbo.slice(0..self.count).unwrap()
                }
            }

            fn update_instance_data(
                display: &glium::Display,
                buffer: &mut InstanceBuffer,
                scene: &Scene,
                selection: Selection)
            {
                let is_selected = |id| match selection {
                    Selection::ID(id_) => id == id_,
                    Selection::All => true,
                    Selection::None => false,
                };
                let collisions: Vec<ID> = scene.collisions.iter()
                    .filter_map(|&(id0, id1)|
                        if is_selected(id0) {
                            Some(id1)
                        } else if is_selected(id1) {
                            Some(id0)
                        } else {
                            None
                        })
                    .collect();
                let is_collided = |id| collisions.iter().any(|&id_| id == id_);
                let mut data: Vec<InstanceData> = Vec::with_capacity(buffer.vbo.len());
                for &(bounds, id) in &scene.object_bounds {
                    let (fill_color, edge_color) = if is_selected(id) {
                        ([0.5f32, 1f32, 0f32, 1f32], [0f32, 0f32, 0f32, 1f32])
                    } else if is_collided(id) {
                        ([1f32, 0.5f32, 0f32, 1f32], [0f32, 0f32, 0f32, 1f32])
                    } else if let Selection::None = selection {
                        ([1f32, 1f32, 1f32, 1f32], [0f32, 0f32, 0f32, 1f32])
                    } else {
                        ([1f32, 1f32, 1f32, 0.5f32], [0f32, 0f32, 0f32, 0.5f32])
                    };
                    data.push(InstanceData{
                        aabb_min: [bounds.min.x, bounds.min.y, bounds.min.z],
                        aabb_max: [bounds.max.x, bounds.max.y, bounds.max.z],
                        fill_color,
                        edge_color});
                }
                for &(index, id) in scene.layer.iter() {
                    use broadphase::SystemBounds;
                    if !is_selected(id) { continue; }
                    let local: Bounds<_> = index.into();
                    let global = scene.system_bounds.to_global(local);
                    data.push(InstanceData{
                        aabb_min: [global.min.x, global.min.y, global.min.z],
                        aabb_max: [global.max.x, global.max.y, global.max.z],
                        fill_color: [1f32, 1f32, 1f32, 0f32],
                        edge_color: [0f32, 0.5f32, 1f32, 1f32]});
                }
                buffer.resize(display, data.len());
                buffer.slice().write(data.as_slice());
            }

            let mut box_instances_buffer = InstanceBuffer{
                vbo: glium::VertexBuffer::<InstanceData>::empty_persistent(&display, scene.object_bounds.len())
                    .expect("failed to create vbo"),
                count: 0};
            update_instance_data(&display, &mut box_instances_buffer, &scene, selection);

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
                    fov_y: Deg(60f32).into(),
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
                    in vec4 fill_color;
                    in vec4 edge_color;

                    out vec4 v_color;

                    uniform transforms {
                        mat4 view_proj;
                    };

                    const vec3 light_dir = vec3(0.802, 0.535, 0.267);

                    void main() {
                        vec4 global = vec4(mix(aabb_min, aabb_max, position), 1.0);
                        v_color = vec4(fill_color.rgb * (0.4 * dot(light_dir, normal) + 0.6), fill_color.a);
                        gl_Position = view_proj * global;
                    }
                "#,
                r#"
                    #version 450 core

                    in vec4 v_color;

                    out vec4 f_color;

                    void main() {
                        int mask = 0x00000000;
                        if (v_color.a > 0.125) { mask |= 0x11111111; }
                        if (v_color.a > 0.375) { mask |= 0x22222222; }
                        if (v_color.a > 0.625) { mask |= 0x44444444; }
                        if (v_color.a > 0.875) { mask |= 0x88888888; }
                        gl_SampleMask[0] = mask;
                        f_color = vec4(v_color.rgb, 1.0);
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
                    in vec4 fill_color;
                    in vec4 edge_color;

                    out vec4 v_color;

                    uniform transforms {
                        mat4 view_proj;
                    };

                    uniform vec3 color;

                    void main() {
                        vec4 global = vec4(mix(aabb_min, aabb_max, position), 1.0);
                        v_color = edge_color;
                        gl_Position = view_proj * global;
                    }
                "#,
                r#"
                    #version 450 core

                    in vec4 v_color;

                    out vec4 f_color;

                    void main() {
                        int mask = 0x00000000;
                        if (v_color.a > 0.125) { mask |= 0x11111111; }
                        if (v_color.a > 0.375) { mask |= 0x22222222; }
                        if (v_color.a > 0.625) { mask |= 0x44444444; }
                        if (v_color.a > 0.875) { mask |= 0x88888888; }
                        gl_SampleMask[0] = mask;
                        f_color = vec4(v_color.rgb, 1.0);
                    }
                "#,
                None)
                .expect("failed to compile shader");

            let mut draw = |display: &glium::Display, camera: &Camera, instance_buffer: &InstanceBuffer| {

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
                        (&box_solid_vbo, instance_buffer.slice().per_instance().unwrap()),
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
                        (&box_outline_vbo, instance_buffer.slice().per_instance().unwrap()),
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
                                DeviceEvent::Key(glutin::KeyboardInput{state, virtual_keycode, ..}) =>
                                    match virtual_keycode {
                                        Some(glutin::VirtualKeyCode::LBracket) |
                                        Some(glutin::VirtualKeyCode::RBracket) =>
                                            if state == glutin::ElementState::Pressed {
                                                let offset = match virtual_keycode {
                                                    Some(glutin::VirtualKeyCode::LBracket) => -1,
                                                    Some(glutin::VirtualKeyCode::RBracket) =>  1,
                                                    _ => panic!()
                                                };
                                                selection.cycle(&ids, offset);
                                                update_instance_data(&display, &mut box_instances_buffer, &scene, selection);
                                                redraw = true;
                                            }
                                        Some(glutin::VirtualKeyCode::Space) =>
                                            if state == glutin::ElementState::Pressed {
                                                selection = match selection {
                                                    Selection::All => Selection::None,
                                                    _ => Selection::All
                                                };
                                                update_instance_data(&display, &mut box_instances_buffer, &scene, selection);
                                                redraw = true;
                                            }
                                        Some(glutin::VirtualKeyCode::W) |
                                        Some(glutin::VirtualKeyCode::A) |
                                        Some(glutin::VirtualKeyCode::S) |
                                        Some(glutin::VirtualKeyCode::D) => {
                                            let move_dir = match virtual_keycode {
                                                Some(glutin::VirtualKeyCode::W) => &mut move_forward,
                                                Some(glutin::VirtualKeyCode::A) => &mut move_left,
                                                Some(glutin::VirtualKeyCode::S) => &mut move_back,
                                                Some(glutin::VirtualKeyCode::D) => &mut move_right,
                                                _ => panic!()
                                            };
                                            *move_dir = state == glutin::ElementState::Pressed;
                                        }
                                        _ => {}
                                    }
                                DeviceEvent::MouseMotion{delta: (dx, dy)} =>
                                    if mouse_grab.is_some() {
                                        const DEG_PER_PX: f32 = 0.2;
                                        let rot_x = -(dy as f32) * DEG_PER_PX / 180f32;
                                        let rot_y = -(dx as f32) * DEG_PER_PX / 180f32;
                                        let rot_w = 1f32 - (rot_x.powi(2) + rot_y.powi(2)).sqrt();
                                        camera.orientation = (camera.orientation * Quaternion::new(rot_w, rot_x, rot_y, 0f32)).normalize();
                                        redraw = true;
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
                    draw(&display, &camera, &box_instances_buffer);
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
            println!("layer:");
            for &(index, id) in scene.layer.iter() {
                println!("\tid: {:5}, index: {:?}", id, index);
            }
            println!("collisions:");
            for ids in scene.collisions {
                println!("\tids: {:5}, {:5}", ids.0, ids.1);
            }
        }
    }
}

struct GenValidationData {}
impl Command for GenValidationData {
    fn name() -> &'static str { "gen_validation_data" }
    fn init() -> clap::App<'static, 'static> {
        use clap::Arg;
        clap::SubCommand::with_name(Self::name())
            .about("generate validation data for testing")
            .arg(Arg::with_name("out_path")
                .short("o")
                .long("out")
                .value_name("DIR")
                .required(true)
                .help("where to write output"))
            .arg(Arg::with_name("in_path")
                .short("i")
                .long("in")
                .value_name("PATH")
                .required(true)
                .help("path to a scene generated with gen_boxes"))
    }

    fn exec(args: &clap::ArgMatches) {
        let input = Scene::load(args.value_of("in_path").unwrap())
            .expect("failed to load scene");
        let out_path = std::path::PathBuf::from(args.value_of("out_path").unwrap());
        let save_scene = |rel_path: &str, scene: &Scene| {
            let mut path = out_path.clone();
            path.push(rel_path);
            scene.save(path).expect("failed to save scene");
        };

        let mut scene = input.clone();
        scene.layer.extend(
            input.system_bounds,
            input.object_bounds.iter().cloned());
        save_scene("0_layer_unsorted.br_scene", &scene);

        scene.layer.sort();
        save_scene("1_layer_sorted.br_scene", &scene);

        {
            let mut scene = scene.clone();
            scene.collisions = scene.layer.scan().clone();
            save_scene("2_layer_collisions.br_scene", &scene);
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
        cmd GenValidationData
    };
}