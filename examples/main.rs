// mlodato, 20190318

extern crate zvxryb_broadphase as broadphase;
extern crate backtrace;
extern crate cgmath;
extern crate env_logger;
extern crate png;
extern crate rand;
extern crate specs;

#[macro_use]
extern crate glium;

use glium::glutin;

#[macro_use]
extern crate log;

#[macro_use(defer)]
extern crate scopeguard;

use cgmath::prelude::*;
use rand::prelude::*;
use specs::prelude::*;

use broadphase::Bounds;
use cgmath::{Point2, Vector2};
use std::alloc::{GlobalAlloc, Layout as AllocLayout, System as SystemAlloc};
use std::sync::atomic::{AtomicU32, AtomicUsize, Ordering as AtomicOrdering};
use std::time::{Duration, Instant};

enum AllocState {
    Normal,
    Dump,
    Alloc,
}

struct AllocLogger {
    count: AtomicUsize,
    state: AtomicU32,
}

impl AllocLogger {
    fn begin_dump(&self) {
        while let Err(val) = self.state.compare_exchange_weak(
            AllocState::Normal as u32,
            AllocState::Dump as u32,
            AtomicOrdering::SeqCst,
            AtomicOrdering::Relaxed
        ) { if val == AllocState::Dump as u32 { break; } }
    }

    fn end_dump(&self) {
        while let Err(val) = self.state.compare_exchange_weak(
            AllocState::Dump as u32,
            AllocState::Normal as u32,
            AtomicOrdering::SeqCst,
            AtomicOrdering::Relaxed
        ) { if val == AllocState::Normal as u32 { break; } }
    }

    fn clear_and_get_stats(&self) -> usize {
        self.count.swap(0, AtomicOrdering::Relaxed)
    }
}

unsafe impl GlobalAlloc for AllocLogger {
    #[inline(never)]
    unsafe fn alloc(&self, layout: AllocLayout) -> *mut u8 {
        let state = self.state.swap(AllocState::Alloc as u32, AtomicOrdering::SeqCst);
        defer!{ 
            if state != AllocState::Alloc as u32 {
                self.state.store(state, AtomicOrdering::SeqCst);
            }
        }

        if state != AllocState::Alloc as u32 {
            self.count.fetch_add(1, AtomicOrdering::Relaxed);
        }

        if state == AllocState::Dump as u32 {
            let bt = backtrace::Backtrace::new();
            trace!("{:?}", bt);
        }

        SystemAlloc.alloc(layout)
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: AllocLayout) {
        SystemAlloc.dealloc(ptr, layout);
    }
}

#[global_allocator]
static ALLOCATOR: AllocLogger = AllocLogger{
    count: AtomicUsize::new(0),
    state: AtomicU32::new(AllocState::Normal as u32),
};

struct Time {
    real: Instant,
    sim : Duration,
    draw: Duration,
    step: Duration,
}

impl Time {
    fn update(&mut self, step: Duration, max_delta: Duration) {
        let now   = Instant::now();
        let delta = now - self.real;
        self.real = now;
        self.draw += std::cmp::min(delta, max_delta);
        self.step = step;
    }

    fn step(&mut self) -> bool {
        if self.sim + self.step <= self.draw {
            self.sim += self.step;
            true
        } else { false }
    }

    fn remainder(&self) -> Duration {
        self.draw - self.sim
    }
}

impl Default for Time {
    fn default() -> Self {
        Self {
            real: Instant::now(),
            sim : Default::default(),
            draw: Default::default(),
            step: Default::default(),
        }
    }
}

struct CollisionSystemConfig {
    enabled: bool,
    dump_frame_allocs: bool,
    bounds: Bounds<Point2<f32>>
}

impl Default for CollisionSystemConfig {
    fn default() -> Self {
        Self{
            enabled: true,
            dump_frame_allocs: false,
            bounds: Bounds::new(
                Point2::new(0f32, 0f32),
                Point2::new(1f32, 1f32))}
    }
}

impl CollisionSystemConfig {
    fn bounds(w: u32, h: u32) -> Bounds<Point2<f32>> {
        const BORDER: f32 = 1f32;
        let scale = if w > h { w } else { h } as f32;
        let min = Point2::new(0f32, 0f32).sub_element_wise(BORDER);
        let max = min.add_element_wise(scale).add_element_wise(BORDER);
        Bounds::new(min, max)
    }

    fn from_screen_size((w, h): (u32, u32)) -> Self {
        Self{
            enabled: true,
            dump_frame_allocs: false,
            bounds: Self::bounds(w, h)}
    }

    fn update_bounds(&mut self, w: u32, h: u32) {
        self.bounds = Self::bounds(w, h);
    }
}

struct ScreenSize(u32, u32);

impl Default for ScreenSize {
    fn default() -> Self {
        Self(1, 1)
    }
}

impl From<(u32, u32)> for ScreenSize {
    fn from((w, h): (u32, u32)) -> Self {
        Self(w, h)
    }
}

impl From<glutin::dpi::PhysicalSize<u32>> for ScreenSize {
    fn from(size: glutin::dpi::PhysicalSize<u32>) -> Self {
        Self(size.width, size.height)
    }
}

struct BallCount(u32);

impl Default for BallCount {
    fn default() -> Self {
        Self(0)
    }
}

#[derive(Copy, Clone)]
struct Color(f32, f32, f32, f32);

impl Default for Color {
    fn default() -> Self {
        Self(1.0, 1.0, 1.0, 1.0)
    }
}

impl Into<[f32; 4]> for Color {
    fn into(self) -> [f32; 4] {
        [self.0, self.1, self.2, self.3]
    }
}

impl specs::Component for Color {
    type Storage = specs::VecStorage<Self>;
}

struct Lifetime(Duration);

impl specs::Component for Lifetime {
    type Storage = specs::VecStorage<Self>;
}

impl From<Duration> for Lifetime {
    fn from(expires: Duration) -> Self {
        Self(expires)
    }
}

#[derive(Copy, Clone)]
struct VerletPosition(Point2<f32>, Point2<f32>);

impl specs::Component for VerletPosition {
    type Storage = specs::VecStorage<Self>;
}

impl From<(f32, f32)> for VerletPosition {
    fn from(pos: (f32, f32)) -> Self {
        Self(pos.into(), pos.into())
    }
}

struct Radius(f32);

impl specs::Component for Radius {
    type Storage = specs::VecStorage<Self>;
}

impl From<f32> for Radius {
    fn from(radius: f32) -> Self {
        Self(radius)
    }
}

fn create_ball<T: specs::Builder>(
    builder: T,
    lifetime: Lifetime,
    position: VerletPosition,
    radius: Radius,
) -> specs::Entity {
    builder
        .with(lifetime)
        .with(position)
        .with(radius)
        .with(Color::default())
        .build()
}

struct Lifecycle;
impl<'a> specs::System<'a> for Lifecycle {
    #[allow(clippy::type_complexity)]
    type SystemData = (
        specs::Entities<'a>,
        specs::Read<'a, specs::LazyUpdate>,
        specs::Read<'a, Time>,
        specs::Read<'a, ScreenSize>,
        specs::Write<'a, BallCount>,
        specs::ReadStorage<'a, Lifetime>,
    );

    #[inline(never)]
    fn run(&mut self, data: Self::SystemData) {
        let (entities, lazy, time, screen_size, mut ball_count, lifetimes) = data;

        for (entity, &Lifetime(expires)) in (&entities, &lifetimes).join() {
            if expires < time.sim {
                entities.delete(entity).unwrap();
                ball_count.0 -= 1;
            }
        }

        const BALL_COUNT_MAX: u32 = 2500;
        const LIFETIME_MIN_MS: u32 = 10000;
        const LIFETIME_MAX_MS: u32 = 50000;
        for _ in 0..BALL_COUNT_MAX*time.step.subsec_millis()/LIFETIME_MIN_MS {
            if ball_count.0 >= BALL_COUNT_MAX {
                break;
            }
            let lifetime = Duration::from_millis(rand::thread_rng().gen_range(
                LIFETIME_MIN_MS as u64, LIFETIME_MAX_MS as u64));

            let r = rand::thread_rng().gen_range(0.5f32, 2.0f32).exp();

            let x0 = 0f32 + r;
            let x1 = screen_size.0 as f32 - 2f32 * r + x0;
            let x = rand::thread_rng().gen_range(x0, x1);

            let y = screen_size.1 as f32 + r;

            create_ball(
                lazy.create_entity(&entities),
                (time.sim + lifetime).into(),
                (x, y).into(),
                r.into(),
            );

            ball_count.0 += 1;
        }
    }
}

struct Kinematics;
impl<'a> specs::System<'a> for Kinematics {
    type SystemData = (
        specs::Read<'a, Time>,
        specs::WriteStorage<'a, VerletPosition>,
    );

    #[inline(never)]
    fn run(&mut self, data: Self::SystemData) {
        let (time, mut positions) = data;
        let dt = (time.step.as_secs() as f32) + (time.step.subsec_micros() as f32) / 1_000_000f32;
        let gravity = 50f32 * dt * dt;
        for mut pos in (&mut positions).join() {
            let mut pos_2 = pos.1 + (pos.1 - pos.0);
            pos_2.y -= gravity;
            pos.0 = pos.1;
            pos.1 = pos_2;
        }
        for mut pos in (&mut positions).join() {
            let velocity = pos.1 - pos.0;
            let speed = velocity.magnitude();
            const SPEED_LIMIT: f32 = 1.5f32;
            if speed > SPEED_LIMIT {
                pos.1 = pos.0 + SPEED_LIMIT * velocity / speed;
            }
        }
    }
}

struct Collisions {
    system: broadphase::Layer<broadphase::Index32_2D, specs::world::Index>,
    collisions: Vec<(specs::Entity, specs::Entity, f32, Vector2<f32>)>,
}

impl Collisions {
    fn new() -> Self {
        Self {
            system: broadphase::LayerBuilder::new()
                .with_min_depth(4)
                .build(),
            collisions: Vec::new(),
        }
    }
}

impl<'a> specs::System<'a> for Collisions {
    #[allow(clippy::type_complexity)]
    type SystemData = (
        specs::Entities<'a>,
        specs::Read<'a, ScreenSize>,
        specs::Write<'a, CollisionSystemConfig>,
        specs::WriteStorage<'a, VerletPosition>,
        specs::ReadStorage<'a, Radius>,
        specs::WriteStorage<'a, Color>
    );

    #[inline(never)]
    fn run(&mut self, data: Self::SystemData) {
        let (entities, screen_size, mut collision_config, mut positions, radii, mut colors) = data;

        for color in (&mut colors).join() {
            *color = Color(1f32, 0.5f32, 0f32, 1f32);
        }

        let start = Instant::now();

        if collision_config.enabled {
            ALLOCATOR.clear_and_get_stats();

            if collision_config.dump_frame_allocs {
                ALLOCATOR.begin_dump();
                collision_config.dump_frame_allocs = false;
            }

            defer!{ALLOCATOR.end_dump();}

            self.system.clear();

            self.system.extend(collision_config.bounds,
                (&entities, &positions, &radii).join()
                    .map(|(ent, &pos, &Radius(r))| {
                        let bounds = Bounds{
                            min: Point2::new(pos.1.x - r, pos.1.y - r),
                            max: Point2::new(pos.1.x + r, pos.1.y + r)};
                        (bounds, ent.id())}));

            self.system.par_sort();

            {
                if let Some((_dist, id, _point)) = self.system.pick_ray(
                    collision_config.bounds,
                    Point2::new(0f32, 360f32),
                    Vector2::new(1f32, 0f32),
                    std::f32::INFINITY, None,
                    |ray_origin, ray_direction, _dist, id| {
                        let ent = entities.entity(id);
                        let color = colors.get_mut(ent).unwrap();
                        *color = Color(0.5f32, 1f32, 0f32, 1f32);

                        let position = positions.get(ent).unwrap();
                        let Radius(r) = radii.get(ent).unwrap();
                        let center = Point2::new(position.1.x, position.1.y);
                        let ball_dir = center - ray_origin;
                        let ball_proj = ray_direction.dot(ball_dir);
                        let ball_extent = (ball_proj.powi(2) - ball_dir.magnitude2() + r.powi(2)).sqrt();

                        let range_min = ball_proj - ball_extent;
                        let range_max = ball_proj + ball_extent;

                        if range_max < 0f32 {
                            std::f32::INFINITY
                        } else if range_min < 0f32 {
                            0f32
                        } else {
                            range_min
                        }
                    })
                {
                    let ent = entities.entity(id);
                    let color = colors.get_mut(ent).unwrap();
                    *color = Color(1f32, 0f32, 0f32, 1f32);
                }
            }

            self.collisions.clear();
            self.collisions.extend(self.system.par_scan()
                .iter()
                .filter_map(|&(id0, id1)| {
                    let ent0 = entities.entity(id0);
                    let ent1 = entities.entity(id1);
                    let pos0 = positions.get(ent0).unwrap();
                    let pos1 = positions.get(ent1).unwrap();
                    let Radius(r0) = radii.get(ent0).unwrap();
                    let Radius(r1) = radii.get(ent1).unwrap();
                    let offset = pos1.1 - pos0.1;
                    let dist = offset.magnitude();
                    let dist_min = r0 + r1;

                    if dist > dist_min {
                        None
                    } else {
                        let d = dist_min - dist;
                        let n = if dist > 0.001 { offset / dist } else { Vector2::unit_x() };
                        let u = r1.powi(3) / (r0.powi(3) + r1.powi(3));
                        Some((ent0, ent1, u, n * d))
                    }}));
            print!("collisions: {:6}  ", self.collisions.len());

            let alloc_count = ALLOCATOR.clear_and_get_stats();
            print!("allocs: {:6}  ", alloc_count);
        } else {
            self.collisions.clear();
            for (ent0, pos0, &Radius(r0)) in (&entities, &positions, &radii).join() {
                for (ent1, pos1, &Radius(r1)) in (&entities, &positions, &radii).join() {
                    if ent1.id() >= ent0.id() {
                        continue;
                    }

                    let offset = pos1.1 - pos0.1;
                    let dist = offset.magnitude();
                    let dist_min = r0 + r1;

                    if dist < dist_min {
                        let d = dist_min - dist;
                        let n = offset / dist;
                        let u = r1.powi(3) / (r0.powi(3) + r1.powi(3));
                        self.collisions
                            .push((ent0, ent1, u, n * d));
                    }
                }
            }
        }
        print!("elapsed: {:6}us\r", start.elapsed().subsec_micros());

        for &(ent0, ent1, u, v) in &self.collisions {
            positions.get_mut(ent0).unwrap().1 -= v * u;
            positions.get_mut(ent1).unwrap().1 += v * (1f32 - u);
        }

        let x_min = 0f32;
        let y_min = 0f32;
        let x_max = screen_size.0 as f32 + x_min;
        let y_max = screen_size.1 as f32 + y_min;

        for (mut pos, &Radius(r)) in (&mut positions, &radii).join() {
            if pos.1.x - r < x_min {
                pos.1.x = x_min + r
            }
            if pos.1.y - r < y_min {
                pos.1.y = y_min + r
            }
            if pos.1.x + r > x_max {
                pos.1.x = x_max - r
            }
            if pos.1.y + r > y_max {
                pos.1.y = y_max - r
            }
        }
    }
}

#[derive(Copy, Clone)]
struct VertexData {
    offset: [f32; 2],
}
implement_vertex!(VertexData, offset);

#[derive(Copy, Clone)]
struct InstanceData {
    origin: [f32; 2],
    scale : [f32; 2],
    color : [f32; 4],
}
implement_vertex!(InstanceData, origin, scale, color);

struct InstanceBuffer {
    vbo: glium::VertexBuffer<InstanceData>,
    count: usize
}

impl InstanceBuffer {
    fn with_capacity(display: &glium::Display, count: usize) -> Option<Self> {
        let vbo = glium::VertexBuffer::<InstanceData>::empty_persistent(display, count).ok()?;
        Some(Self{ vbo, count })
    }

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

    fn write(&mut self, display: &glium::Display, data: &[InstanceData]) {
        self.resize(display, data.len());
        if !data.is_empty() {
            self.slice().write(data);
        }
    }
}

struct OffscreenTarget {
    texture: glium::texture::SrgbTexture2d,
    pbo: Option<glium::texture::pixel_buffer::PixelBuffer<(u8, u8, u8, u8)>>,
}

impl OffscreenTarget {
    fn with_size(display: &glium::Display, w: u32, h: u32) -> Option<Self> {
        let texture = match glium::texture::SrgbTexture2d::empty_with_format(display,
            glium::texture::SrgbFormat::U8U8U8U8,
            glium::texture::MipmapsOption::NoMipmap,
            w, h)
        {
            Ok(texture) => texture,
            Err(_) => return None,
        };
        Some(Self{ texture, pbo: None })
    }
}

const ASYNC_READBACK_FRAMES: usize = 3;

struct AsyncReadback {
    on_frame_data: Box<dyn FnMut(u32, u32, &[u8])>,
    frames: [OffscreenTarget; ASYNC_READBACK_FRAMES],
    i: usize,
}

impl AsyncReadback {
    fn png_writer(display: &glium::Display, w: u32, h: u32, base: std::path::PathBuf) -> Self {
        let on_frame_data = Box::new(move |w: u32, h: u32, data: &[u8]| {
            static INDEX: AtomicU32 = AtomicU32::new(0);
            let (path, f) = loop {
                let path = base.with_file_name(format!("{}_{:05}.{}",
                    base.file_stem().and_then(|s| s.to_str()).unwrap_or("default"),
                    INDEX.fetch_add(1, AtomicOrdering::Relaxed),
                    base.extension().and_then(|s| s.to_str()).unwrap_or("png")));
                match std::fs::File::create(path.clone()) {
                    Ok(f) => break (path, f),
                    Err(e) => match e.kind() {
                        std::io::ErrorKind::AlreadyExists => {},
                        _ => {
                            error!("Failed to create {}: {:?}", path.to_str().unwrap_or("EMPTY PATH"), e);
                            return;
                        }
                    },
                };
            };
            let mut encoder = png::Encoder::new(f, w, h);
            encoder.set_color(png::ColorType::RGBA);
            encoder.set_depth(png::BitDepth::Eight);
            let mut writer = match encoder.write_header() {
                Ok(writer) => writer,
                Err(e) => {
                    error!("Failed to write PNG header for {}: {:?}", path.to_str().unwrap_or("EMPTY PATH"), e);
                    return;
                },
            };
            let mut writer = writer.stream_writer();
            for y in 0..h {
                use std::io::Write;
                let y_ = h-y-1;
                let i0 = 4 * (w *  y_     ) as usize;
                let i1 = 4 * (w * (y_ + 1)) as usize;
                if let Err(e) = writer.write_all(&data[i0..i1]) {
                    error!("Failed to encode PNG data for {}: {:?}", path.to_str().unwrap_or("EMPTY PATH"), e);
                }
            }
        });

        let frames = [
            OffscreenTarget::with_size(display, w, h).unwrap(),
            OffscreenTarget::with_size(display, w, h).unwrap(),
            OffscreenTarget::with_size(display, w, h).unwrap(),
        ];

        Self{ on_frame_data, frames, i: 0 }
    }

    fn with_surface<Draw: FnMut(&mut glium::framebuffer::SimpleFrameBuffer)>(&mut self, display: &glium::Display, mut draw: Draw) {
        let frame = &mut self.frames[self.i];
        if let Some(pbo) = &mut frame.pbo {
            let w = frame.texture.width();
            let h = frame.texture.height();
            let bytes = 4 * w as usize * h as usize;
            assert_eq!(bytes, pbo.get_size());
            let data = unsafe { std::slice::from_raw_parts((*pbo.map_read()).as_ptr() as *const u8, bytes) };
            (self.on_frame_data)(w, h, data);
        }
        let mut fbo = glium::framebuffer::SimpleFrameBuffer::new(display, &frame.texture)
            .expect("failed to create framebuffer");
        draw(&mut fbo);
        frame.pbo = Some(frame.texture.read_to_pixel_buffer());
        self.i += 1;
        self.i = self.i % ASYNC_READBACK_FRAMES;
    }

    fn flush(&mut self) {
        for _ in 0..ASYNC_READBACK_FRAMES {
            let frame = &mut self.frames[self.i];
            if let Some(pbo) = &mut frame.pbo {
                let w = frame.texture.width();
                let h = frame.texture.height();
                let bytes = 4 * w as usize * h as usize;
                assert_eq!(bytes, pbo.get_size());
                let data = unsafe { std::slice::from_raw_parts((*pbo.map_read()).as_ptr() as *const u8, bytes) };
                (self.on_frame_data)(w, h, data);
            }
            frame.pbo = None;
            self.i += 1;
            self.i = self.i % ASYNC_READBACK_FRAMES;
        }
    }
}

struct Renderer {
    program_main: glium::Program,
    screen_size : [f32; 2],
    boxes       : InstanceBuffer,
    circles     : InstanceBuffer,
    vbo_box     : glium::VertexBuffer<VertexData>,
    vbo_circle  : glium::VertexBuffer<VertexData>,
}

impl Renderer {
    fn from_display(display: &glium::Display) -> Self {
        let screen_size = display.get_framebuffer_dimensions();
        let screen_size = [screen_size.0 as f32, screen_size.1 as f32];

        let program_main = glium::Program::from_source(display,
            r#"
                #version 450 core

                in vec2 offset;
                in vec2 origin;
                in vec2 scale;
                in vec4 color;

                out vec4 v_color;

                uniform vec2 screen_size;

                void main() {
                    vec2 position = origin + scale * offset;
                    v_color = color;
                    gl_Position = vec4(2.0 * position / screen_size - 1.0, 0.0, 1.0);
                }
            "#,
            r#"
                #version 450 core

                in vec4 v_color;

                out vec4 f_color;

                void main() {
                    f_color = vec4(v_color.rgb, 1.0);
                }
            "#,
            None)
            .expect("failed to compile shader");
        let boxes   = InstanceBuffer::with_capacity(display, 40_000).expect("failed to create boxes instance buffer");
        let circles = InstanceBuffer::with_capacity(display, 10_000).expect("failed to create circles instance buffer");
        let vbo_box = glium::VertexBuffer::immutable(display, &[
            VertexData{ offset: [-0.5, -0.5] },
            VertexData{ offset: [ 0.5, -0.5] },
            VertexData{ offset: [ 0.5,  0.5] },
            VertexData{ offset: [-0.5,  0.5] },
        ]).expect("failed to create box vbo");
        let vbo_circle = {
            const SIDES: u32 = 16;
            let data: Vec<_> = (0..SIDES)
                .map(|i| 2f32 * std::f32::consts::PI * (i as f32) / (SIDES as f32))
                .map(|u| [u.cos(), u.sin()])
                .map(|offset| VertexData{ offset })
                .collect();
            glium::VertexBuffer::immutable(display, data.as_slice())
                .expect("failed to create circle vbo")
        };
        Self{
            program_main,
            screen_size,
            boxes,
            circles,
            vbo_box,
            vbo_circle,
        }
    }

    fn update_screen_size(&mut self, size: [f32; 2]) {
        self.screen_size = size;
    }

    fn update_boxes(&mut self, display: &glium::Display, boxes: &[InstanceData]) {
        self.boxes.write(display, boxes);
    }

    fn update_circles(&mut self, display: &glium::Display, circles: &[InstanceData]) {
        self.circles.write(display, circles);
    }

    fn draw<Surf: glium::Surface>(&self, surface: &mut Surf) {
        let params = glium::DrawParameters{
            .. Default::default()
        };
        let uniforms = uniform!{
            screen_size: self.screen_size
        };
        surface.draw(
            (&self.vbo_circle, self.circles.slice().per_instance().unwrap()),
            glium::index::NoIndices(glium::index::PrimitiveType::LineLoop),
            &self.program_main, &uniforms, &params)
            .expect("failed to draw circles");
        surface.draw(
            (&self.vbo_box, self.boxes.slice().per_instance().unwrap()),
            glium::index::NoIndices(glium::index::PrimitiveType::LineLoop),
            &self.program_main, &uniforms, &params)
            .expect("failed to draw boxes");
    }
}

struct GameState {
    world: specs::World,
    lifecycle: Lifecycle,
    kinematics: Kinematics,
    collisions: Collisions,
}

impl GameState {
    const FRAME_RATE: u32 = 100;
    const FRAME_TIME_US: u32 = 1_000_000u32 / Self::FRAME_RATE;
    const MIN_FRAME_RATE: u32 = 20;
    const MAX_FRAME_TIME_US: u32 = 1_000_000u32 / Self::MIN_FRAME_RATE;

    fn new() -> Self {
        Self{
            world: specs::World::new(),
            lifecycle: Lifecycle {},
            kinematics: Kinematics {},
            collisions: Collisions::new(),
        }
    }
}

impl GameState {
    #[inline(never)]
    fn update(&mut self) {
        let step = Duration::from_micros(Self::FRAME_TIME_US as u64);
        let max_delta = Duration::from_micros(Self::MAX_FRAME_TIME_US as u64);
        self.world.get_mut::<Time>().unwrap().update(step, max_delta);

        while self.world.get_mut::<Time>().unwrap().step() {
            self.lifecycle.run_now(&self.world);
            self.world.maintain();
            self.kinematics.run_now(&self.world);
            self.collisions.run_now(&self.world);
        }
    }

    #[inline(never)]
    fn draw<Surf: glium::Surface>(&mut self, renderer: &mut Renderer, display: &glium::Display, surface: &mut Surf) {
        surface.clear_color(0f32, 0.0f32, 0.0f32, 1f32);

        let time = self.world.get_mut::<Time>().unwrap();

        let t = time.remainder();
        let u: f32 = ((t.as_secs() as f32) * 1_000_000f32 + (t.subsec_micros() as f32))
            / (Self::FRAME_TIME_US as f32);

        let circles: Vec<_> = {
            let positions = self.world.read_storage::<VerletPosition>();
            let radii     = self.world.read_storage::<Radius>();
            let colors    = self.world.read_storage::<Color>();

            (&positions, &radii, &colors).join()
                .map(|(&pos, &Radius(r), &color)|
                    InstanceData{
                        origin: (pos.1 + (pos.1 - pos.0) * u).into(),
                        scale : [r, r],
                        color : color.into(),
                    })
                .collect()
        };
        renderer.update_circles(display, circles.as_slice());

        let collision_config = self.world.read_resource::<CollisionSystemConfig>();
        let boxes: Vec<_> = if collision_config.enabled {
            self.collisions.system.iter()
                .map(|&(index, _)| {
                    use broadphase::SystemBounds;
                    let local: Bounds<_> = index.into();
                    let global = collision_config.bounds.to_global(local);
                    InstanceData{
                        origin: global.center().to_vec().into(),
                        scale : global.sizef().into(),
                        color : [0.3, 0.3, 0.3, 1.0],
                    }
                })
                .collect()
        } else { Vec::default() };
        renderer.update_boxes(display, boxes.as_slice());

        renderer.draw(surface);
    }
}

fn main() {
    env_logger::init();

    #[cfg(debug_assertions)]
    println!("Example should be run in RELEASE MODE for optimal performance!");

    let events_loop = glutin::event_loop::EventLoop::new();
    let window_builder = glutin::window::WindowBuilder::new()
        .with_title("Broadphase Util: Show Boxes")
        .with_resizable(true);
    let context = glutin::ContextBuilder::new()
        .with_gl(glutin::GlRequest::Specific(glutin::Api::OpenGl, (4, 5)))
        .with_gl_profile(glutin::GlProfile::Core)
        .with_multisampling(4);
    let display = glium::Display::new(window_builder, context, &events_loop)
        .expect("failed to create display");

    let screen_size = display.get_framebuffer_dimensions();

    let mut renderer = Renderer::from_display(&display);
    let mut screenshots = AsyncReadback::png_writer(&display, screen_size.0, screen_size.1, std::path::PathBuf::from("./screenshots/example"));
    let mut game_state = GameState::new();
    game_state.world.insert(Time::default());
    game_state.world.insert(ScreenSize::from(screen_size));
    game_state.world.insert(CollisionSystemConfig::from_screen_size(screen_size));
    game_state.world.insert(BallCount(0));
    game_state.world.register::<Lifetime>();
    game_state.world.register::<VerletPosition>();
    game_state.world.register::<Radius>();
    game_state.world.register::<Color>();

    events_loop.run(move |event, _window_target, control| {
        use crate::glutin::event::{DeviceEvent, ElementState, Event, VirtualKeyCode, WindowEvent};
        match event {
            Event::DeviceEvent{event, ..} =>
                match event {
                    DeviceEvent::Key(glutin::event::KeyboardInput{state: key_state, virtual_keycode, ..}) =>
                        match virtual_keycode {
                            Some(VirtualKeyCode::Space) =>
                                if key_state == ElementState::Pressed {
                                    let mut collision_config = game_state.world.write_resource::<CollisionSystemConfig>();
                                    collision_config.enabled = !collision_config.enabled;
                                }
                            Some(VirtualKeyCode::D) =>
                                if key_state == ElementState::Pressed {
                                    let mut collision_config = game_state.world.write_resource::<CollisionSystemConfig>();
                                    collision_config.dump_frame_allocs = true;
                                    println!("\nLOGGING ALLOCATIONS (\"TRACE\" LEVEL)\n");
                                }
                            Some(VirtualKeyCode::Snapshot) =>
                                if key_state == ElementState::Pressed {
                                    screenshots.with_surface(&display, |surface| {
                                        game_state.draw(&mut renderer, &display, surface);
                                    });
                                    screenshots.flush();
                                }
                            _ => {}
                        }
                    _ => {}
                }
            Event::WindowEvent{event, ..} => {
                match event {
                    WindowEvent::CloseRequested => {
                        screenshots.flush();
                        *control = glutin::event_loop::ControlFlow::Exit;
                    }
                    WindowEvent::Resized(size) => {
                        renderer.update_screen_size(size.into());
                        *game_state.world.get_mut::<ScreenSize>().unwrap() = size.into();
                        game_state.world.get_mut::<CollisionSystemConfig>().unwrap()
                            .update_bounds(size.width, size.height);
                        display.gl_window().window().request_redraw();
                        screenshots.flush();
                        screenshots = AsyncReadback::png_writer(&display, size.width, size.height, std::path::PathBuf::from("./screenshots/example"));
                    }
                    _ => {}
                }
            }
            Event::MainEventsCleared => {
                game_state.update();
                display.gl_window().window().request_redraw();
            }
            Event::RedrawRequested(..) => {
                let mut frame = display.draw();
                game_state.draw(&mut renderer, &display, &mut frame);
                frame.finish().unwrap();
            }
            _ => {}
        }
    });
}
