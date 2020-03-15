// mlodato, 20190318

extern crate zvxryb_broadphase as broadphase;
extern crate cgmath;
extern crate env_logger;
extern crate ggez;
extern crate rand;
extern crate specs;

use cgmath::prelude::*;
use rand::prelude::*;
use specs::prelude::*;

use broadphase::Bounds;
use cgmath::{Point2, Point3, Vector2, Vector3};
use std::alloc::{GlobalAlloc, Layout as AllocLayout, System as SystemAlloc};
use std::sync::atomic::{AtomicUsize, Ordering as AtomicOrdering};
use std::time::{Duration, Instant};

struct AllocLogger {
    count: AtomicUsize
}

impl AllocLogger {
    fn clear_and_get_stats(&self) -> usize {
        self.count.swap(0, AtomicOrdering::Relaxed)
    }
}

unsafe impl GlobalAlloc for AllocLogger {
    #[inline(never)]
    unsafe fn alloc(&self, layout: AllocLayout) -> *mut u8 {
        self.count.fetch_add(1, AtomicOrdering::Relaxed);
        let result = SystemAlloc.alloc(layout);
        result
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: AllocLayout) {
        SystemAlloc.dealloc(ptr, layout);
    }
}

#[global_allocator]
static ALLOCATOR: AllocLogger = AllocLogger{
    count: AtomicUsize::new(0)};

struct Time {
    current: Duration,
    delta: Duration,
}

impl Default for Time {
    fn default() -> Self {
        Self {
            current: Duration::default(),
            delta: Duration::default(),
        }
    }
}

struct CollisionSystemConfig {
    enabled: bool,
    bounds: Bounds<Point3<f32>>
}

impl Default for CollisionSystemConfig {
    fn default() -> Self {
        Self{
            enabled: true,
            bounds: Bounds::new(
                Point3::new(0f32, 0f32, 0f32),
                Point3::new(1f32, 1f32, 1f32))}
    }
}

impl CollisionSystemConfig {
    fn from_screen_coords(rect: ggez::graphics::Rect) -> Self {
        let scale = if rect.w > rect.h { rect.w } else { rect.h };
        let min = Point3::new(rect.x, rect.y, 0f32);
        let max = min.add_element_wise(scale);
        Self{
            enabled: true,
            bounds: Bounds::new(min, max)}
    }
}

struct ScreenCoords(ggez::graphics::Rect);

impl Default for ScreenCoords {
    fn default() -> Self {
        Self(ggez::graphics::Rect::default())
    }
}

struct BallCount(u32);

impl Default for BallCount {
    fn default() -> Self {
        Self(0)
    }
}

struct Color(ggez::graphics::Color);

impl Default for Color {
    fn default() -> Self {
        Self(ggez::graphics::WHITE)
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
    type SystemData = (
        specs::Entities<'a>,
        specs::Read<'a, specs::LazyUpdate>,
        specs::Read<'a, Time>,
        specs::Read<'a, ScreenCoords>,
        specs::Write<'a, BallCount>,
        specs::ReadStorage<'a, Lifetime>,
    );

    #[inline(never)]
    fn run(&mut self, data: Self::SystemData) {
        let (entities, lazy, time, screen_coords, mut ball_count, lifetimes) = data;

        for (entity, &Lifetime(expires)) in (&entities, &lifetimes).join() {
            if expires < time.current {
                entities.delete(entity).unwrap();
                ball_count.0 -= 1;
            }
        }

        const BALL_COUNT_MAX: u32 = 2500;
        const LIFETIME_MIN_MS: u32 = 10000;
        const LIFETIME_MAX_MS: u32 = 50000;
        for _ in 0..BALL_COUNT_MAX*time.delta.subsec_millis()/LIFETIME_MIN_MS {
            if ball_count.0 >= BALL_COUNT_MAX {
                break;
            }
            let lifetime = Duration::from_millis(rand::thread_rng().gen_range(
                LIFETIME_MIN_MS as u64, LIFETIME_MAX_MS as u64));

            let r = rand::thread_rng().gen_range(0.5f32, 2.0f32).exp();

            let x0 = screen_coords.0.x + r;
            let x1 = screen_coords.0.w - 2f32 * r + x0;
            let x = rand::thread_rng().gen_range(x0, x1);

            let y = screen_coords.0.y + r;

            create_ball(
                lazy.create_entity(&entities),
                (time.current + lifetime).into(),
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
        let dt = (time.delta.as_secs() as f32) + (time.delta.subsec_micros() as f32) / 1_000_000f32;
        let gravity = 50f32 * dt * dt;
        for mut pos in (&mut positions).join() {
            let mut pos_2 = pos.1 + (pos.1 - pos.0);
            pos_2.y += gravity;
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
    system: broadphase::Layer<broadphase::Index64_3D, specs::world::Index>,
    collisions: Vec<(specs::Entity, specs::Entity, f32, Vector2<f32>)>,
}

impl Collisions {
    fn new() -> Self {
        Self {
            system: broadphase::LayerBuilder::new()
                .with_min_depth(6)
                .build(),
            collisions: Vec::new(),
        }
    }
}

impl<'a> specs::System<'a> for Collisions {
    type SystemData = (
        specs::Entities<'a>,
        specs::Read<'a, ScreenCoords>,
        specs::Read<'a, CollisionSystemConfig>,
        specs::WriteStorage<'a, VerletPosition>,
        specs::ReadStorage<'a, Radius>,
        specs::WriteStorage<'a, Color>
    );

    #[inline(never)]
    fn run(&mut self, data: Self::SystemData) {
        let (entities, screen_coords, collision_config, mut positions, radii, mut colors) = data;

        for Color(color) in (&mut colors).join() {
            *color = ggez::graphics::Color::new(1f32, 0.5f32, 0f32, 1f32);
        }

        let start = Instant::now();

        if collision_config.enabled {
            ALLOCATOR.clear_and_get_stats();

            self.system.clear();

            self.system.extend(collision_config.bounds,
                (&entities, &positions, &radii).join()
                    .map(|(ent, &pos, &Radius(r))| {
                        let bounds = Bounds{
                            min: Point3::new(pos.1.x - r, pos.1.y - r, 0.0f32),
                            max: Point3::new(pos.1.x + r, pos.1.y + r, 0.0f32)};
                        (bounds, ent.id())}));

            self.system.par_sort();

            {
                if let Some((_dist, id, _point)) = self.system.pick_ray(
                    collision_config.bounds,
                    Point3::new(0f32, 360f32, 0f32),
                    Vector3::new(1f32, 0f32, 0f32),
                    std::f32::INFINITY, None,
                    |ray_origin, ray_direction, _dist, id| {
                        let ent = entities.entity(id);
                        let Color(color) = colors.get_mut(ent).unwrap();
                        *color = ggez::graphics::Color::new(0.5f32, 1f32, 0f32, 1f32);
                        
                        let position = positions.get(ent).unwrap();
                        let Radius(r) = radii.get(ent).unwrap();
                        let center = Point3::new(position.1.x, position.1.y, 0f32);
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
                    let Color(color) = colors.get_mut(ent).unwrap();
                    *color = ggez::graphics::Color::new(1f32, 0f32, 0f32, 1f32);
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
                        let n = offset / dist;
                        let u = r1.powi(3) / (r0.powi(3) + r1.powi(3));
                        Some((ent0, ent1, u, n * d))
                    }}));

            let alloc_count = ALLOCATOR.clear_and_get_stats();
            print!("allocs: {:3}     ", alloc_count);
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

        let x_min = screen_coords.0.x;
        let y_min = screen_coords.0.y;
        let x_max = screen_coords.0.w + x_min;
        let y_max = screen_coords.0.h + y_min;

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

struct GameState {
    world: specs::World,
    lifecycle: Lifecycle,
    kinematics: Kinematics,
    collisions: Collisions,
}

impl GameState {
    const FRAME_RATE: u32 = 100;
    const FRAME_TIME_US: u32 = 1_000_000u32 / Self::FRAME_RATE;

    fn new() -> Self {
        Self{
            world: specs::World::new(),
            lifecycle: Lifecycle {},
            kinematics: Kinematics {},
            collisions: Collisions::new(),
        }
    }
}

impl ggez::event::EventHandler for GameState {
    #[inline(never)]
    fn update(&mut self, context: &mut ggez::Context) -> ggez::GameResult<()> {
        self.world.write_resource::<Time>().current = ggez::timer::time_since_start(&context);

        self.world.write_resource::<Time>().delta =
            Duration::from_micros(Self::FRAME_TIME_US as u64);
        let mut i = 0;
        while ggez::timer::check_update_time(context, Self::FRAME_RATE) {
            if i < 10 {
                self.world.write_resource::<Time>().current = ggez::timer::time_since_start(&context);
                self.lifecycle.run_now(&self.world);
                self.world.maintain();
                self.kinematics.run_now(&self.world);
                self.collisions.run_now(&self.world);
            }
            i += 1;
        }
        Ok(())
    }

    fn key_down_event(&mut self,
        _context: &mut ggez::Context,
        key: ggez::input::keyboard::KeyCode,
        _mods: ggez::input::keyboard::KeyMods,
        _repeat: bool)
    {
        match key {
            ggez::input::keyboard::KeyCode::Space => {
                let mut collision_config = self.world.write_resource::<CollisionSystemConfig>();
                collision_config.enabled = !collision_config.enabled;
            },
            _ => {}
        }
    }

    #[inline(never)]
    fn draw(&mut self, context: &mut ggez::Context) -> ggez::GameResult<()> {
        use ggez::graphics::{BLACK, clear, draw, DrawMode, MeshBuilder, present, Rect};

        clear(context, BLACK);

        let t = ggez::timer::remaining_update_time(context);
        let u: f32 = ((t.as_secs() as f32) * 1_000_000f32 + (t.subsec_micros() as f32))
            / (Self::FRAME_TIME_US as f32);

        {
            let positions = self.world.read_storage::<VerletPosition>();
            let radii     = self.world.read_storage::<Radius>();
            let colors    = self.world.read_storage::<Color>();

            let iter = &mut (&positions, &radii, &colors).join().peekable();
            while let Some(_) = iter.peek() {
                let mut mesh_builder = MeshBuilder::new();
                for (&pos, &Radius(r), &Color(color)) in iter.take(500) {
                    let pos_ = pos.1 + (pos.1 - pos.0) * u;
                    mesh_builder.circle::<[f32; 2]>(DrawMode::stroke(1.5f32), pos_.into(), r, 0.7f32, color);
                }
                let mesh = mesh_builder.build(context)?;
                draw(context, &mesh, ([0f32, 0f32],))?;
            }
        }

        let collision_config = self.world.read_resource::<CollisionSystemConfig>();
        if collision_config.enabled {
            let iter = &mut self.collisions.system.iter().peekable();
            while let Some(_) = iter.peek() {
                let mut mesh_builder = MeshBuilder::new();
                for &(index, _) in iter.take(1000) {
                    use broadphase::SystemBounds;
                    let local: Bounds<_> = index.into();
                    let global = collision_config.bounds.to_global(local);
                    let global_size = global.sizef();
                    let rect = Rect::new(
                        global.min.x,
                        global.min.y,
                        global_size.x,
                        global_size.y);
                    mesh_builder.rectangle(DrawMode::stroke(1.5f32), rect,
                        ggez::graphics::Color::new(0.3f32, 0.3f32, 0.3f32, 1f32));
                }
                let mesh = mesh_builder.build(context)?;
                draw(context, &mesh, ([0f32, 0f32],))?;
            }
        }

        present(context)?;

        Ok(())
    }
}

fn main() {
    env_logger::init();

    #[cfg(debug_assertions)]
    println!("Example should be run in RELEASE MODE for optimal performance!");

    use ggez::conf::*;
    let (mut context, mut event_loop) = ggez::ContextBuilder::new("broadphase_demo", "zvxryb")
        .window_setup(
            WindowSetup::default()
                .title("broadphase demo")
                .samples(NumSamples::Eight))
        .window_mode(
            WindowMode::default()
                .dimensions(1280f32, 720f32))
        .build()
        .expect("failed to build ggez context");

    let mut state = GameState::new();
    let screen_rect = ggez::graphics::screen_coordinates(&context);
    state.world.insert(Time::default());
    state.world.insert(ScreenCoords(screen_rect));
    state.world.insert(CollisionSystemConfig::from_screen_coords(screen_rect));
    state.world.insert(BallCount(0));
    state.world.register::<Lifetime>();
    state.world.register::<VerletPosition>();
    state.world.register::<Radius>();
    state.world.register::<Color>();

    ggez::event::run(&mut context, &mut event_loop, &mut state).expect("failed to run game");
}
