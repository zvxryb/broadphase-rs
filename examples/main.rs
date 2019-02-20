// mlodato, 20190219

extern crate broadphase;
extern crate cgmath;
extern crate ggez;
extern crate rand;
extern crate specs;

use rand::prelude::*;
use specs::prelude::*;
use std::time::Instant;

struct Time {
    current: std::time::Duration,
    delta: std::time::Duration,
}

impl Default for Time {
    fn default() -> Self {
        Self {
            current: std::time::Duration::default(),
            delta: std::time::Duration::default(),
        }
    }
}

struct CollisionSystemConfig {
    enabled: bool,
    bounds: broadphase::Bounds<cgmath::Point3<f32>>
}

impl Default for CollisionSystemConfig {
    fn default() -> Self {
        Self{
            enabled: true,
            bounds: broadphase::Bounds::new(
                cgmath::Point3::new(0f32, 0f32, 0f32),
                cgmath::Point3::new(1f32, 1f32, 1f32))}
    }
}

impl CollisionSystemConfig {
    fn from_screen_coords(rect: ggez::graphics::Rect) -> Self {
        use cgmath::ElementWise;
        let scale = if rect.w > rect.h { rect.w } else { rect.h };
        let min = cgmath::Point3::new(rect.x, rect.y, 0f32);
        let max = min.add_element_wise(scale);
        Self{
            enabled: true,
            bounds: broadphase::Bounds::new(min, max)}
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

struct Lifetime(std::time::Duration);

impl specs::Component for Lifetime {
    type Storage = specs::VecStorage<Self>;
}

impl From<std::time::Duration> for Lifetime {
    fn from(expires: std::time::Duration) -> Self {
        Self(expires)
    }
}

#[derive(Copy, Clone)]
struct VerletPosition {
    x0: f32,
    y0: f32,
    x1: f32,
    y1: f32,
}

impl specs::Component for VerletPosition {
    type Storage = specs::VecStorage<Self>;
}

impl From<(f32, f32)> for VerletPosition {
    fn from(pos: (f32, f32)) -> Self {
        Self {
            x0: pos.0,
            y0: pos.1,
            x1: pos.0,
            y1: pos.1,
        }
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

        while ball_count.0 < 1500 {
            let lifetime = std::time::Duration::from_millis(rand::thread_rng().gen_range(5000, 30000));

            let r = rand::thread_rng().gen_range(1f32, 3f32).exp();

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
        let gravity = 300f32 * dt * dt;
        for mut pos in (&mut positions).join() {
            let x0 = pos.x0;
            let y0 = pos.y0;
            let x1 = pos.x1;
            let y1 = pos.y1;
            let x2 = 2f32 * x1 - x0;
            let y2 = 2f32 * y1 - y0 + gravity;
            pos.x1 = x2;
            pos.y1 = y2;
            pos.x0 = x1;
            pos.y0 = y1;
        }
        for mut pos in (&mut positions).join() {
            let dx = pos.x1 - pos.x0;
            let dy = pos.y1 - pos.y0;
            let d = (dx * dx + dy * dy).sqrt();
            const SPEED_LIMIT: f32 = 1f32;
            if d > SPEED_LIMIT {
                pos.x1 = pos.x0 + SPEED_LIMIT * dx / d;
                pos.y1 = pos.y0 + SPEED_LIMIT * dy / d;
            }
        }
    }
}

struct Collisions {
    system: broadphase::Layer<broadphase::Index64_3D, specs::Entity, cgmath::Point3<u32>>,
    collisions: Vec<(specs::Entity, specs::Entity, (f32, f32))>,
}

impl Collisions {
    fn new() -> Self {
        Self {
            system: broadphase::Layer::new(),
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
    );

    #[inline(never)]
    fn run(&mut self, data: Self::SystemData) {
        let (entities, screen_coords, collision_config, mut positions, radii) = data;

        let start = Instant::now();

        if collision_config.enabled {
            self.system.clear();
            self.system.extend(collision_config.bounds,
                (&entities, &positions, &radii).join()
                    .map(|(ent, &pos, &Radius(r))| {
                        let bounds = broadphase::Bounds{
                            min: cgmath::Point3::new(pos.x1 - r, pos.y1 - r, 0.0f32),
                            max: cgmath::Point3::new(pos.x1 + r, pos.y1 + r, 0.0f32)};
                        (bounds, ent)}));
            self.system.sort();

            self.collisions = self.system.detect_collisions()
                .iter()
                .filter_map(|&(ent0, ent1)| {
                    let pos0 = positions.get(ent0).unwrap();
                    let pos1 = positions.get(ent1).unwrap();
                    let Radius(r0) = radii.get(ent0).unwrap();
                    let Radius(r1) = radii.get(ent1).unwrap();
                    let dx = pos1.x1 - pos0.x1;
                    let dy = pos1.y1 - pos0.y1;
                    let dist = (dx * dx + dy * dy).sqrt();
                    let dist_min = r0 + r1;

                    if dist > dist_min {
                        None
                    } else {
                        let d = dist_min - dist;
                        let nx = dx / dist;
                        let ny = dy / dist;
                        Some((ent0, ent1, (nx * d / 2f32, ny * d / 2f32)))
                    }})
                .collect();
        } else {
            self.collisions.clear();
            for (ent0, pos0, &Radius(r0)) in (&entities, &positions, &radii).join() {
                for (ent1, pos1, &Radius(r1)) in (&entities, &positions, &radii).join() {
                    if ent1.id() >= ent0.id() {
                        continue;
                    }

                    let dx = pos1.x1 - pos0.x1;
                    let dy = pos1.y1 - pos0.y1;
                    let dist = (dx * dx + dy * dy).sqrt();
                    let dist_min = r0 + r1;

                    if dist < dist_min {
                        let d = dist_min - dist;
                        let nx = dx / dist;
                        let ny = dy / dist;
                        self.collisions
                            .push((ent0, ent1, (nx * d / 2f32, ny * d / 2f32)));
                    }
                }
            }
        }
        print!("elapsed: {}    \r", start.elapsed().subsec_micros());

        for &(ent0, ent1, (dx, dy)) in &self.collisions {
            {
                let pos = positions.get_mut(ent0).unwrap();
                pos.x1 -= dx;
                pos.y1 -= dy;
            }
            {
                let pos = positions.get_mut(ent1).unwrap();
                pos.x1 += dx;
                pos.y1 += dy;
            }
        }

        let x_min = screen_coords.0.x;
        let y_min = screen_coords.0.y;
        let x_max = screen_coords.0.w + x_min - 1f32;
        let y_max = screen_coords.0.h + y_min - 1f32;

        for (mut pos, &Radius(r)) in (&mut positions, &radii).join() {
            if pos.x1 - r < x_min {
                pos.x1 = x_min + r
            }
            if pos.y1 - r < y_min {
                pos.y1 = y_min + r
            }
            if pos.x1 + r > x_max {
                pos.x1 = x_max - r
            }
            if pos.y1 + r > y_max {
                pos.y1 = y_max - r
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

        self.lifecycle.run_now(&self.world.res);
        self.world.maintain();

        self.world.write_resource::<Time>().delta =
            std::time::Duration::from_micros(Self::FRAME_TIME_US as u64);
        while ggez::timer::check_update_time(context, Self::FRAME_RATE) {
            self.world.write_resource::<Time>().current = ggez::timer::time_since_start(&context);
            self.kinematics.run_now(&self.world.res);
            self.collisions.run_now(&self.world.res);
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
        use ggez::graphics::*;

        clear(context, BLACK);

        let t = ggez::timer::remaining_update_time(context);
        let u: f32 = ((t.as_secs() as f32) * 1_000_000f32 + (t.subsec_micros() as f32))
            / (Self::FRAME_TIME_US as f32);

        {
            let positions = self.world.read_storage::<VerletPosition>();
            let radii     = self.world.read_storage::<Radius>();

            let iter = &mut (&positions, &radii).join().peekable();
            while let Some(_) = iter.peek() {
                let mut mesh_builder = MeshBuilder::new();
                for (&pos, &Radius(r)) in iter.take(500) {
                    let x = (pos.x1 - pos.x0) * u + pos.x1;
                    let y = (pos.y1 - pos.y0) * u + pos.y1;

                    mesh_builder.circle(DrawMode::stroke(1f32), [x, y], r, 1f32, Color::new(1f32, 0.5f32, 0f32, 1f32));
                }
                let mesh = mesh_builder.build(context)?;
                draw(context, &mesh, ([0f32, 0f32],))?;
            }
        }

        let collision_config = self.world.read_resource::<CollisionSystemConfig>();
        if collision_config.enabled {
            let scale  = collision_config.bounds.size();
            let offset = collision_config.bounds.min;

            let iter = &mut self.collisions.system.tree.iter().peekable();
            while let Some(_) = iter.peek() {
                let mut mesh_builder = MeshBuilder::new();
                for &(index, _) in iter.take(1000) {
                    use broadphase::SpatialIndex;
                    let origin = index.origin().map(|x| (x as f32) / ((u32::max_value() - 1) as f32));
                    let size = ((32 - index.depth()) as f32).exp2() / ((u32::max_value() - 1) as f32);
                    let rect = Rect::new(
                        scale.x * origin.x + offset.x,
                        scale.y * origin.y + offset.y,
                        scale.x * size,
                        scale.y * size);
                    mesh_builder.rectangle(DrawMode::stroke(1f32), rect, Color::new(0.3f32, 0.3f32, 0.3f32, 1f32));
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
    let (mut context, mut event_loop) = ggez::ContextBuilder::new("broadphase demo", "zvxryb")
        .window_mode(
            ggez::conf::WindowMode::default()
                .dimensions(1280f32, 720f32))
        .build()
        .expect("failed to build ggez context");

    let mut state = GameState::new();
    let screen_rect = ggez::graphics::screen_coordinates(&context);
    state.world.add_resource(Time::default());
    state.world.add_resource(ScreenCoords(screen_rect));
    state.world.add_resource(CollisionSystemConfig::from_screen_coords(screen_rect));
    state.world.add_resource(BallCount(0));
    state.world.register::<Lifetime>();
    state.world.register::<VerletPosition>();
    state.world.register::<Radius>();

    ggez::event::run(&mut context, &mut event_loop, &mut state).expect("failed to run game");
}
