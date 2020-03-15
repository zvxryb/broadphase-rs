extern crate zvxryb_broadphase as broadphase;

extern crate bincode;
extern crate cgmath;

#[macro_use]
extern crate serde;

use broadphase::{Bounds, Layer, Index64_3D};
use cgmath::Point3;

use std::fs::File;
use std::io::{Read, Write};
use std::path::Path;

pub type ID = u32;
pub type Index = Index64_3D;

const FORMAT_SIGNATURE: [u8;8] = *b"BR_SCENE";
const FORMAT_VERSION: (u16, u16) = (1, 2);

#[derive(Deserialize, Serialize)]
struct Header {
    signature: [u8;8],
    version: (u16, u16)
}

#[derive(Clone, Deserialize, Serialize)]
pub struct SceneV1_0 {
    pub system_bounds: Bounds<Point3<f32>>,
    pub object_bounds: Vec<(Bounds<Point3<f32>>, ID)>
}

#[derive(Clone, Deserialize, Serialize)]
pub struct SceneV1_1 {
    pub system_bounds: Bounds<Point3<f32>>,
    pub object_bounds: Vec<(Bounds<Point3<f32>>, ID)>,
    pub layer: Layer<Index, ID>
}

#[derive(Clone, Deserialize, Serialize)]
pub struct SceneV1_2 {
    pub system_bounds: Bounds<Point3<f32>>,
    pub object_bounds: Vec<(Bounds<Point3<f32>>, ID)>,
    pub layer: Layer<Index, ID>,
    pub collisions: Vec<(ID, ID)>,
    pub hits: Vec<ID>,
    pub nearest: Option<(ID, f32)>
}

pub type Scene = SceneV1_2;

#[derive(Debug)]
pub enum SceneIOError {
    IOError(std::io::Error),
    BincodeError(bincode::Error),
    InvalidSignature([u8;8]),
    UnsupportedVersion((u16, u16))
}

impl Scene {
    pub fn load<P: AsRef<Path>>(path: P) -> Result<Scene, SceneIOError> {
        let f = File::open(path)
            .map_err(SceneIOError::IOError)?;

        Self::parse(&f)
    }

    pub fn parse<IO: Read>(mut io: IO) -> Result<Scene, SceneIOError> {
        let header: Header = bincode::deserialize_from(io.by_ref())
            .map_err(SceneIOError::BincodeError)?;

        if header.signature != FORMAT_SIGNATURE {
            return Err(SceneIOError::InvalidSignature(header.signature));
        }

        if header.version.0 != FORMAT_VERSION.0 || header.version.1 > FORMAT_VERSION.1 {
            return Err(SceneIOError::UnsupportedVersion(header.version));
        }

        #[allow(clippy::identity_conversion)]
        match header.version.1 {
            0 => bincode::deserialize_from::<_, SceneV1_0>(io).map(|scene| scene.into()),
            1 => bincode::deserialize_from::<_, SceneV1_1>(io).map(|scene| scene.into()),
            2 => bincode::deserialize_from::<_, SceneV1_2>(io).map(|scene| scene.into()),
            _ => panic!()
        }.map_err(SceneIOError::BincodeError)
    }

    pub fn save<P: AsRef<Path>>(&self, path: P) -> Result<(), SceneIOError> {
        let f = File::create(path)
            .map_err(SceneIOError::IOError)?;

        self.assemble(&f)
    }

    pub fn assemble<IO: Write>(&self, mut io: IO) -> Result<(), SceneIOError> {
        bincode::serialize_into(io.by_ref(), &Header{
            signature: FORMAT_SIGNATURE,
            version: FORMAT_VERSION
        }).map_err(SceneIOError::BincodeError)?;

        bincode::serialize_into(io, self)
            .map_err(SceneIOError::BincodeError)
    }
}

impl From<SceneV1_0> for Scene {
    fn from(scene: SceneV1_0) -> Self {
        Scene{
            system_bounds: scene.system_bounds,
            object_bounds: scene.object_bounds,
            layer: Default::default(),
            collisions: Default::default(),
            hits: Default::default(),
            nearest: Default::default()
        }
    }
}

impl From<SceneV1_1> for Scene {
    fn from(scene: SceneV1_1) -> Self {
        Scene{
            system_bounds: scene.system_bounds,
            object_bounds: scene.object_bounds,
            layer: scene.layer,
            collisions: Default::default(),
            hits: Default::default(),
            nearest: Default::default()
        }
    }
}