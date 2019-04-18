extern crate broadphase;

extern crate bincode;
extern crate cgmath;

#[macro_use]
extern crate serde;

use broadphase::Bounds;
use cgmath::Point3;

use std::fs::File;

pub type ID = u32;

const FORMAT_SIGNATURE: [u8;8] = *b"BR_SCENE";
const FORMAT_VERSION: (u16, u16) = (1, 0);

#[derive(Deserialize, Serialize)]
struct Header {
    signature: [u8;8],
    version: (u16, u16)
}

#[derive(Deserialize, Serialize)]
pub struct Scene {
    pub system_bounds: Bounds<Point3<f32>>,
    pub object_bounds: Vec<(Bounds<Point3<f32>>, ID)>
}

#[derive(Debug)]
pub enum SceneIOError {
    IOError(std::io::Error),
    BincodeError(bincode::Error),
    InvalidSignature([u8;8]),
    InvalidVersion((u16, u16))
}

impl Scene {
    pub fn load(path: &str) -> Result<Scene, SceneIOError> {
        let f = File::open(path)
            .map_err(|err| SceneIOError::IOError(err))?;

        let header: Header = bincode::deserialize_from(&f)
            .map_err(|err| SceneIOError::BincodeError(err))?;

        if header.signature != FORMAT_SIGNATURE {
            return Err(SceneIOError::InvalidSignature(header.signature));
        }

        if header.version.0 != FORMAT_VERSION.0 {
            return Err(SceneIOError::InvalidVersion(header.version));
        }

        bincode::deserialize_from::<_, Scene>(f)
            .map_err(|err| SceneIOError::BincodeError(err))
    }

    pub fn save(&self, path: &str) -> Result<(), SceneIOError> {
        let f = File::create(path)
            .map_err(|err| SceneIOError::IOError(err))?;

        bincode::serialize_into(&f, &Header{
            signature: FORMAT_SIGNATURE,
            version: FORMAT_VERSION
        }).map_err(|err| SceneIOError::BincodeError(err))?;

        bincode::serialize_into(f, self)
            .map_err(|err| SceneIOError::BincodeError(err))
    }
}