//! This Module contains the File Source.

use crate::{image::Image, types::Frame};
use anyhow::Result;
use std::path::PathBuf;
use std::time::SystemTime;

/// File is a OneShot Source meaning it reads once and gives a frame by tuning it to image then frame.
/// If the file asked is not a Image loading fails.
pub struct File {
    path: PathBuf,
}
impl File {
    pub fn new(path: impl Into<PathBuf>) -> Self {
        Self { path: path.into() }
    }
}

impl super::OneShotSource for File {
    fn load(self) -> Result<Frame> {
        let timestamp = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap()
            .as_nanos() as u64;
        Ok(Image::from_file(self.path.as_path())?.to_frame(0, timestamp))
    }
}
