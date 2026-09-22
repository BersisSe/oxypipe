//! This module contains any kind of Sources that power the pipeline.
//! It provides a base traits by every source used in oxypipe

use anyhow::Result;
#[cfg(feature = "camera")]
mod camera;
#[cfg(feature = "camera")]
pub use camera::Camera;

mod file;
pub use file::File;

use crate::types::Frame;
use tokio::sync::mpsc::Receiver;

/// StreamSource trait provides a Source to Stream its data in a loop. Useful for cameras or videos.
pub trait StreamSource {
    /// Consumes the source, spawns its capture loop, and returns a channel of frames.
    fn stream(self) -> Receiver<Frame>;
}
/// OneShotSource trait provides a Source to load a single time a. Useful for single capture with a camera or a image file
pub trait OneShotSource {
    fn load(self) -> Result<Frame>;
}
