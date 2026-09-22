//! This module contains the Camera source.

use crate::image::Image;
use crate::types::{Frame, PixelFormat};
use anyhow::Result;
use bytes::Bytes;
use image::DynamicImage;
use nokhwa::Camera as NCamera;
use nokhwa::pixel_format::RgbFormat;
use nokhwa::utils::{CameraIndex, RequestedFormat, RequestedFormatType};
use std::thread;
use tokio::sync::mpsc;

/// Camera wrapper from nokhwa that generates `oxypipe` frames
/// Only stores the index the camera starts when with the stream.
pub struct Camera {
    index: CameraIndex,
}

impl Camera {
    /// Create a new Camera with a given index.
    pub fn new(index: CameraIndex) -> Self {
        Self { index }
    }
    /// Create a new Camera with a integger index.
    pub fn with_index(index: u32) -> Self {
        Self {
            index: CameraIndex::Index(index),
        }
    }
    /// Create new Camera with an string index. Usefull for things like IP Cameras.
    pub fn with_string(index: &str) -> Self {
        Self {
            index: CameraIndex::String(index.to_string()),
        }
    }
    /// Starts the camera and gets a single `Frame` before shutting the connection.
    pub fn get_frame(&self) -> Result<Frame> {
        let format = RequestedFormat::new::<RgbFormat>(RequestedFormatType::None);
        // Index is very cheap to clone btw.
        let mut camera = NCamera::new(self.index.clone(), format)?;
        camera.open_stream()?;
        let nframe = camera.frame()?;
        let image = nframe.decode_image::<RgbFormat>()?;
        Ok(Frame {
            id: 0,
            width: image.width(),
            height: image.height(),
            format: PixelFormat::Rgb8,
            data: Bytes::from(image.into_raw()),
            timestamp_ns: 0,
        })
    }
    /// Starts the camera and gets a single `Image` before shutting the connection.
    pub fn get_image(&self) -> Result<Image> {
        let format = RequestedFormat::new::<RgbFormat>(RequestedFormatType::None);
        // Index is very cheap to clone btw.
        let mut camera = NCamera::new(self.index.clone(), format)?;
        camera.open_stream()?;
        let nframe = camera.frame()?;
        let image = nframe.decode_image::<RgbFormat>()?;
        let width = image.width();
        let height = image.height();
        // Expect should be safe here if its not big enough we are doing something wrong.
        let rgb_img = image::RgbImage::from_raw(width, height, image.into_raw())
            .expect("The image container is not big enough");
        Ok(Image {
            inner: DynamicImage::ImageRgb8(rgb_img),
            format: crate::types::PixelFormat::Rgb8,
        })
    }
}

impl super::StreamSource for Camera {
    fn stream(self) -> mpsc::Receiver<Frame> {
        let (tx, rx) = mpsc::channel(2);
        let index = self.index;

        thread::spawn(move || {
            let format = RequestedFormat::new::<RgbFormat>(RequestedFormatType::None);
            let mut camera = match NCamera::new(index, format) {
                Ok(cam) => cam,
                Err(e) => {
                    tracing::error!("Failed to initialize camera: {:?}", e);
                    return;
                }
            };
            if let Err(e) = camera.open_stream() {
                tracing::error!("Failed to open camera stream: {:?}", e);
                return;
            }

            let mut frame_id = 0;

            loop {
                let Ok(buffer) = camera.frame() else { break };
                let Ok(rgb_image) = buffer.decode_image::<RgbFormat>() else {
                    continue;
                };
                let frame = Frame {
                    id: frame_id,
                    width: rgb_image.width(),
                    height: rgb_image.height(),
                    format: PixelFormat::Rgb8,
                    data: Bytes::from(rgb_image.into_raw()),
                    timestamp_ns: 0,
                };
                frame_id += 1;
                if tx.try_send(frame).is_err() && tx.is_closed() {
                    break;
                }
            }
        });

        rx
    }
}

impl super::OneShotSource for Camera {
    fn load(self) -> Result<Frame> {
        self.get_frame()
    }
}
