//! This module contains Image loading and manupilation utilities that fit right in with the oxypipe ecosystem.

use anyhow::{Result, anyhow};
use bytes::Bytes;
use image::imageops::FilterType;
use image::{DynamicImage, ImageBuffer, Rgb};
use ndarray::Array4;
use std::path::Path;

use crate::types::{BoundingBox, Frame, PixelFormat};
use crate::vision::Normalization;

/// A Basic Image Struct that it used for loading in-out of frames. Internally uses a `image::DynamicImage` which is public.
pub struct Image {
    pub inner: DynamicImage,
    pub format: PixelFormat,
}

impl Image {
    /// Turn a frame into a image.
    /// Accepts every supported `PixelFormat`; BGR(A) input is normalized to RGB(A) channel order.
    pub fn from_frame(frame: &Frame) -> Result<Self> {
        let (width, height) = (frame.width, frame.height);
        let raw = frame.data.clone().into();

        let inner = match frame.format {
            PixelFormat::Rgb8 => DynamicImage::ImageRgb8(
                ImageBuffer::from_raw(width, height, raw)
                    .ok_or_else(|| Self::buffer_len_error(width, height, 3))?,
            ),
            PixelFormat::Bgr8 => DynamicImage::ImageRgb8(
                ImageBuffer::from_raw(width, height, Self::swap_channels(raw, 3))
                    .ok_or_else(|| Self::buffer_len_error(width, height, 3))?,
            ),
            PixelFormat::Rgba8 => DynamicImage::ImageRgba8(
                ImageBuffer::from_raw(width, height, raw)
                    .ok_or_else(|| Self::buffer_len_error(width, height, 4))?,
            ),
            PixelFormat::Bgra8 => DynamicImage::ImageRgba8(
                ImageBuffer::from_raw(width, height, Self::swap_channels(raw, 4))
                    .ok_or_else(|| Self::buffer_len_error(width, height, 4))?,
            ),
            PixelFormat::Mono8 => DynamicImage::ImageLuma8(
                ImageBuffer::from_raw(width, height, raw)
                    .ok_or_else(|| Self::buffer_len_error(width, height, 1))?,
            ),
        };

        Ok(Self {
            inner,
            format: frame.format,
        })
    }

    /// Loads directly from disk, Usefull when working with image files
    pub fn from_file<P: AsRef<Path>>(path: P) -> Result<Self> {
        let img = image::open(path)?;
        Ok(Self {
            // Force RGB8 to keep our ONNX inputs consistent
            inner: DynamicImage::ImageRgb8(img.into_rgb8()),
            format: PixelFormat::Rgb8,
        })
    }

    /// Builds an image from raw bytes laid out in `format` channel order.
    /// `data.len()` must be exactly `w * h * channels(format)`.
    pub fn from_bytes(width: u32, height: u32, format: PixelFormat, data: Bytes) -> Result<Self> {
        let frame = Frame {
            id: 0,
            width,
            height,
            format,
            data,
            timestamp_ns: 0,
        };
        Self::from_frame(&frame)
    }

    /// Resizes the image to the exact dimensions does not preserve aspect-ratio.
    /// Uses Bilinear (Triangle) filtering.
    pub fn resize_exact(&self, width: u32, height: u32) -> Self {
        let resized = self.inner.resize_exact(width, height, FilterType::Triangle);
        Self {
            inner: resized,
            format: self.format,
        }
    }

    /// Converts the image to the requested pixel format.
    /// `Mono8` targets take the luma channel alpha formats get full opacity.
    pub fn convert(&self, format: PixelFormat) -> Self {
        let inner = match format {
            PixelFormat::Rgb8 => DynamicImage::ImageRgb8(self.inner.to_rgb8()),
            PixelFormat::Bgr8 => DynamicImage::ImageRgb8(self.inner.to_rgb8()),
            PixelFormat::Rgba8 => DynamicImage::ImageRgba8(self.inner.to_rgba8()),
            PixelFormat::Bgra8 => DynamicImage::ImageRgba8(self.inner.to_rgba8()),
            PixelFormat::Mono8 => DynamicImage::ImageLuma8(self.inner.to_luma8()),
        };
        Self { inner, format }
    }

    /// HWC RGB bytes -> NCHW f32 tensor with the requested normalization.
    /// The image is converted to Rgb8 first; dims are whatever the image currently is.
    pub fn to_nchw(&self, norm: Normalization) -> Result<Array4<f32>> {
        let rgb = self.inner.to_rgb8();
        let (width, height) = (rgb.width() as usize, rgb.height() as usize);
        let raw = rgb.as_raw();
        let mut tensor = Array4::<f32>::zeros((1, 3, height, width));

        for y in 0..height {
            for x in 0..width {
                let idx = (y * width + x) * 3;
                let (r, g, b) = (raw[idx] as f32, raw[idx + 1] as f32, raw[idx + 2] as f32);
                match norm {
                    Normalization::SignedUnit => {
                        tensor[[0, 0, y, x]] = (r - 127.5) / 128.0;
                        tensor[[0, 1, y, x]] = (g - 127.5) / 128.0;
                        tensor[[0, 2, y, x]] = (b - 127.5) / 128.0;
                    }
                    Normalization::Unit => {
                        tensor[[0, 0, y, x]] = r / 255.0;
                        tensor[[0, 1, y, x]] = g / 255.0;
                        tensor[[0, 2, y, x]] = b / 255.0;
                    }
                    Normalization::Custom { mean, std } => {
                        let (r, g, b) = (r / 255.0, g / 255.0, b / 255.0);
                        tensor[[0, 0, y, x]] = (r - mean[0]) / std[0];
                        tensor[[0, 1, y, x]] = (g - mean[1]) / std[1];
                        tensor[[0, 2, y, x]] = (b - mean[2]) / std[2];
                    }
                }
            }
        }

        Ok(tensor)
    }

    /// Packs the image back into a pipeline Frame.
    /// Rgb8 is the canonical wire format inside the pipeline.
    pub fn to_frame(&self, id: u64, timestamp_ns: u64) -> Frame {
        let rgb = self.inner.to_rgb8();
        Frame {
            id,
            width: rgb.width(),
            height: rgb.height(),
            format: PixelFormat::Rgb8,
            data: Bytes::from(rgb.into_raw()),
            timestamp_ns,
        }
    }

    pub fn draw_boxes(&mut self, boxes: &[BoundingBox]) {
        let img_buffer = match self.inner.as_mut_rgb8() {
            Some(buf) => buf,
            None => return,
        };

        let (img_width, img_height) = img_buffer.dimensions();
        let color = Rgb([255u8, 0u8, 0u8]);

        for bbox in boxes {
            let x_start = (bbox.x1.max(0.0).round() as u32).min(img_width - 1);
            let y_start = (bbox.y1.max(0.0).round() as u32).min(img_height - 1);
            let x_end = (bbox.x2.max(0.0).round() as u32).min(img_width - 1);
            let y_end = (bbox.y2.max(0.0).round() as u32).min(img_height - 1);

            if x_start >= x_end || y_start >= y_end {
                continue;
            }

            for x in x_start..=x_end {
                img_buffer.put_pixel(x, y_start, color);
                img_buffer.put_pixel(x, y_end, color);
            }
            for y in y_start..=y_end {
                img_buffer.put_pixel(x_start, y, color);
                img_buffer.put_pixel(x_end, y, color);
            }
        }
    }

    /// Swaps R-B in place for 3-channel buffers, R-B for 4-channel. Does not touch the Alpha channel
    fn swap_channels(mut raw: Vec<u8>, channels: usize) -> Vec<u8> {
        for px in raw.chunks_exact_mut(channels) {
            px.swap(0, 2);
        }
        raw
    }

    /// Internal Error Helper
    fn buffer_len_error(width: u32, height: u32, channels: usize) -> anyhow::Error {
        anyhow!(
            "Frame buffer size does not match {} * {} * {}",
            width,
            height,
            channels
        )
    }
}
