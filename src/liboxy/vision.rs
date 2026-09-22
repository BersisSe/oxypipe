//! This module contains Base traits/types and preprocessing utilities for Vision Models
use crate::types::{Embedding, Frame, PixelFormat};
use anyhow::{bail, Result};
use ndarray::{Array1, Array4};
use ort::session::{SessionInputs, SessionOutputs};
use ort::value::Tensor;
use wide::f32x8;

/// Implement this for any model with its own pre/postprocessing.
pub trait VisionModel: Send + Sync + 'static {
    type Input;
    type Output;

    fn preprocess(&self, input: &Self::Input) -> Result<SessionInputs<'static, 'static, 1>>;
    fn postprocess(&self, outputs: SessionOutputs<'_>) -> Result<Self::Output>;
}

/// Pixel normalization. SignedUnit for ArcFace/SCRFD, Unit for YOLO,
/// Custom for anything ImageNet-normalized.
#[derive(Debug, Clone, Copy)]
pub enum Normalization {
    SignedUnit,
    Unit,
    Custom { mean: [f32; 3], std: [f32; 3] },
}

#[derive(Debug, Clone)]
pub struct PreprocessConfig {
    pub target_width: u32,
    pub target_height: u32,
    pub norm: Normalization,
}

/// Frame -> NCHW f32 tensor. Rgb8 frames already at the target size skip
/// straight to `rgb8_to_nchw` anything else gets converted/resized first.
pub fn frame_to_nchw(frame: &Frame, config: &PreprocessConfig) -> Result<Array4<f32>> {
    if frame.format == PixelFormat::Rgb8
        && frame.width == config.target_width
        && frame.height == config.target_height
    {
        return rgb8_to_nchw(frame.data.as_ref(), frame.width, frame.height, config.norm);
    }

    let img = crate::image::Image::from_frame(frame)?
        .resize_exact(config.target_width, config.target_height);
    img.to_nchw(config.norm)
}
/// Internal fn to turn a rgb8 buffer to a NCHW f32 tensor via wide simd.
fn rgb8_to_nchw(raw: &[u8], width: u32, height: u32, norm: Normalization) -> Result<Array4<f32>> {
    let (width, height) = (width as usize, height as usize);
    if raw.len() != width * height * 3 {
        bail!(
            "RGB8 buffer length {} does not match {} * {} * 3",
            raw.len(),
            width,
            height
        );
    }
 
    let (sub, div) = match norm {
        Normalization::SignedUnit => ([127.5, 127.5, 127.5], [128.0, 128.0, 128.0]),
        Normalization::Unit => ([0.0, 0.0, 0.0], [255.0, 255.0, 255.0]),
        Normalization::Custom { mean, std } => (
            [mean[0] * 255.0, mean[1] * 255.0, mean[2] * 255.0],
            [std[0] * 255.0, std[1] * 255.0, std[2] * 255.0],
        ),
    };
 
    let mut tensor = Array4::<f32>::zeros((1, 3, height, width));
    let pixels = width * height;
 
    for c in 0..3 {
        let plane = tensor
            .as_slice_mut()
            .expect("tensor is contiguous");
        let plane = &mut plane[c * pixels..(c + 1) * pixels];
 
        let sub_v = f32x8::splat(sub[c]);
        let div_v = f32x8::splat(div[c]);
 
        let mut i = 0;
        while i + 8 <= pixels {
            let lanes = [
                raw[(i) * 3 + c] as f32,
                raw[(i + 1) * 3 + c] as f32,
                raw[(i + 2) * 3 + c] as f32,
                raw[(i + 3) * 3 + c] as f32,
                raw[(i + 4) * 3 + c] as f32,
                raw[(i + 5) * 3 + c] as f32,
                raw[(i + 6) * 3 + c] as f32,
                raw[(i + 7) * 3 + c] as f32,
            ];
            let normed = (f32x8::new(lanes) - sub_v) / div_v;
            plane[i..i + 8].copy_from_slice(normed.as_array());
            i += 8;
        }
        
        while i < pixels {
            plane[i] = (raw[i * 3 + c] as f32 - sub[c]) / div[c];
            i += 1;
        }
    }
 
    Ok(tensor)
}


#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::PixelFormat;
    use bytes::Bytes;

    fn make_frame(format: PixelFormat, w: u32, h: u32, fill: u8) -> Frame {
        let channels = match format {
            PixelFormat::Mono8 => 1,
            PixelFormat::Rgb8 | PixelFormat::Bgr8 => 3,
            PixelFormat::Rgba8 | PixelFormat::Bgra8 => 4,
        };
        Frame {
            id: 0,
            width: w,
            height: h,
            format,
            data: Bytes::from(vec![fill; (w * h * channels) as usize]),
            timestamp_ns: 0,
        }
    }

    fn config(w: u32, h: u32, norm: Normalization) -> PreprocessConfig {
        PreprocessConfig { target_width: w, target_height: h, norm }
    }

    #[test]
    fn fast_path_and_resize_fallback_agree_on_shape() {
        // exact dims
        let exact = frame_to_nchw(&make_frame(PixelFormat::Rgb8, 4, 2, 255), &config(4, 2, Normalization::SignedUnit)).unwrap();
        assert_eq!(exact.dim(), (1, 3, 2, 4));
        assert!((exact[[0, 0, 0, 0]] - 0.9960937).abs() < 1e-6);

        // mismatched dims
        let resized = frame_to_nchw(&make_frame(PixelFormat::Rgb8, 640, 480, 128), &config(320, 240, Normalization::Unit)).unwrap();
        assert_eq!(resized.dim(), (1, 3, 240, 320));
    }

    #[test]
    fn each_normalization_hits_expected_range() {
        let signed = frame_to_nchw(&make_frame(PixelFormat::Rgb8, 4, 4, 0), &config(4, 4, Normalization::SignedUnit)).unwrap();
        assert!((signed[[0, 1, 0, 0]] + 0.9960937).abs() < 1e-6); 

        let unit = frame_to_nchw(&make_frame(PixelFormat::Rgb8, 4, 4, 255), &config(4, 4, Normalization::Unit)).unwrap();
        assert!((unit[[0, 2, 3, 3]] - 1.0).abs() < 1e-6);

        let custom = frame_to_nchw(
            &make_frame(PixelFormat::Rgb8, 2, 2, 255),
            &config(2, 2, Normalization::Custom { mean: [0.5; 3], std: [0.5; 3] }),
        ).unwrap();
        assert!((custom[[0, 0, 0, 0]] - 1.0).abs() < 1e-6); 
    }

    #[test]
    fn bgr8_matches_rgb8_after_channel_swap() {
        let rgb = frame_to_nchw(&make_frame(PixelFormat::Rgb8, 4, 4, 200), &config(4, 4, Normalization::Unit)).unwrap();
        let bgr = frame_to_nchw(&make_frame(PixelFormat::Bgr8, 4, 4, 200), &config(4, 4, Normalization::Unit)).unwrap();
        assert_eq!(rgb, bgr);
    }

    #[test]
    fn corrupt_buffer_length_errors() {
        let mut frame = make_frame(PixelFormat::Rgb8, 4, 4, 0);
        frame.data = Bytes::from(vec![0u8; 10]);
        assert!(frame_to_nchw(&frame, &config(4, 4, Normalization::Unit)).is_err());
    }
}

pub mod pre {
    use super::*;

    /// SignedUnit-normalized preprocessing closure (used ArcFace, SCRFD).
    pub fn signed_norm(target_w: u32, target_h: u32) -> impl Fn(&Frame) -> Result<SessionInputs<'static, 'static, 1>> + Send + Sync {
        let config = PreprocessConfig {
            target_width: target_w,
            target_height: target_h,
            norm: Normalization::SignedUnit,
        };
        move |frame: &Frame| {
            let tensor = frame_to_nchw(frame, &config)?;
            Ok(ort::inputs![Tensor::from_array(tensor)?].into())
        }
    }

    /// Unit-normalized preprocessing closure (used by YOLO).
    pub fn unit_norm(target_w: u32, target_h: u32) -> impl Fn(&Frame) -> Result<SessionInputs<'static, 'static, 1>> + Send + Sync {
        let config = PreprocessConfig {
            target_width: target_w,
            target_height: target_h,
            norm: Normalization::Unit,
        };
        move |frame: &Frame| {
            let tensor = frame_to_nchw(frame, &config)?;
            Ok(ort::inputs![Tensor::from_array(tensor)?].into())
        }
    }
}

pub mod post {
    use super::*;

    /// Pulls a flat embedding vector out of one named/indexed output.
    pub fn extract_1d_vector(
        output_idx: usize,
    ) -> impl Fn(SessionOutputs<'_>) -> Result<Embedding> + Send + Sync {
        move |outputs: SessionOutputs| {
            let (_shape, slice) = outputs[output_idx].try_extract_tensor::<f32>()?;
            let vector = Array1::from_vec(slice.to_vec());
            Ok(Embedding::new(vector))
        }
    }
}