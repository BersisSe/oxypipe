//! This module contains fundemental Types in the pipeline
//! From Frames to Embeddings are in this module.

use bytes::Bytes;
use ndarray::Array1;

/// Supported pixel formats for incoming frame buffers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PixelFormat {
    Rgb8,
    Bgr8,
    Rgba8,
    Bgra8,
    Mono8,
}

/// Represents a single video or camera frame entering the pipeline.
#[derive(Debug, Clone)]
pub struct Frame {
    pub id: u64,
    pub width: u32,
    pub height: u32,
    pub format: PixelFormat,
    pub data: Bytes,
    pub timestamp_ns: u64,
}

impl Frame {
    pub fn buffer_size(&self) -> usize {
        let channels = match self.format {
            PixelFormat::Mono8 => 1,
            PixelFormat::Rgb8 | PixelFormat::Bgr8 => 3,
            PixelFormat::Rgba8 | PixelFormat::Bgra8 => 4,
        };
        (self.width * self.height * channels) as usize
    }
}

/// 2D coordinate of a Frame.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Point2D {
    pub x: f32,
    pub y: f32,
}

/// Axis-aligned bounding box with confidence score.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BoundingBox {
    pub x1: f32,
    pub y1: f32,
    pub x2: f32,
    pub y2: f32,
    pub score: f32,
}

impl BoundingBox {
    #[inline]
    pub fn width(&self) -> f32 {
        self.x2 - self.x1
    }

    #[inline]
    pub fn height(&self) -> f32 {
        self.y2 - self.y1
    }

    #[inline]
    pub fn area(&self) -> f32 {
        self.width() * self.height()
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Embedding {
    pub vector: Array1<f32>,
}

impl Embedding {
    pub fn new(vector: Array1<f32>) -> Self {
        Self { vector }
    }

    /// Calculates Cosine Distance (Similarity) assuming the vector is L2-normalized.
    /// Fast dot product auto-vectorized by LLVM SIMD.
    #[inline]
    pub fn cosine_similarity(&self, other: &Embedding) -> f32 {
        self.vector.dot(&other.vector)
    }
}
