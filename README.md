# Oxypipe
Oxypipe is a highly performant and efficient vision pipeline built natively in Rust. Designed for Edge and Mobile hardware here to advance Rust ML Ecosystem and be a alternative for memory heavy Python, OpenCV, PyTorch  

[![Crates.io](https://img.shields.io/crates/v/oxypipe.svg)](https://crates.io/crates/oxypipe)
[![Documentation](https://docs.rs/oxypipe/badge.svg)](https://docs.rs/oxypipe)
[![License: LGPL v3](https://img.shields.io/badge/License-LGPL_v3-blue.svg)](LICENSE.md)

## Why Rust?

- **No GIL, no interpreter** : compiled code, not Python glue around a C++ core.
- **Memory safety without GC pauses** : no pauses when a frame is on the way.
- **A genuinely async pipeline** : capture and inference overlap instead of blocking each other.
- **SIMD where we can, SIMT where we can't** : vectorized CPU paths, CUDA/ROCm for the rest.

## Core Features

- **Run ONNX, anywhere** : write your pipeline once, accelerate it via CUDA, CoreML, or any other ONNX execution provider.
- **Zero-cost, async-first architecture** : capture, pre-process, infer, and post-process without blocking each other.
- **SIMD-accelerated CPU paths** : vectorized where the CPU does the work; SIMT (CUDA/ROCm) handles the rest.
- **Native source interfaces** : webcams, RTSP streams, and image files, accessed the way you'd expect from OpenCV, without the Python bridge.

## Execution Providers & Backend Support

| Provider | CPU / SIMD | CUDA | CoreML | ROCm | DirectML |
| --- | --- | --- | --- | --- | --- |
| **Status** | ✅ Supported | 🚧 Planned | 🚧 Planned | 🚧 Planned | 🚧 Planned |

## Quickstart

```bash
cargo new --bin vision
cd vision
cargo add oxypipe anyhow
cargo add -F full tokio
```

### 1. Capture a frame

```rust
use oxypipe::sources::Camera;

#[tokio::main]
async fn main() {
    let cam = Camera::with_index(0);
    let image = cam.get_image().unwrap();
    image.inner.save("image.jpg").unwrap();
}
```

### 2. Load an ONNX model

```rust
use anyhow::Result;
use oxypipe::engine::EngineBuilder;
use oxypipe::types::{Frame, BoundingBox};
use oxypipe::vision::pre::signed_norm;

fn main() -> Result<()> {
    let engine = EngineBuilder::<Frame, Vec<BoundingBox>>::new("model.onnx")
        .preprocess(signed_norm(320, 240))
        .postprocess(|_outputs| Ok(Vec::new())) 
        .build()?;

    Ok(())
}
```

### 3. Run a minimal pipeline

A `Pipeline` chains stages with `.map()`; each stage runs concurrently with the next.

```rust
use anyhow::Result;
use oxypipe::pipeline::Pipeline;
use oxypipe::sources::{File, OneShot};

#[tokio::main]
async fn main() -> Result<()> {
    let frame = File::new("test.jpg").load()?;

    let mut pipe = Pipeline::from_frame(frame)
        .map(|f| Ok(f.width * f.height)); // any CPU-bound stage

    while let Some(result) = pipe.next().await {
        println!("{}", result?);
    }

    Ok(())
}
```

### 4. Full example: face detection

Capture → preprocess → infer → decode → draw. `oxypipe::detect::decode` handles the generic anchor-decoding + NMS math; reading your model's specific output tensors is a few lines of user code.

```rust
use anyhow::{anyhow, Result};
use ndarray::ArrayView2;
use oxypipe::detect::decode;
use oxypipe::engine::EngineBuilder;
use oxypipe::image::Image;
use oxypipe::pipeline::Pipeline;
use oxypipe::sources::{File, OneShot};
use oxypipe::types::Frame;
use oxypipe::vision::pre::signed_norm;

#[tokio::main]
async fn main() -> Result<()> {
    let frame = File::new("test.jpg").load()?;
    let (src_w, src_h) = (frame.width, frame.height);

    let mut engine = EngineBuilder::<Frame, Vec<oxypipe::types::BoundingBox>>::new("ultraface.onnx")
        .preprocess(signed_norm(320, 240))
        .postprocess(move |outputs| {
            let (_, scores) = outputs["scores"].try_extract_tensor::<f32>()
                .map_err(|e| anyhow!("failed to extract scores: {e}"))?;
            let (_, boxes) = outputs["boxes"].try_extract_tensor::<f32>()
                .map_err(|e| anyhow!("failed to extract boxes: {e}"))?;

            let anchors = scores.len() / 2;
            let scores = ArrayView2::from_shape((anchors, 2), scores)?;
            let boxes = ArrayView2::from_shape((anchors, 4), boxes)?;

            Ok(decode(&scores, &boxes, src_w, src_h, 0.7, 0.3))
        })
        .build()?;

    let mut pipe = Pipeline::from_frame(frame)
        .map(move |f| engine.run(&f));

    let mut boxes = Vec::new();
    while let Some(result) = pipe.next().await {
        boxes = result?;
    }

    println!("{} face(s) detected", boxes.len());

    let mut img = Image::from_file("test.jpg")?;
    img.draw_boxes(&boxes);
    img.inner.save("out.jpg")?;

    Ok(())
}
```

## How It Works

ONNX Runtime executes the model; Oxypipe focuses on everything around it resize, normalize, NMS, and the glue from capture to detection accelerated via SIMD/SIMT and built on zero-cost, memory-safe Rust!

## Contributing

Thanks for considering it. See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

Oxypipe is licensed under LGPL-3.0. See [LICENSE.md](LICENSE.md).