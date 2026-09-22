# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- `engine`: closure-based `EngineBuilder` (`preprocess`/`postprocess`) with
  `VisionModel` adapter via `with_vision_model` sessions with full graph optimization.
- `vision`: `VisionModel` trait, `Normalization` (`SignedUnit`/`Unit`/`Custom`),
  `ImagePreprocessor::frame_to_nchw` with automatic resize fallback, and
  `pre::{signed_norm, unit_norm}` / `post::extract_1d_vector` helpers.
- `image`: `Image` wrapper with `from_frame` (all `PixelFormat`s),
  `from_file`, `from_bytes`, `resize_exact`, `convert`, `to_nchw`,
  `to_frame`, and `draw_boxes` box annotation.
- `pipeline`: `from_source` / `from_frame` / `from_receiver` entry points,
  `FnMut` CPU-bound stages on `spawn_blocking` with bounded-channel
  backpressure; stage errors propagate as values and short-circuit
  downstream stages.
- `sources`: split `Source` (continuous feeds) and `OneShotSource`
  traits; `Camera` (feature-gated) and `File::load` implementors.
- `detect`: decoupled UltraFace decode — `iou`, greedy `nms`, `decode`
  (threshold -> clip -> scale -> NMS), and `ultraface_postprocess` closure.
- `types`: `Frame`, `PixelFormat`, `BoundingBox`, `Embedding`
  (`cosine_similarity`), `Point2D`.
- Demo binary: `test.jpg` through `ultraface.onnx` prints `Vec<BoundingBox>`
  and writes annotated `out.jpg`.
- 20 unit tests (`vision`, `pipeline`, `detect` suites).

