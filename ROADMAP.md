# Roadmap

## Vision
Oxypipe will be the Rust standard for vision ML applications. For its convenience, speed, and familiarity enough to Python users that switching doesn't feel like a rewrite.

## Now
- [ ] Stabilize the core API (`VisionModel`, `EngineBuilder`, pre/postprocess boundaries)
- [ ] CUDA + CPU execution providers hardened for real workloads

## Next
- [ ] CoreML, ROCm, and DirectML execution providers
- [ ] Visual output, `cv.imshow()`-style, for quick debugging of a pipeline's output
- [ ] More source types (beyond camera/RTSP/file)

## Later
- [ ] Benchmark suite vs. OpenCV + Python, published and reproducible
- [ ] Mobile cross-compilation docs (iOS, Android)