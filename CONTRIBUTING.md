# Contributing to Oxypipe

First off, thank you for considering contributing to Oxypipe! Oxypipe aims to build a modern, memory-safe, and ultra-fast vision pipeline for the Rust ML ecosystem and we want the community contributions to be a huge part of making that happen.

All types of contributions are welcome: bug reports, performance improvements, documentation updates, feature requests, and pull requests.

---

## Code of Conduct

By participating in this project, you agree to abide by our [Code of Conduct](CODE_OF_CONDUCT.md). Please report unacceptable behavior to <bersissevimli18@gmail.com>.

---

## Getting Started

### Prerequisites

Ensure you have a recent stable version of the Rust toolchain installed via `rustup`:

```bash
rustup update stable
```
Then Clone the repo:
```bash
git clone https://github.com/BersisSe/oxypipe.git
cd oxypipe
```
for building just use `cargo build` and for running the cli just use `cargo run`

## Development
We follow standart Rust conventions keep our codebase clean and maintainable.

1. Formatting & Linting
All code must be formatted using rustfmt. Before committing just run:
```bash
cargo fmt
cargo clippy
```
And Resolve any clippy errors.

2. Testing
Ensure all existing tests pass and add new unit/integration tests for your changes when you see fit.
```bash
cargo test
```

3. Performance
Oxypipe is meant to be performant and efficient while writing your code optimize where you can. Check the Patterns we use to keep oxypipe performant section.


### Patterns we use to keep `oxypipe` performant

To keep Oxypipe fast and lightweight, We follow these development guidelines when writing low-level frame processing or pipeline architecture:

- **Minimize Heap Allocations in Warm Paths:** 
  Avoid allocating memory (`vec![]`, `String`, `Box`) inside the frame processing loop. Reuse buffers across frames or use fixed-size stack arrays where possible.

- **Favor Iterator Chaining & Slices over Raw Indexing:**
  Use rustc's bounds-check elimination by processing pixel buffers using contiguous slice operations or iterators (`chunks_exact`, `zip`). This allows the compiler to auto-vectorize loops via SIMD.
  ```rust
  // Good: Allows LLVM to optimize this to SIMD without bounds checking per pixel
  for chunk in buffer.chunks_exact_mut(4) {
      // No big heap alloc here..
  }
  ```


- **Explicit SIMD & Intrinsics:**
For core array/tensor transformations (like normalization, layout conversions like `HWC` to `CHW`, or color space changes), use the `wide` crate rather than manual scalar math.

- **Zero-Copy Pipeline Transfers:**
Pass frames using shared reference types (`Arc<TensorBuffer>` or `ndarray` views) between async tasks instead of cloning underlying memory blocks.

### Conventions
- **Module Docs**
If you create a new module start the first line with a "//! This module contains ...." this is a documentation convention we use to keep our modules docs similiar.

## AI/LLM Policy
While we are not against AI, We want all the Code in `oxypipe` to be Human tested at least. We only accept **AI Assisted** Code not **AI Written** Code, they are very diffrent. Also AI written PR + Commit messages will not be merged.

Please do not argue with lead maintainers regarding this policy. If requested, rewrite your code or message yourself and resubmit. Inappropriate behavior or hostility regarding these rules will not be tolerated.


## License

By contributing to Oxypipe, you agree that your contributions will be licensed under the project's LGPL-3.0 License.