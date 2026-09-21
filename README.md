# Oxypipe Modern, Efficient, Portable ONNX Pipeline
Oxypipe is a highly performant and efficient vision pipeline built natively in Rust. Designed for Edge and Mobile hardware here to advence Rust ML Ecosystem and be a alternative for memory heavy Python, OpenCV, PyTorch

## Core Features
- **Run ONNX** : Write once, run anywhere. Accelerate via CoreML, Cuda Or any other ONNX provider.
- **Zero Cost Abstraction + Concurrency** : By using Rust and its _Zero Cost Abstractions_ and the architecture is asynchronous from capture to detection 
- **SIMD Where we can** : When calculations happen on cpu they are acceleared via SIMD Instruction on gpu the Cuda/ROCm interface already uses SIMT
- **Native Source Interfaces** : Access webcams, RTSP streams file system exactly like OpenCV with python.