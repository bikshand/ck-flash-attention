# ck-flash-attention
Efficient Flash Attention implementation built on AMD's Composable Kernel (CK) framework for high-performance Transformer Architectures.

## Overview

This repository provides examples and utilities for working with AMD's Composable Kernel (CK) framework, including:
- Simple hello world example
- Basic GEMM (General Matrix Multiply) implementation
- Foundation for flash attention kernels

## Prerequisites

### Required
- CMake 3.16 or higher
- C++17 compatible compiler (g++ 7.0+, clang 5.0+)

### Optional (for GPU acceleration)
- ROCm 5.0 or higher
- HIP runtime
- AMD GPU with ROCm support (gfx908, gfx90a, gfx942, etc.)

## Building

### Quick Start

```bash
# Clone the repository
git clone https://github.com/bikshand/ck-flash-attention.git
cd ck-flash-attention

# Create build directory
mkdir build && cd build

# Configure and build
cmake ..
make

# Run examples
./examples/hello_world
./examples/simple_gemm  # If ROCm/HIP is available
```

### Build Options

The project automatically detects ROCm/HIP availability:
- **With ROCm/HIP**: Builds both CPU and GPU examples
- **Without ROCm/HIP**: Builds CPU-only hello_world example

To specify a custom ROCm path:
```bash
cmake -DROCM_PATH=/path/to/rocm ..
```

To specify a custom Composable Kernel path:
```bash
cmake -DCK_DIR=/path/to/ck/include ..
```

## Examples

### 1. Hello World
A simple program that displays information about the project and build configuration.

```bash
./examples/hello_world
```

**Output:**
```
========================================
  CK Flash Attention - Hello World
========================================

This is a simple hello world example for the 
CK (Composable Kernel) based flash attention repository.

Repository: bikshand/ck-flash-attention
Purpose: Efficient Flash Attention implementation
         built on AMD's Composable Kernel framework

Status: CPU-only mode (HIP/ROCm not available)

Examples available:
  - hello_world: This program
  - simple_gemm: Basic matrix multiplication using CK

Build system: CMake
C++ Standard: C++17

========================================
  Hello World - Complete!
========================================
```

### 2. Simple GEMM
A basic matrix multiplication example using HIP for GPU acceleration.

**Features:**
- Performs C = α·A·B + β·C operation
- Default matrix size: 512×512
- CPU reference implementation for verification
- Automatic fallback to CPU if no GPU is available

```bash
./examples/simple_gemm
```

**Output (with GPU):**
```
========================================
  Simple GEMM Example
========================================

Matrix dimensions:
  A: 512 x 512
  B: 512 x 512
  C: 512 x 512
  alpha: 1, beta: 0

Matrices initialized with random values

HIP device found: AMD Radeon GPU
  Compute capability: 9.0

Computing on GPU...
GPU computation complete!

Computing on CPU for verification...
CPU computation complete!

Verifying results...
✓ Results match! GEMM is correct.

Sample results (first 4 elements of C):
  C[0] = -12.3456 (CPU: -12.3456)
  C[1] = 8.7654 (CPU: 8.7654)
  C[2] = -3.2109 (CPU: -3.2109)
  C[3] = 15.6789 (CPU: 15.6789)

========================================
  GEMM Example Complete!
========================================
```

## Project Structure

```
ck-flash-attention/
├── CMakeLists.txt          # Main build configuration
├── README.md               # This file
├── .gitignore             # Git ignore patterns
├── include/               # Header files (future use)
├── src/                   # Source files (future use)
└── examples/              # Example programs
    ├── CMakeLists.txt     # Examples build configuration
    ├── hello_world.cpp    # Hello world example
    └── simple_gemm.cpp    # GEMM example
```

## Development

### Adding New Examples

1. Create a new `.cpp` file in the `examples/` directory
2. Add the executable to `examples/CMakeLists.txt`:
   ```cmake
   add_executable(my_example my_example.cpp)
   ```
3. For GPU examples, configure HIP:
   ```cmake
   set_source_files_properties(my_example.cpp PROPERTIES LANGUAGE HIP)
   target_link_libraries(my_example PRIVATE hip::device)
   ```

### Code Style
- C++17 standard
- Use meaningful variable names
- Include comments for complex operations
- Follow existing code patterns

## Future Work

- [ ] Implement flash attention kernel using CK
- [ ] Add performance benchmarking utilities
- [ ] Support for different data types (fp16, bf16)
- [ ] Multi-GPU support
- [ ] Python bindings

## Resources

- [AMD Composable Kernel (CK)](https://github.com/ROCmSoftwarePlatform/composable_kernel)
- [ROCm Documentation](https://rocmdocs.amd.com/)
- [HIP Programming Guide](https://rocmdocs.amd.com/en/latest/Programming_Guides/HIP-GUIDE.html)

## License

This project is provided as-is for educational and development purposes.

## Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.
