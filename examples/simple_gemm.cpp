#include <iostream>
#include <vector>
#include <random>
#include <hip/hip_runtime.h>

// Simple GEMM kernel using HIP
// C = alpha * A * B + beta * C
// A: M x K, B: K x N, C: M x N
__global__ void simple_gemm_kernel(
    const float* A,
    const float* B,
    float* C,
    int M,
    int N,
    int K,
    float alpha,
    float beta)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; ++k) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = alpha * sum + beta * C[row * N + col];
    }
}

// Host function to perform GEMM
void gemm_host(
    const std::vector<float>& A,
    const std::vector<float>& B,
    std::vector<float>& C,
    int M, int N, int K,
    float alpha, float beta)
{
    // Allocate device memory
    float *d_A, *d_B, *d_C;
    size_t size_A = M * K * sizeof(float);
    size_t size_B = K * N * sizeof(float);
    size_t size_C = M * N * sizeof(float);
    
    hipMalloc(&d_A, size_A);
    hipMalloc(&d_B, size_B);
    hipMalloc(&d_C, size_C);
    
    // Copy data to device
    hipMemcpy(d_A, A.data(), size_A, hipMemcpyHostToDevice);
    hipMemcpy(d_B, B.data(), size_B, hipMemcpyHostToDevice);
    hipMemcpy(d_C, C.data(), size_C, hipMemcpyHostToDevice);
    
    // Launch kernel
    dim3 blockDim(16, 16);
    dim3 gridDim((N + blockDim.x - 1) / blockDim.x,
                 (M + blockDim.y - 1) / blockDim.y);
    
    hipLaunchKernelGGL(simple_gemm_kernel, gridDim, blockDim, 0, 0,
                       d_A, d_B, d_C, M, N, K, alpha, beta);
    
    // Copy result back to host
    hipMemcpy(C.data(), d_C, size_C, hipMemcpyDeviceToHost);
    
    // Free device memory
    hipFree(d_A);
    hipFree(d_B);
    hipFree(d_C);
}

// Simple CPU reference implementation
void gemm_cpu_reference(
    const std::vector<float>& A,
    const std::vector<float>& B,
    std::vector<float>& C,
    int M, int N, int K,
    float alpha, float beta)
{
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k) {
                sum += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] = alpha * sum + beta * C[i * N + j];
        }
    }
}

// Verify results
bool verify_results(const std::vector<float>& C_gpu,
                   const std::vector<float>& C_cpu,
                   float tolerance = 1e-4f)
{
    for (size_t i = 0; i < C_gpu.size(); ++i) {
        float diff = std::abs(C_gpu[i] - C_cpu[i]);
        if (diff > tolerance) {
            std::cerr << "Mismatch at index " << i << ": "
                     << "GPU=" << C_gpu[i] << " CPU=" << C_cpu[i]
                     << " diff=" << diff << std::endl;
            return false;
        }
    }
    return true;
}

int main(int argc, char* argv[]) {
    std::cout << "========================================" << std::endl;
    std::cout << "  Simple GEMM Example" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << std::endl;
    
    // Matrix dimensions
    const int M = 512;  // Rows of A and C
    const int N = 512;  // Columns of B and C
    const int K = 512;  // Columns of A, rows of B
    const float alpha = 1.0f;
    const float beta = 0.0f;
    
    std::cout << "Matrix dimensions:" << std::endl;
    std::cout << "  A: " << M << " x " << K << std::endl;
    std::cout << "  B: " << K << " x " << N << std::endl;
    std::cout << "  C: " << M << " x " << N << std::endl;
    std::cout << "  alpha: " << alpha << ", beta: " << beta << std::endl;
    std::cout << std::endl;
    
    // Initialize matrices
    std::vector<float> A(M * K);
    std::vector<float> B(K * N);
    std::vector<float> C_gpu(M * N, 0.0f);
    std::vector<float> C_cpu(M * N, 0.0f);
    
    // Fill with random values
    std::random_device rd;
    std::mt19937 gen(42);  // Fixed seed for reproducibility
    std::uniform_real_distribution<float> dis(-1.0f, 1.0f);
    
    for (auto& val : A) val = dis(gen);
    for (auto& val : B) val = dis(gen);
    
    std::cout << "Matrices initialized with random values" << std::endl;
    std::cout << std::endl;
    
    // Check HIP device
    int deviceCount = 0;
    hipError_t err = hipGetDeviceCount(&deviceCount);
    
    if (err != hipSuccess || deviceCount == 0) {
        std::cout << "No HIP devices found. Running CPU reference only." << std::endl;
        std::cout << std::endl;
        
        std::cout << "Computing on CPU..." << std::endl;
        gemm_cpu_reference(A, B, C_cpu, M, N, K, alpha, beta);
        std::cout << "CPU computation complete!" << std::endl;
        std::cout << std::endl;
        
        std::cout << "Sample results (first 4 elements of C):" << std::endl;
        for (int i = 0; i < 4 && i < static_cast<int>(C_cpu.size()); ++i) {
            std::cout << "  C[" << i << "] = " << C_cpu[i] << std::endl;
        }
    } else {
        hipDeviceProp_t prop;
        hipGetDeviceProperties(&prop, 0);
        std::cout << "HIP device found: " << prop.name << std::endl;
        std::cout << "  Compute capability: " << prop.major << "." << prop.minor << std::endl;
        std::cout << std::endl;
        
        std::cout << "Computing on GPU..." << std::endl;
        gemm_host(A, B, C_gpu, M, N, K, alpha, beta);
        std::cout << "GPU computation complete!" << std::endl;
        std::cout << std::endl;
        
        std::cout << "Computing on CPU for verification..." << std::endl;
        gemm_cpu_reference(A, B, C_cpu, M, N, K, alpha, beta);
        std::cout << "CPU computation complete!" << std::endl;
        std::cout << std::endl;
        
        std::cout << "Verifying results..." << std::endl;
        if (verify_results(C_gpu, C_cpu)) {
            std::cout << "✓ Results match! GEMM is correct." << std::endl;
        } else {
            std::cout << "✗ Results don't match! There's an error." << std::endl;
        }
        std::cout << std::endl;
        
        std::cout << "Sample results (first 4 elements of C):" << std::endl;
        for (int i = 0; i < 4 && i < static_cast<int>(C_gpu.size()); ++i) {
            std::cout << "  C[" << i << "] = " << C_gpu[i] 
                     << " (CPU: " << C_cpu[i] << ")" << std::endl;
        }
    }
    
    std::cout << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "  GEMM Example Complete!" << std::endl;
    std::cout << "========================================" << std::endl;
    
    return 0;
}
