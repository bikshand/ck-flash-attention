#include <iostream>
#include <string>

int main(int argc, char* argv[]) {
    std::cout << "========================================" << std::endl;
    std::cout << "  CK Flash Attention - Hello World" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << std::endl;
    
    std::cout << "This is a simple hello world example for the " << std::endl;
    std::cout << "CK (Composable Kernel) based flash attention repository." << std::endl;
    std::cout << std::endl;
    
    std::cout << "Repository: bikshand/ck-flash-attention" << std::endl;
    std::cout << "Purpose: Efficient Flash Attention implementation" << std::endl;
    std::cout << "         built on AMD's Composable Kernel framework" << std::endl;
    std::cout << std::endl;
    
    #ifdef __HIP_PLATFORM_AMD__
    std::cout << "Status: HIP/ROCm support ENABLED" << std::endl;
    #else
    std::cout << "Status: CPU-only mode (HIP/ROCm not available)" << std::endl;
    #endif
    
    std::cout << std::endl;
    std::cout << "Examples available:" << std::endl;
    std::cout << "  - hello_world: This program" << std::endl;
    std::cout << "  - simple_gemm: Basic matrix multiplication using CK" << std::endl;
    std::cout << std::endl;
    
    std::cout << "Build system: CMake" << std::endl;
    std::cout << "C++ Standard: C++17" << std::endl;
    std::cout << std::endl;
    
    std::cout << "========================================" << std::endl;
    std::cout << "  Hello World - Complete!" << std::endl;
    std::cout << "========================================" << std::endl;
    
    return 0;
}
