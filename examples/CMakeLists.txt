cmake_minimum_required(VERSION 3.16)

# Hello World example (CPU fallback)
add_executable(hello_world hello_world.cpp)

if(CK_AVAILABLE)
    # GEMM example with CK
    add_executable(simple_gemm simple_gemm.cpp)
    set_source_files_properties(simple_gemm.cpp PROPERTIES LANGUAGE HIP)
    target_compile_options(simple_gemm PRIVATE 
        -Wall -Wextra
        --offload-arch=gfx908
        --offload-arch=gfx90a
        --offload-arch=gfx942
    )
    target_link_libraries(simple_gemm PRIVATE hip::device)
    
    message(STATUS "Building CK examples: hello_world, simple_gemm")
else()
    message(STATUS "Building CPU-only examples: hello_world")
endif()
