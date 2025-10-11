cmake_minimum_required(VERSION 3.16)
project(ck-flash-attention LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ROCm and CK paths
set(ROCM_PATH "/opt/rocm" CACHE PATH "Path to ROCm installation")
set(CK_DIR "${ROCM_PATH}/include/ck" CACHE PATH "Path to Composable Kernel headers")

# Try to find ROCm/HIP
find_package(HIP QUIET)

if(HIP_FOUND)
    message(STATUS "HIP found: ${HIP_VERSION}")
    enable_language(HIP)
    set(CMAKE_HIP_STANDARD 17)
    set(CMAKE_HIP_STANDARD_REQUIRED ON)
    
    # Check for CK headers
    if(EXISTS "${CK_DIR}")
        message(STATUS "Composable Kernel found at: ${CK_DIR}")
        set(CK_AVAILABLE TRUE)
    else()
        message(WARNING "Composable Kernel headers not found at: ${CK_DIR}")
        set(CK_AVAILABLE FALSE)
    endif()
else()
    message(WARNING "HIP not found. Building CPU-only examples.")
    set(CK_AVAILABLE FALSE)
endif()

# Include directories
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
if(CK_AVAILABLE)
    include_directories(${CK_DIR})
    include_directories(${ROCM_PATH}/include)
endif()

# Examples
add_subdirectory(examples)
