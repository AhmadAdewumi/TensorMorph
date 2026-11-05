
cmake_minimum_required(VERSION 3.20)

project(TensorMorph
    VERSION 0.1.0
    LANGUAGES CXX
)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

include(CTest)
include(CheckLanguage)

option(TENSORMORPH_ENABLE_CUDA "Build CUDA backend" OFF)

# Core library
add_library(tensormorph
    src/tensor.cpp
    src/operations.cpp
    src/cpu/elementwise.cpp
    src/cpu/reduction.cpp
    src/cpu/matmul.cpp
)

target_include_directories(tensormorph
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_compile_options(tensormorph PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic;-Wconversion;-Wshadow>
)

# Optional CUDA backend
if(TENSORMORPH_ENABLE_CUDA)

    check_language(CUDA)

    if(NOT CMAKE_CUDA_COMPILER)
        message(FATAL_ERROR "CUDA requested, but nvcc was not found.")
    endif()

    enable_language(CUDA)

    set(CMAKE_CUDA_STANDARD 20)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)

    # Default architecture if none was supplied.
    if(NOT CMAKE_CUDA_ARCHITECTURES)
        set(CMAKE_CUDA_ARCHITECTURES 75)
    endif()

    # tensormorph was created before CUDA was enabled.
    set_property(TARGET tensormorph
        PROPERTY CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}"
    )

    find_package(CUDAToolkit REQUIRED)

    target_sources(tensormorph PRIVATE
        src/cuda/memory.cu
        src/cuda/runtime.cu
        src/cuda/stream.cu
        src/cuda/event.cu
        src/cuda/element_wise.cu
        src/cuda/reduction.cu
        src/cuda/matmul.cu
        src/cuda/transpose.cu
    )

    target_compile_definitions(tensormorph
        PUBLIC TENSORMORPH_HAS_CUDA=1
    )

    target_link_libraries(tensormorph
        PUBLIC CUDA::cudart
    )

    set_target_properties(tensormorph PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        POSITION_INDEPENDENT_CODE ON
    )

endif()

# Tests
if(BUILD_TESTING)
    add_executable(tensormorph_tests
        tests/test_main.cpp
    )

    target_link_libraries(tensormorph_tests
        PRIVATE tensormorph
    )

    add_test(
        NAME tensormorph_tests
        COMMAND tensormorph_tests
    )
endif()

# Benchmarks
add_executable(vector_add_bench
    benchmarks/vector_add_bench.cpp
)

target_link_libraries(vector_add_bench
    PRIVATE tensormorph
)

add_executable(elementwise_bench
    benchmarks/elementwise_bench.cpp
)

target_link_libraries(elementwise_bench
    PRIVATE tensormorph
)

add_executable(matmul_bench
    benchmarks/matmul_bench.cpp
)

target_link_libraries(matmul_bench
    PRIVATE tensormorph
)
add_executable(cuda_elementwise_bench
        benchmarks/cuda_elementwise_bench.cu
    )

target_link_libraries(cuda_elementwise_bench PRIVATE tensormorph)
