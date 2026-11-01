
cmake_minimum_required(VERSION 3.20)

project(TensorMorph
    VERSION 0.1.0
    LANGUAGES CXX
)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

include(CTest)

add_library(tensormorph
    src/tensor.cpp
    # src/operations.cpp
    src/cpu/elementwise.cpp
    src/cpu/reduction.cpp
    src/cpu/matmul.cpp
)

target_include_directories(tensormorph
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(tensormorph PRIVATE
        -Wall
        -Wextra
        -Wpedantic
        -Wconversion
        -Wshadow
    )
endif()

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

add_executable(vector_add_bench
    benchmarks/vector_add_bench.cpp
)

target_link_libraries(vector_add_bench
    PRIVATE tensormorph
)

add_executable(elementwise_bench benchmarks/elementwise_bench.cpp)
target_link_libraries(elementwise_bench PRIVATE tensormorph)

add_executable(matmul_bench benchmarks/matmul_bench.cpp)
target_link_libraries(matmul_bench PRIVATE tensormorph)