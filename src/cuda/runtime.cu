#include "tensormorph/cuda/runtime.hpp"
#include "../include/tensormorph/cuda/error.hpp"

#include <cuda_runtime.h>

#include <stdexcept>

namespace tensormorph::cuda {

int device_count() {
    int count = 0;
    TM_CUDA_CHECK(cudaGetDeviceCount(&count));
    return count;
}

DeviceInfo device_info(int device_index) {
    if (device_index < 0 || device_index >= device_count()) {
        throw std::out_of_range("Invalid CUDA device index");
    }

    cudaDeviceProp properties{};
    TM_CUDA_CHECK(cudaGetDeviceProperties(&properties, device_index));

    DeviceInfo info;

    info.id = device_index;
    info.name = properties.name;

    info.compute_major = properties.major;
    info.compute_minor = properties.minor;

    info.multiprocessor_count = properties.multiProcessorCount;
    info.warp_size = properties.warpSize;

    info.total_global_memory = properties.totalGlobalMem;
    info.shared_memory_per_block = properties.sharedMemPerBlock;

    info.registers_per_block = properties.regsPerBlock;
    info.async_engine_count = properties.asyncEngineCount;
    info.concurrent_kernels = properties.concurrentKernels;

    return info;
}

void set_device(int device_index) {
    if (device_index < 0 || device_index >= device_count()) {
        throw std::out_of_range("Invalid CUDA device index");
    }

    TM_CUDA_CHECK(cudaSetDevice(device_index));
}

void synchronize(int device_index) {
    set_device(device_index);
    TM_CUDA_CHECK(cudaDeviceSynchronize());
}

}