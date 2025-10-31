#include <cstdio>
#include <cuda_runtime.h>

int main() {
    int count = 0;
    cudaError_t err = cudaGetDeviceCount(&count);
    if (err != cudaSuccess) {
        std::fprintf(stderr, "cudaGetDeviceCount error: %s (%d)\n",
                     cudaGetErrorString(err), static_cast<int>(err));
        return 1;
    }
    std::printf("device count: %d\n", count);
    return 0;
}
