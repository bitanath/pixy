#include <metal_stdlib>
using namespace metal;

kernel void double_values(device float* data [[buffer(0)]],
                          uint index [[thread_position_in_grid]]) {
    data[index] *= 2.0;
}

