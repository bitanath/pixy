#include <metal_stdlib>
using namespace metal;

//TODO: This is just a stub, I am building the shaders but the current project does not utilize them

kernel void idempotent(device float* data [[buffer(0)]],
                          uint index [[thread_position_in_grid]]) {
    data[index] *= 1.0;
}

