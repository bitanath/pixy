#include <metal_stdlib>
using namespace metal;

//TODO: This is just a stub, I am building the shaders but the current project does not utilize them

kernel void matmul_byte(
    device float*       out       [[buffer(0)]],
    device const float* x         [[buffer(1)]],
    device const uchar* w         [[buffer(2)]],
    constant uint&      rows      [[buffer(3)]],
    constant uint&      cols      [[buffer(4)]],
    constant uint&      row_size  [[buffer(5)]],
    uint gid [[thread_position_in_grid]])
{
    uint row = gid >> 5;
    if (row >= rows) return;

    uint tid = gid & 31;
    uint nb = cols >> 5;
    float sum = 0.0;

    for (uint b = tid; b < nb; b += 32) {
        uint bo = row * row_size + b * 34;
        float scale = (float)(*(device const half*)(w + bo));

        uint qoff = bo + 2;
        uint xb = b << 5;

        float4 d0(0.0), d1(0.0);

        for (uint j = 0; j < 32; j += 8) {
            float4 x0 = *(device const float4*)(x + xb + j);
            float4 x1 = *(device const float4*)(x + xb + j + 4);

            uchar2 p0 = *(device const uchar2*)(w + qoff + j);
            uchar2 p1 = *(device const uchar2*)(w + qoff + j + 2);
            uchar2 p2 = *(device const uchar2*)(w + qoff + j + 4);
            uchar2 p3 = *(device const uchar2*)(w + qoff + j + 6);
            char4 w0 = {(char)p0.x, (char)p0.y, (char)p1.x, (char)p1.y};
            char4 w1 = {(char)p2.x, (char)p2.y, (char)p3.x, (char)p3.y};

            d0 = fma(x0, float4(w0), d0);
            d1 = fma(x1, float4(w1), d1);
        }

        sum += scale * (d0.x + d0.y + d0.z + d0.w + d1.x + d1.y + d1.z + d1.w);
    }

    float total = simd_sum(sum);
    if (tid == 0) {
        out[row] = total;
    }
}
