#include <metal_stdlib>
using namespace metal;

kernel void dequantize_byte(
    device const uchar* src        [[buffer(0)]],
    device float*       dst        [[buffer(1)]],
    constant uint&      src_offset [[buffer(2)]],
    constant uint&      dst_offset [[buffer(3)]],
    constant uint&      nb         [[buffer(4)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= nb) return;

    uint bo = src_offset + gid * 34;
    float scale = (float)(*(device const half*)(src + bo));

    uint qoff = bo + 2;
    uint out_off = dst_offset + gid * 32;

    for (uint j = 0; j < 32; j += 8) {
        uchar2 p0 = *(device const uchar2*)(src + qoff + j);
        uchar2 p1 = *(device const uchar2*)(src + qoff + j + 2);
        uchar2 p2 = *(device const uchar2*)(src + qoff + j + 4);
        uchar2 p3 = *(device const uchar2*)(src + qoff + j + 6);

        char4 w0 = {(char)p0.x, (char)p0.y, (char)p1.x, (char)p1.y};
        char4 w1 = {(char)p2.x, (char)p2.y, (char)p3.x, (char)p3.y};

        float4 v0 = float4(w0) * scale;
        float4 v1 = float4(w1) * scale;

        dst[out_off + j + 0] = v0.x;
        dst[out_off + j + 1] = v0.y;
        dst[out_off + j + 2] = v0.z;
        dst[out_off + j + 3] = v0.w;
        dst[out_off + j + 4] = v1.x;
        dst[out_off + j + 5] = v1.y;
        dst[out_off + j + 6] = v1.z;
        dst[out_off + j + 7] = v1.w;
    }
}