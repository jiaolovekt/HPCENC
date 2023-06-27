__device__ static const int width = ${width};
__device__ static const int height = ${height};
__device__ static const float sigmaS = ${sigmaS};
__device__ static const float sigmaR = ${sigmaR};
__device__ static const int radius = ${radius};
__device__ static const bool use_shared_memory = ${use_shared_memory};
__device__ static const bool is_float = ${is_float};
__device__ static const float peak = ${peak_value};

__device__ static const int kernel_size_x = 2 * radius + ${block_x};
__device__ static const int kernel_size_y = 2 * radius + ${block_y};

extern "C"
__global__ void bilateral(const ${data_type} * __restrict__ src, ${data_type} * __restrict__ dst) {
    const int x = threadIdx.x + blockIdx.x * blockDim.x;
    const int y = threadIdx.y + blockIdx.y * blockDim.y;

    float num {};
    float den {};

    float center {};
    float value {};

    if constexpr (use_shared_memory) {
        extern __shared__ ${data_type} buffer[];
        
        for (int cy = threadIdx.y; cy < kernel_size_y; cy += blockDim.y) {
            int sy = min(max(cy - static_cast<int>(threadIdx.y) - radius + y, 0), height - 1);
            for (int cx = threadIdx.x; cx < kernel_size_x; cx += blockDim.x) {
                int sx = min(max(cx - static_cast<int>(threadIdx.x) - radius + x, 0), width - 1);
                buffer[cy * kernel_size_x + cx] = src[sy * width + sx];
            }
        }

        __syncthreads();

        if (x >= width || y >= height)
            return;

        if constexpr (is_float) {
            center = src[y * width + x];
        } else {
            center = (float)(src[y * width + x]) / peak;
        }
        
        #pragma unroll 4
        for (int cy = -radius; cy <= radius; ++cy) {
            int sy = cy + radius + threadIdx.y;
            
            #pragma unroll 4
            for (int cx = -radius; cx <= radius; ++cx) {
                int sx = cx + radius + threadIdx.x;

                if constexpr (is_float) {
                    value = buffer[sy * kernel_size_x + sx];
                } else {
                    value = (float)(buffer[sy * kernel_size_x + sx]) / peak;
                }
                
                float range = value - center;
                
                float weight = exp2f((cy * cy + cx * cx) * sigmaS + (range * range) * sigmaR);
                
                num += weight * value;
                den += weight;
            }
        }
        
        if constexpr (is_float) {
            dst[y * width + x] = num / den;
        } else {
            dst[y * width + x] = __float2int_rn(num / den * peak);
        }
    } else {
        if (x >= width || y >= height)
            return;

        if constexpr (is_float) {
            center = src[y * width + x];
        } else {
            center = (float)(src[y * width + x]) / peak;
        }
        
        #pragma unroll 4
        for (int cy = max(y - radius, 0); cy <= min(y + radius, height); ++cy) {
            #pragma unroll 4
            for (int cx = max(x - radius, 0); cx <= min(x + radius, width); ++cx) {
                if constexpr (is_float) {
                    value = src[cy * width + cx];
                } else {
                    value = (float)(src[cy * width + cx]) / peak;
                }
                
                float range = value - center;
                
                float weight = exp2f(
                    ((y - cy) * (y - cy) + (x - cx) * (x - cx)) * sigmaS + (range * range) * sigmaR
                );

                num += weight * value;
                den += weight;
            }
        }
        
        if constexpr (is_float) {
            dst[y * width + x] = num / den;
        } else {
            dst[y * width + x] = __float2int_rn(num / den * peak);
        }
    }
}