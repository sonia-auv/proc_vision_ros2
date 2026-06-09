#include "proc_vision_ros2/preprocess.h"
#include "proc_vision_ros2/cuda_utils.h"
#include "device_launch_parameters.h"

// Host and device pointers for image buffers
static uint8_t* img_buffer_host = nullptr;    // Pinned memory on the host for faster transfers
static uint8_t* img_buffer_device = nullptr;  // Memory on the device (GPU)

// CUDA kernel to perform affine warp on the image
__global__ void letterboxNormalizeKernel(
    const uint8_t* __restrict__ src,   // HWC BGR source image (device)
    float*         __restrict__ dst,   // NCHW RGB float output
    int srcW, int srcH,                // source dimensions
    int dstW, int dstH,                // destination (letterbox) dimensions
    int newW, int newH,                // resized source dimensions
    int padLeft, int padTop,           // padding offsets
    float normScale,                   // 1/255
    float padValue                     // normalised pad colour (114/255)
) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x; // dst column
    const int y = blockIdx.y * blockDim.y + threadIdx.y; // dst row

    if (x >= dstW || y >= dstH) return;

    const int planeSize = dstH * dstW;
    const int idx = y * dstW + x;

    // Check if this pixel is inside the resized source region
    const int srcX = x - padLeft;
    const int srcY = y - padTop;

    if (srcX >= 0 && srcX < newW && srcY >= 0 && srcY < newH) {
        // Bilinear coordinate mapping back to original source
        // src coords (float) for bilinear interpolation
        const float fx = static_cast<float>(srcX) * srcW / static_cast<float>(newW);
        const float fy = static_cast<float>(srcY) * srcH / static_cast<float>(newH);

        const int x0 = static_cast<int>(fx);
        const int y0 = static_cast<int>(fy);
        const int x1 = min(x0 + 1, srcW - 1);
        const int y1 = min(y0 + 1, srcH - 1);

        const float ax = fx - x0;
        const float ay = fy - y0;

        // Read 4 corners (BGR, HWC layout)
        const int stride = srcW * 3;
        const uint8_t* p00 = src + y0 * stride + x0 * 3;
        const uint8_t* p01 = src + y0 * stride + x1 * 3;
        const uint8_t* p10 = src + y1 * stride + x0 * 3;
        const uint8_t* p11 = src + y1 * stride + x1 * 3;

        // Bilinear interpolation per channel, BGR→RGB + normalize
        #pragma unroll
        for (int c = 0; c < 3; ++c) {
            // BGR channel index: 0=B, 1=G, 2=R
            // RGB output plane:  0=R, 1=G, 2=B  →  mapping: out_c = 2-c
            const int srcC = 2 - c; // BGR→RGB swap

            float val = (1.0f - ax) * (1.0f - ay) * p00[srcC]
                      + ax          * (1.0f - ay) * p01[srcC]
                      + (1.0f - ax) * ay          * p10[srcC]
                      + ax          * ay          * p11[srcC];

            dst[c * planeSize + idx] = val * normScale;
        }
    } else {
        // Padding pixel
        dst[0 * planeSize + idx] = padValue; // R
        dst[1 * planeSize + idx] = padValue; // G
        dst[2 * planeSize + idx] = padValue; // B
    }
}

// Host function to perform CUDA-based preprocessing
void cuda_preprocess(
    uint8_t* src,        // Source image data on host
    int src_width,       // Source image width
    int src_height,      // Source image height
    float* dst,          // Destination buffer on device
    int dst_width,       // Destination image width
    int dst_height,      // Destination image height
    cudaStream_t stream  // CUDA stream for asynchronous execution
) {
    // Calculate the size of the image in bytes (3 channels: BGR)
    int img_size = src_width * src_height * 3;

    // Copy source image data to pinned host memory for faster transfer
    memcpy(img_buffer_host, src, img_size);

    // Asynchronously copy image data from host to device memory
    CUDA_CHECK(cudaMemcpyAsync(
        img_buffer_device,
        img_buffer_host,
        img_size,
        cudaMemcpyHostToDevice,
        stream
    ));

    // Compute resize dimensions (maintain aspect ratio)
    const float scale = fminf(
        static_cast<float>(dst_height) / src_height,
        static_cast<float>(dst_width) / src_width
    );
    const int newW = static_cast<int>(roundf(src_width * scale));
    const int newH = static_cast<int>(roundf(src_height * scale));

    // Ultralytics-compatible asymmetric padding
    const float dw = (dst_width - newW) / 2.0f;
    const float dh = (dst_height - newH) / 2.0f;
    const int padLeft = static_cast<int>(roundf(dw - 0.1f));
    const int padTop  = static_cast<int>(roundf(dh - 0.1f));

    constexpr float normScale = 1.0f / 255.0f;
    constexpr float padValue  = 114.0f / 255.0f;

    // Calculate the total number of pixels to process
    int jobs = dst_height * dst_width;

    // Define the number of threads per block
    int threads = 256;

    // Calculate the number of blocks needed
    int blocks = ceil(jobs / (float)threads);
    
    letterboxNormalizeKernel<<< blocks, threads, 0, stream >>>(
        src, dst,
        src_width, src_height, dst_width, dst_height,
        newW, newH, padLeft, padTop,
        normScale, padValue
    );

    // Optionally, you might want to check for kernel launch errors
    CUDA_CHECK(cudaGetLastError());
}

// Initialize CUDA preprocessing by allocating memory
void cuda_preprocess_init(int max_image_size) {
    // Allocate pinned (page-locked) memory on the host for faster transfers
    CUDA_CHECK(cudaMallocHost((void**)&img_buffer_host, max_image_size * 3));

    // Allocate memory on the device (GPU) for the image
    CUDA_CHECK(cudaMalloc((void**)&img_buffer_device, max_image_size * 3));
}

// Clean up and free allocated memory
void cuda_preprocess_destroy() {
    // Free device memory
    CUDA_CHECK(cudaFree(img_buffer_device));

    // Free pinned host memory
    CUDA_CHECK(cudaFreeHost(img_buffer_host));
}