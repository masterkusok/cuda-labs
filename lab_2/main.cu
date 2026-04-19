#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define CSC(call) \
do { \
    cudaError_t res = call; \
    if (res != cudaSuccess) { \
        fprintf(stderr, "ERROR in %s:%d. Message: %s\n", \
                __FILE__, __LINE__, cudaGetErrorString(res)); \
        exit(0); \
    } \
} while(0)

__device__ float brightness(uchar4 p) {
    return 0.299f * p.x + 0.587f * p.y + 0.114f * p.z;
}

__global__ void kernel(cudaTextureObject_t tex, uchar4 *out, int w, int h) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int idy = blockDim.y * blockIdx.y + threadIdx.y;
    int offsetx = blockDim.x * gridDim.x;
    int offsety = blockDim.y * gridDim.y;

    for (int y = idy; y < h; y += offsety) {
        for (int x = idx; x < w; x += offsetx) {
            int x0 = max(x - 1, 0);
            int x2 = min(x + 1, w - 1);
            int y0 = max(y - 1, 0);
            int y2 = min(y + 1, h - 1);

            float p00 = brightness(tex2D<uchar4>(tex, x0, y0));
            float p10 = brightness(tex2D<uchar4>(tex, x, y0));
            float p20 = brightness(tex2D<uchar4>(tex, x2, y0));
            float p01 = brightness(tex2D<uchar4>(tex, x0, y));
            float p21 = brightness(tex2D<uchar4>(tex, x2, y));
            float p02 = brightness(tex2D<uchar4>(tex, x0, y2));
            float p12 = brightness(tex2D<uchar4>(tex, x, y2));
            float p22 = brightness(tex2D<uchar4>(tex, x2, y2));

            float Gx = -p00 + p20 - 2.0f * p01 + 2.0f * p21 - p02 + p22;
            float Gy = p00 + 2.0f * p10 + p20 - p02 - 2.0f * p12 - p22;

            float G = sqrtf(Gx * Gx + Gy * Gy);
            unsigned char val = (unsigned char)fminf(G, 255.0f);

            out[y * w + x] = make_uchar4(val, val, val, 255);
        }
    }
}

int main() {
    char input[256], output[256];
    scanf("%s", input);
    scanf("%s", output);

    int w, h;

    FILE *fp = fopen(input, "rb");
    if (!fp) {
        fprintf(stderr, "Cannot open input file\n");
        return 1;
    }
    fread(&w, sizeof(int), 1, fp);
    fread(&h, sizeof(int), 1, fp);

    uchar4 *data = (uchar4*)malloc(sizeof(uchar4) * w * h);
    fread(data, sizeof(uchar4), w * h, fp);
    fclose(fp);

    cudaChannelFormatDesc ch = cudaCreateChannelDesc<uchar4>();

    cudaArray *arr;
    CSC(cudaMallocArray(&arr, &ch, w, h));
    CSC(cudaMemcpy2DToArray(arr, 0, 0, data,
                           w * sizeof(uchar4),
                           w * sizeof(uchar4),
                           h,
                           cudaMemcpyHostToDevice));

    struct cudaResourceDesc resDesc;
    memset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = arr;

    struct cudaTextureDesc texDesc;
    memset(&texDesc, 0, sizeof(texDesc));
    texDesc.addressMode[0] = cudaAddressModeClamp;
    texDesc.addressMode[1] = cudaAddressModeClamp;
    texDesc.filterMode = cudaFilterModePoint;
    texDesc.readMode = cudaReadModeElementType;
    texDesc.normalizedCoords = false;

    cudaTextureObject_t tex = 0;
    CSC(cudaCreateTextureObject(&tex, &resDesc, &texDesc, NULL));

    uchar4 *dev_out;
    CSC(cudaMalloc(&dev_out, sizeof(uchar4) * w * h));

    kernel<<<dim3(16, 16), dim3(32, 32)>>>(tex, dev_out, w, h);
    CSC(cudaGetLastError());
    CSC(cudaDeviceSynchronize());

    CSC(cudaMemcpy(data, dev_out,
                   sizeof(uchar4) * w * h,
                   cudaMemcpyDeviceToHost));

    CSC(cudaDestroyTextureObject(tex));
    CSC(cudaFreeArray(arr));
    CSC(cudaFree(dev_out));

    fp = fopen(output, "wb");
    fwrite(&w, sizeof(int), 1, fp);
    fwrite(&h, sizeof(int), 1, fp);
    fwrite(data, sizeof(uchar4), w * h, fp);
    fclose(fp);

    free(data);
    return 0;
}