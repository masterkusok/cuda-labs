#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define MAX_CLASSES 32

#define CSC(call)                                                              \
  do {                                                                         \
    cudaError_t res = call;                                                    \
    if (res != cudaSuccess) {                                                  \
      fprintf(stderr, "ERROR in %s:%d. Message: %s\n", __FILE__, __LINE__,     \
              cudaGetErrorString(res));                                        \
      exit(0);                                                                 \
    }                                                                          \
  } while (0)

__device__ float d_class_centroids[MAX_CLASSES][3];

__global__ void kernel(uchar4 *img, int w, int h, int num_classes) {
  int idx = blockDim.x * blockIdx.x + threadIdx.x;
  int idy = blockDim.y * blockIdx.y + threadIdx.y;
  int offsetx = blockDim.x * gridDim.x;
  int offsety = blockDim.y * gridDim.y;

  for (int y = idy; y < h; y += offsety) {
    for (int x = idx; x < w; x += offsetx) {
      long long i = (long long)y * w + x;
      uchar4 pixel = img[i];
      float r = (float)pixel.x;
      float g = (float)pixel.y;
      float b = (float)pixel.z;

      int best_class = 0;
      float best_score = -1e30f;

      for (int c = 0; c < num_classes; c++) {
        float cr = d_class_centroids[c][0];
        float cg = d_class_centroids[c][1];
        float cb = d_class_centroids[c][2];
        float norm = sqrtf(cr * cr + cg * cg + cb * cb);

        float score =
            (norm > 1e-10f) ? (r * cr + g * cg + b * cb) / norm : -1e30f;

        if (score > best_score) {
          best_score = score;
          best_class = c;
        }
      }
      img[i].w = (unsigned char)best_class;
    }
  }
}

int main() {
  char input_path[256], output_path[256];
  scanf("%s %s", input_path, output_path);

  int w, h;
  FILE *fp = fopen(input_path, "rb");
  if (!fp) {
    fprintf(stderr, "Cannot open input file\n");
    return 1;
  }
  fread(&w, sizeof(int), 1, fp);
  fread(&h, sizeof(int), 1, fp);

  size_t total_pixels = (size_t)w * h;
  uchar4 *data = (uchar4 *)malloc(sizeof(uchar4) * total_pixels);
  fread(data, sizeof(uchar4), total_pixels, fp);
  fclose(fp);

  int num_classes;
  scanf("%d", &num_classes);

  float host_centroids[MAX_CLASSES][3] = {0};

  for (int c = 0; c < num_classes; c++) {
    int num_points;
    scanf("%d", &num_points);
    double sum_r = 0.0, sum_g = 0.0, sum_b = 0.0;

    for (int i = 0; i < num_points; i++) {
      int px, py;
      scanf("%d %d", &px, &py);
      if (px >= 0 && px < w && py >= 0 && py < h) {
        uchar4 p = data[py * w + px];
        sum_r += p.x;
        sum_g += p.y;
        sum_b += p.z;
      }
    }
    if (num_points > 0) {
      host_centroids[c][0] = (float)(sum_r / num_points);
      host_centroids[c][1] = (float)(sum_g / num_points);
      host_centroids[c][2] = (float)(sum_b / num_points);
    }
  }

  CSC(cudaMemcpyToSymbol(d_class_centroids, host_centroids,
                         sizeof(float) * MAX_CLASSES * 3));

  uchar4 *dev_data;
  CSC(cudaMalloc(&dev_data, sizeof(uchar4) * total_pixels));
  CSC(cudaMemcpy(dev_data, data, sizeof(uchar4) * total_pixels,
                 cudaMemcpyHostToDevice));

  dim3 block_size(32, 32);
  dim3 grid_size(16, 16);
  kernel<<<grid_size, block_size>>>(dev_data, w, h, num_classes);

  CSC(cudaGetLastError());
  CSC(cudaDeviceSynchronize());

  CSC(cudaMemcpy(data, dev_data, sizeof(uchar4) * total_pixels,
                 cudaMemcpyDeviceToHost));

  fp = fopen(output_path, "wb");
  fwrite(&w, sizeof(int), 1, fp);
  fwrite(&h, sizeof(int), 1, fp);
  fwrite(data, sizeof(uchar4), total_pixels, fp);
  fclose(fp);

  CSC(cudaFree(dev_data));
  free(data);

  return 0;
}