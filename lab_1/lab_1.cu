#include <iostream>
#include <stdio.h>
#include <stdlib.h>

__global__ void min_kernel(double *a, double* b, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n && b[idx] < a[idx]) {
        a[idx] = b[idx];
    }
}

int main() {
    int n;
    std::cin >> n;
    
    double *a = (double *)malloc(sizeof(double) * n);
    for(int i = 0; i < n; i++)
        std::cin >> a[i];

    double *b = (double *)malloc(sizeof(double) * n);
    for(int i = 0; i < n; i++)
        std::cin >> b[i];

    double *gpu_a;
    cudaMalloc(&gpu_a, sizeof(double) * n);
    cudaMemcpy(gpu_a, a, sizeof(double) * n, cudaMemcpyHostToDevice);

    double *gpu_b;
    cudaMalloc(&gpu_b, sizeof(double) * n);
    cudaMemcpy(gpu_b, b, sizeof(double) * n, cudaMemcpyHostToDevice);
    
    min_kernel<<<32768, 1024>>>(gpu_a, gpu_b, n);
    cudaMemcpy(a, gpu_a, sizeof(double) * n, cudaMemcpyDeviceToHost);

    for(int i = 0; i < n; i++) 
        printf("%.10lf ", a[i]);
    
    cudaFree(gpu_b);
    cudaFree(gpu_a);

    free(a);
    free(b);

    return 0;
}