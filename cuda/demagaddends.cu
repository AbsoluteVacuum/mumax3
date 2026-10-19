#include "demagkernel_gpu.h"

// Phase 1 of the GPU demag kernel (see demagkernel_gpu.go): one thread per (image k, job).
// Images k0 .. k0+K-1 of every job; writes B[d] (float64), d >= s, to
// buf[(d*K + kk)*nJobs + job], kk = k-k0. Images a job does not have are left untouched.
extern "C" __global__ void
demagaddends(double* __restrict__ buf, int k0, int K, int nJobs,
             int Nx, int Ny, int Nz,
             int r1x, int r1y, int r1z, int r2x, int r2y, int r2z,
             double cx, double cy, double cz, double accuracy, double L) {

    int g = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;
    if (g >= K * nJobs) {
        return;
    }
    int job = g % nJobs;
    int kk = g / nJobs;
    int k = k0 + kk;

    int hx = Nx / 2 + 1, hy = Ny / 2 + 1, hz = Nz / 2 + 1;
    int xw = job % hx;
    int yw = (job / hx) % hy;
    int zw = (job / (hx * hy)) % hz;
    int s  = job / (hx * hy * hz);

    int nx = kcount(xw, r1x, r2x, Nx), ny = kcount(yw, r1y, r2y, Ny), nz = kcount(zw, r1z, r2z, Nz);
    if (k >= nx * ny * nz) {
        return;
    }
    int x = kfirst(xw, r1x, Nx) + (k % nx) * Nx;
    int y = kfirst(yw, r1y, Ny) + ((k / nx) % ny) * Ny;
    int z = kfirst(zw, r1z, Nz) + (k / (nx * ny)) * Nz;

    double B[3];
    kernelB(B, s, x, y, z, cx, cy, cz, accuracy, L);
    for (int d = s; d < 3; d++) {
        buf[(d * K + kk) * nJobs + job] = B[d];
    }
}
