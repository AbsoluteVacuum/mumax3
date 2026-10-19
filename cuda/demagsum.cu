#include "demagkernel_gpu.h"

// Phase 2 of the GPU demag kernel (see demagkernel_gpu.go): one thread per job.
// Adds the job's addends k0 .. k0+K-1 from buf to its running float64 sums in out,
// sequentially in image order: the summation order of mag.CalcDemagKernel (which sums in float32).
// out: 6 slots (XX XY XZ YY YZ ZZ) of hz*hy*hx doubles, zeroed before the first chunk.
extern "C" __global__ void
demagsum(double* __restrict__ out, double* __restrict__ buf, int k0, int K, int nJobs,
         int Nx, int Ny, int Nz,
         int r1x, int r1y, int r1z, int r2x, int r2y, int r2z) {

    int job = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;
    if (job >= nJobs) {
        return;
    }
    int hx = Nx / 2 + 1, hy = Ny / 2 + 1, hz = Nz / 2 + 1;
    int xw = job % hx;
    int yw = (job / hx) % hy;
    int zw = (job / (hx * hy)) % hz;
    int s  = job / (hx * hy * hz);

    int n = kcount(xw, r1x, r2x, Nx) * kcount(yw, r1y, r2y, Ny) * kcount(zw, r1z, r2z, Nz) - k0;
    if (n > K) {
        n = K;
    }
    int base = (s == 0) ? 0 : ((s == 1) ? 2 : 3);
    int ncell = hx * hy * hz;
    int cell = (zw * hy + yw) * hx + xw;

    for (int d = s; d < 3; d++) {
        double acc = out[(base + d) * ncell + cell];
        for (int kk = 0; kk < n; kk++) {
            acc += buf[(d * K + kk) * nJobs + job];
        }
        out[(base + d) * ncell + cell] = acc;
    }
}
