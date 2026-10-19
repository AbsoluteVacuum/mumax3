#ifndef _DEMAGKERNEL_GPU_H_
#define _DEMAGKERNEL_GPU_H_

// Shared by demagaddends.cu and demagsum.cu, see demagkernel_gpu.go.
// A "job" is one (source component s, destination cell xw,yw,zw) in the non-skipped half space;
// its "images" are all loop coordinates (x,y,z) of mag.CalcDemagKernel that wrap onto that cell,
// numbered k = (iz*ny + iy)*nx + ix: exactly the order in which the CPU code adds them.

#define PI4 (4 * 3.141592653589793)

// closest distance between cells, given center distance d (mag.delta)
__device__ static double kdelta(int d) {
    if (d < 0) d = -d;
    if (d > 0) d -= 1;
    return (double)d;
}

// smallest coordinate >= r1 that wraps onto t (period n)
__device__ static int kfirst(int t, int r1, int n) {
    return r1 + ((t - r1) % n + n) % n;
}

// number of coordinates in [r1, r2] that wrap onto t
__device__ static int kcount(int t, int r1, int r2, int n) {
    int f = kfirst(t, r1, n);
    return (f > r2) ? 0 : (r2 - f) / n + 1;
}

// Field B at offset (x,y,z) cells from the face charges perpendicular to s of a unit cell.
// Same arithmetic, in the same order, as mag.CalcDemagKernel. Products, quotients and square roots
// use the _rn intrinsics: nvcc never fuses them into FMAs nor approximates them, whatever the build
// flags (Makefile, deploy_windows.ps1, -use_fast_math), so results match the CPU bit for bit.
#define MUL(a, b) __dmul_rn((a), (b))
#define DIV(a, b) __ddiv_rn((a), (b))
#define SQRT(a) __dsqrt_rn(a)

__device__ static void kernelB(double* B, int s, int x, int y, int z,
                               double cx, double cy, double cz, double accuracy, double L) {
    double cs[3] = {cx, cy, cz};
    int u = s, v = (s + 1) % 3, w = (s + 2) % 3;
    double Rx = MUL((double)x, cx), Ry = MUL((double)y, cy), Rz = MUL((double)z, cz);

    double dx = MUL(kdelta(x), cx), dy = MUL(kdelta(y), cy), dz = MUL(kdelta(z), cz);
    double d = SQRT(MUL(dx, dx) + MUL(dy, dy) + MUL(dz, dz));
    if (d == 0) {
        d = L;
    }
    double maxSize = DIV(d, accuracy);

    int nv = (int)(fmax(DIV(cs[v], maxSize), 1.0) + 0.5);
    int nw = (int)(fmax(DIV(cs[w], maxSize), 1.0) + 0.5);
    int nx = (int)(fmax(DIV(cx, maxSize), 1.0) + 0.5);
    int ny = (int)(fmax(DIV(cy, maxSize), 1.0) + 0.5);
    int nz = (int)(fmax(DIV(cz, maxSize), 1.0) + 0.5);
    nv *= 2;
    nw *= 2;

    double scale = DIV(1.0, (double)(nv * nw * nx * ny * nz));
    double surface = MUL(cs[v], cs[w]);
    double charge = MUL(surface, scale);
    double pu1 = DIV(cs[u], 2.);
    double pu2 = -pu1;

    B[0] = B[1] = B[2] = 0;
    double p1[3], p2[3]; // positive and negative pole positions
    p1[u] = pu1;
    p2[u] = pu2;

    for (int i = 0; i < nv; i++) {
        p1[v] = p2[v] = -DIV(cs[v], 2.) + DIV(cs[v], (double)(2 * nv)) + MUL((double)i, DIV(cs[v], (double)nv));
        for (int j = 0; j < nw; j++) {
            p1[w] = p2[w] = -DIV(cs[w], 2.) + DIV(cs[w], (double)(2 * nw)) + MUL((double)j, DIV(cs[w], (double)nw));
            for (int a = 0; a < nx; a++) {
                double rx = Rx - DIV(cx, 2.) + DIV(cx, (double)(2 * nx)) + MUL(DIV(cx, (double)nx), (double)a);
                for (int b = 0; b < ny; b++) {
                    double ry = Ry - DIV(cy, 2.) + DIV(cy, (double)(2 * ny)) + MUL(DIV(cy, (double)ny), (double)b);
                    for (int c = 0; c < nz; c++) {
                        double rz = Rz - DIV(cz, 2.) + DIV(cz, (double)(2 * nz)) + MUL(DIV(cz, (double)nz), (double)c);

                        double R2x = rx - p1[0], R2y = ry - p1[1], R2z = rz - p1[2];
                        double r = SQRT(MUL(R2x, R2x) + MUL(R2y, R2y) + MUL(R2z, R2z));
                        double qr = DIV(charge, MUL(MUL(MUL(PI4, r), r), r));
                        double bx = MUL(R2x, qr);
                        double by = MUL(R2y, qr);
                        double bz = MUL(R2z, qr);

                        R2x = rx - p2[0];
                        R2y = ry - p2[1];
                        R2z = rz - p2[2];
                        r = SQRT(MUL(R2x, R2x) + MUL(R2y, R2y) + MUL(R2z, R2z));
                        qr = DIV(-charge, MUL(MUL(MUL(PI4, r), r), r));
                        B[0] += (bx + MUL(R2x, qr));
                        B[1] += (by + MUL(R2y, qr));
                        B[2] += (bz + MUL(R2z, qr));
                    }
                }
            }
        }
    }
}

#endif
