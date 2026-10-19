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
// Same arithmetic as mag.CalcDemagKernel; compile with -fmad=false (see Makefile).
__device__ static void kernelB(double* B, int s, int x, int y, int z,
                               double cx, double cy, double cz, double accuracy, double L) {
    double cs[3] = {cx, cy, cz};
    int u = s, v = (s + 1) % 3, w = (s + 2) % 3;
    double Rx = (double)x * cx, Ry = (double)y * cy, Rz = (double)z * cz;

    double dx = kdelta(x) * cx, dy = kdelta(y) * cy, dz = kdelta(z) * cz;
    double d = sqrt(dx * dx + dy * dy + dz * dz);
    if (d == 0) {
        d = L;
    }
    double maxSize = d / accuracy;

    int nv = (int)(fmax(cs[v] / maxSize, 1.0) + 0.5);
    int nw = (int)(fmax(cs[w] / maxSize, 1.0) + 0.5);
    int nx = (int)(fmax(cx / maxSize, 1.0) + 0.5);
    int ny = (int)(fmax(cy / maxSize, 1.0) + 0.5);
    int nz = (int)(fmax(cz / maxSize, 1.0) + 0.5);
    nv *= 2;
    nw *= 2;

    double scale = 1 / (double)(nv * nw * nx * ny * nz);
    double surface = cs[v] * cs[w];
    double charge = surface * scale;
    double pu1 = cs[u] / 2.;
    double pu2 = -pu1;

    B[0] = B[1] = B[2] = 0;
    double p1[3], p2[3]; // positive and negative pole positions
    p1[u] = pu1;
    p2[u] = pu2;

    for (int i = 0; i < nv; i++) {
        p1[v] = p2[v] = -(cs[v] / 2.) + cs[v] / (double)(2 * nv) + (double)i * (cs[v] / (double)nv);
        for (int j = 0; j < nw; j++) {
            p1[w] = p2[w] = -(cs[w] / 2.) + cs[w] / (double)(2 * nw) + (double)j * (cs[w] / (double)nw);
            for (int a = 0; a < nx; a++) {
                double rx = Rx - cx / 2 + cx / (double)(2 * nx) + (cx / (double)nx) * (double)a;
                for (int b = 0; b < ny; b++) {
                    double ry = Ry - cy / 2 + cy / (double)(2 * ny) + (cy / (double)ny) * (double)b;
                    for (int c = 0; c < nz; c++) {
                        double rz = Rz - cz / 2 + cz / (double)(2 * nz) + (cz / (double)nz) * (double)c;

                        double R2x = rx - p1[0], R2y = ry - p1[1], R2z = rz - p1[2];
                        double r = sqrt(R2x * R2x + R2y * R2y + R2z * R2z);
                        double qr = charge / (PI4 * r * r * r);
                        double bx = R2x * qr;
                        double by = R2y * qr;
                        double bz = R2z * qr;

                        R2x = rx - p2[0];
                        R2y = ry - p2[1];
                        R2z = rz - p2[2];
                        r = sqrt(R2x * R2x + R2y * R2y + R2z * R2z);
                        qr = -charge / (PI4 * r * r * r);
                        B[0] += (bx + R2x * qr);
                        B[1] += (by + R2y * qr);
                        B[2] += (bz + R2z * qr);
                    }
                }
            }
        }
    }
}

#endif
