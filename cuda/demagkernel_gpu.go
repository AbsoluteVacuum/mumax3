package cuda

import (
	"fmt"
	"math"
	"unsafe"

	"github.com/mumax/3/cuda/cu"
	"github.com/mumax/3/mag"
	"github.com/mumax/3/util"
)

// Plug the GPU demag kernel into package mag (used when FasterPBCKernel = true).
func init() { mag.DemagKernelGPU = demagKernelHalfGPU }

// demagKernelHalfGPU computes the non-skipped half space of the demag kernel.
// Returns 6 slots (XX XY XZ YY YZ ZZ) of hz*hy*hx floats; any CUDA failure becomes an error.
//
// Same images, arithmetic and order as mag.CalcDemagKernel, but summed in float64 and rounded to
// float32 once (like mag.CalcDemagKernelFast). Parallel over PBC images, not only over cells,
// so small grids with huge PBC still fill the GPU. Per chunk of K images:
//
//	demagaddends: one thread per (image, cell) computes B (float64) into a buffer (the expensive part)
//	demagsum:     one thread per cell adds its K addends in image order onto its running float64 sum
//
// Sequential sums per cell in the CPU order, carried across chunks: bit-identical to the same
// float64 sum on the CPU.
func demagKernelHalfGPU(size, r1, r2 [3]int, cellsize [3]float64, accuracy, L float64) (out []float32, err error) {
	defer func() {
		if e := recover(); e != nil {
			err = fmt.Errorf("GPU demag kernel: %v", e)
		}
	}()

	hx, hy, hz := size[X]/2+1, size[Y]/2+1, size[Z]/2+1
	nJobs := 3 * hx * hy * hz
	n := 6 * hx * hy * hz

	// M: most images any cell has (counts per direction are independent, so take each max)
	M := 1
	for c, h := range [3]int{hx, hy, hz} {
		maxc := 0
		for t := 0; t < h; t++ {
			maxc = max(maxc, imageCount(t, r1[c], r2[c], size[c]))
		}
		M *= maxc
	}

	// K: images per chunk. Buffer = 3 doubles per (image, job); half the free GPU memory,
	// int32 indexing in the kernels (3*K*nJobs < 2^31), and at most 2^26 threads per launch so a
	// launch stays well below the ~2 s Windows display-driver watchdog (TDR), even on GPUs with slow FP64.
	free, _ := cu.MemGetInfo()
	K := min(M, int((free/2-int64(n)*8)/(24*int64(nJobs))), (math.MaxInt32-1)/(3*nJobs), max(1, (1<<26)/nJobs))
	if K < 1 {
		return nil, fmt.Errorf("not enough GPU memory (%d bytes free)", free)
	}
	util.Log(fmt.Sprintf("//FasterPBCKernel: GPU, %d cells x %d images, %d images per chunk", nJobs, M, K))

	dOut := cu.MemAlloc(int64(n) * 8)
	defer cu.MemFree(dOut)
	cu.MemsetD32(dOut, 0, 2*int64(n)) // all-zero bits = 0.0
	dBuf := cu.MemAlloc(24 * int64(K) * int64(nJobs))
	defer cu.MemFree(dBuf)
	pOut, pBuf := unsafe.Pointer(uintptr(dOut)), unsafe.Pointer(uintptr(dBuf))

	for k0 := 0; k0 < M; k0 += K {
		k_demagaddends_async(pBuf, k0, K, nJobs, size[X], size[Y], size[Z],
			r1[X], r1[Y], r1[Z], r2[X], r2[Y], r2[Z],
			cellsize[X], cellsize[Y], cellsize[Z], accuracy, L, demagConf(K*nJobs))
		k_demagsum_async(pOut, pBuf, k0, K, nJobs, size[X], size[Y], size[Z],
			r1[X], r1[Y], r1[Z], r2[X], r2[Y], r2[Z], demagConf(nJobs))
		Sync()
		util.Progress(min(k0+K, M), M, "Calculating demag kernel (GPU)")
	}

	sum := make([]float64, n)
	cu.MemcpyDtoH(unsafe.Pointer(&sum[0]), dOut, int64(n)*8)
	out = make([]float32, n)
	for i, v := range sum {
		out[i] = float32(v)
	}
	return out, nil
}

// number of coordinates in [r1, r2] that wrap onto t (period n); same as kcount in demagkernel_gpu.h
func imageCount(t, r1, r2, n int) int {
	f := r1 + ((t-r1)%n+n)%n
	if f > r2 {
		return 0
	}
	return (r2-f)/n + 1
}

// 1D launch configuration with 128-thread blocks: demagaddends uses ~140 registers/thread
// (ptxas, sm_80/90), so the default 512-thread blocks would exceed 64k registers/block.
func demagConf(N int) *config {
	const block = 128
	nb := divUp(N, block)
	gx := divUp(nb, MaxGridSize)
	return &config{Grid: cu.Dim3{X: gx, Y: divUp(nb, gx), Z: 1}, Block: cu.Dim3{X: block, Y: 1, Z: 1}}
}
