package cuda

import (
	"math"
	"os"
	"runtime"
	"sync"
	"testing"
	"time"

	"github.com/mumax/3/cuda/cu"
	"github.com/mumax/3/data"
	"github.com/mumax/3/mag"
)

// Tolerances, relative to max|N| over all components:
//   - vs mag.CalcDemagKernel: it sums PBC images in float32 (off by a few 1e-6, growing with the
//     number of images); the GPU sums in float64. Logic errors show up as >=1e-3.
//   - vs mag.CalcDemagKernelFast: same float64 sums in the same order. Bit-identical without the
//     CPU's y-mirror trick (Ny>1), float64 rounding of mirrored terms with it (Ny=1).
const (
	oldKernelTol  = 1e-4
	fastKernelTol = 1e-10
)

// pinGPU locks the test goroutine to its OS thread and makes a CUDA context current there.
// A CUDA context is current per OS thread, and Go moves goroutines between threads (the CPU
// reference kernels in these tests start many goroutines), so without this a later GPU call can
// fail with CUDA_ERROR_INVALID_CONTEXT. mumax3 itself is not affected: cuda.Init locks the main
// thread, which then does all CUDA work.
var (
	testCtx     cu.Context
	testCtxOnce sync.Once
)

func pinGPU(t *testing.T) {
	runtime.LockOSThread()
	t.Cleanup(runtime.UnlockOSThread)
	testCtxOnce.Do(func() { testCtx = cu.CtxCreate(cu.CTX_SCHED_AUTO, 0) })
	cu.CtxSetCurrent(testCtx)
}

func maxAbsKernel(k [3][3]*data.Slice) float64 {
	m := 0.0
	for i := 0; i < 3; i++ {
		for j := i; j < 3; j++ {
			if k[i][j] != nil {
				for _, v := range k[i][j].Host()[0] {
					m = math.Max(m, math.Abs(float64(v)))
				}
			}
		}
	}
	return m
}

// compareKernels fails if any entry differs by more than tol * max|ref| and logs per-component diffs.
func compareKernels(t *testing.T, name string, ref, got [3][3]*data.Slice, tol float64) {
	t.Helper()
	scale := maxAbsKernel(ref)
	for i := 0; i < 3; i++ {
		for j := i; j < 3; j++ {
			if (ref[i][j] == nil) != (got[i][j] == nil) {
				t.Fatalf("%s N%d%d: nil mismatch", name, i, j)
			}
			if ref[i][j] == nil {
				continue
			}
			a, b := ref[i][j].Host()[0], got[i][j].Host()[0]
			var maxD float64
			same := 0
			for k := range a {
				maxD = math.Max(maxD, math.Abs(float64(a[k]-b[k])))
				if a[k] == b[k] {
					same++
				}
			}
			t.Logf("%s N%d%d: max|diff| = %.1e of max|N|, bit-identical %d/%d", name, i, j, maxD/scale, same, len(a))
			if maxD > tol*scale {
				t.Errorf("%s N%d%d: max|diff| %.2e > %.0e of max|N|", name, i, j, maxD/scale, tol)
			}
		}
	}
}

// With Ny=1 and PBC in y, Nxy and Nyz are odd in y and must vanish; float64 sums get them to
// rounding level, the float32 sums of mag.CalcDemagKernel leave ~1e-9 of max|N|.
func checkOddComponentsVanish(t *testing.T, name string, k [3][3]*data.Slice) {
	t.Helper()
	scale := maxAbsKernel(k)
	for _, c := range [][2]int{{X, Y}, {Y, Z}} {
		if k[c[0]][c[1]] == nil {
			continue
		}
		m := 0.0
		for _, v := range k[c[0]][c[1]].Host()[0] {
			m = math.Max(m, math.Abs(float64(v)))
		}
		t.Logf("%s N%d%d (should be 0): max = %.1e of max|N|", name, c[0], c[1], m/scale)
		if m > 1e-10*scale {
			t.Errorf("%s N%d%d should vanish by symmetry: %.1e of max|N|", name, c[0], c[1], m/scale)
		}
	}
}

func TestDemagKernelGPU(t *testing.T) {
	pinGPU(t)
	cases := []struct {
		name      string
		size, pbc [3]int
		cell      [3]float64
	}{
		{"thin film PBCy", [3]int{128, 1, 8}, [3]int{3, 200, 0}, [3]float64{5e-9, 5e-9, 2e-9}},
		{"2D film PBCy", [3]int{64, 1, 1}, [3]int{0, 300, 0}, [3]float64{4e-9, 4e-9, 1e-9}},
		{"3D PBC all", [3]int{8, 6, 5}, [3]int{2, 3, 1}, [3]float64{3e-9, 4e-9, 2e-9}},
		{"no PBC", [3]int{10, 7, 3}, [3]int{0, 0, 0}, [3]float64{2e-9, 3e-9, 5e-9}},
		{"odd sizes", [3]int{7, 5, 3}, [3]int{1, 2, 3}, [3]float64{1e-9, 2.5e-9, 1.5e-9}},
		{"small grid, huge PBCy", [3]int{16, 1, 1}, [3]int{0, 50000, 0}, [3]float64{4e-9, 4e-9, 1e-9}},
		{"small film, huge PBCy", [3]int{8, 1, 2}, [3]int{2, 20000, 0}, [3]float64{5e-9, 5e-9, 2e-9}},
	}
	for _, c := range cases {
		t0 := time.Now()
		old := mag.CalcDemagKernel(c.size, c.pbc, c.cell, 6)
		tOld := time.Since(t0)
		t0 = time.Now()
		fast, ok := mag.CalcDemagKernelFast(c.size, c.pbc, c.cell, 6, 1<<32)
		tFast := time.Since(t0)
		if !ok {
			t.Fatal("CPU fast kernel refused to run")
		}
		t0 = time.Now()
		gpu, err := mag.CalcDemagKernelGPU(c.size, c.pbc, c.cell, 6)
		tGPU := time.Since(t0)
		if err != nil {
			t.Fatal(err)
		}
		t.Logf("%s: original %v, CPU fast %v, GPU %v", c.name, tOld, tFast, tGPU)
		compareKernels(t, c.name+" GPU vs original", old, gpu, oldKernelTol)
		compareKernels(t, c.name+" GPU vs CPU fast", fast, gpu, fastKernelTol)
		if c.size[Y] == 1 && c.pbc[Y] > 0 {
			checkOddComponentsVanish(t, c.name+" GPU", gpu)
		}
	}
}

// Full test case 16384x1x32, PBC (3,1000,0), and a small grid with PBCy = 1e6, against the
// multithreaded CPU kernel (the original takes too long); a 1024-wide case also against the original.
// Run with: MUMAX_BIG_KERNEL=1 go test -run DemagKernelGPUBig -v -timeout 0 ./cuda
func TestDemagKernelGPUBig(t *testing.T) {
	if os.Getenv("MUMAX_BIG_KERNEL") == "" {
		t.Skip("set MUMAX_BIG_KERNEL=1")
	}
	pinGPU(t)
	cell := [3]float64{5e-9, 5e-9, 2e-9}
	cases := []struct {
		name      string
		size, pbc [3]int
		original  bool
	}{
		{"1024x1x32 PBC(3,1000,0)", [3]int{1024, 1, 32}, [3]int{3, 1000, 0}, true},
		{"16384x1x32 PBC(3,1000,0)", [3]int{16384, 1, 32}, [3]int{3, 1000, 0}, false},
		{"64x1x4 PBC(3,1e6,0)", [3]int{64, 1, 4}, [3]int{3, 1000000, 0}, false},
	}
	for _, c := range cases {
		t0 := time.Now()
		gpu, err := mag.CalcDemagKernelGPU(c.size, c.pbc, cell, 6)
		if err != nil {
			t.Fatal(err)
		}
		t.Logf("%s GPU: %v", c.name, time.Since(t0))
		checkOddComponentsVanish(t, c.name+" GPU", gpu)

		t0 = time.Now()
		fast, ok := mag.CalcDemagKernelFast(c.size, c.pbc, cell, 6, 1<<32)
		if !ok {
			t.Fatal("CPU fast kernel refused to run")
		}
		t.Logf("%s CPU fast: %v", c.name, time.Since(t0))
		compareKernels(t, c.name+" GPU vs CPU fast", fast, gpu, fastKernelTol)

		if c.original {
			t0 = time.Now()
			old := mag.CalcDemagKernel(c.size, c.pbc, cell, 6)
			t.Logf("%s original: %v", c.name, time.Since(t0))
			compareKernels(t, c.name+" GPU vs original", old, gpu, oldKernelTol)
		}
	}
}
