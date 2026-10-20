package mag

import (
	"math"
	"os"
	"testing"
	"time"

	"github.com/mumax/3/data"
)

// maxAbsKernel returns max |N| over all components.
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

// compareKernels fails if any entry of got differs from ref by more than tol * max|ref| (over all
// components, the scale that matters for the field) and logs per-component differences.
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

// checkOddComponentsVanish: with Ny=1 and symmetric PBC in y, Nxy and Nyz are odd in y and must
// sum to zero. A float64 image sum gets this to rounding level; CalcDemagKernel's float32 sum
// leaves ~1e-9 of max|N|.
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

// Tolerance vs CalcDemagKernel: its float32 image sums are off by a few 1e-6 of max|N|, growing with
// the number of images per cell; the fast kernel sums in float64. Logic errors show up as >=1e-3.
const oldKernelTol = 1e-4

func TestFastKernelEquivalence(t *testing.T) {
	cases := []struct {
		name      string
		size, pbc [3]int
		cell      [3]float64
	}{
		{"thin film PBCy (scaled user case)", [3]int{128, 1, 8}, [3]int{3, 200, 0}, [3]float64{5e-9, 5e-9, 2e-9}},
		{"2D film PBCy", [3]int{64, 1, 1}, [3]int{0, 300, 0}, [3]float64{4e-9, 4e-9, 1e-9}},
		{"small film, huge PBCy", [3]int{8, 1, 2}, [3]int{2, 20000, 0}, [3]float64{5e-9, 5e-9, 2e-9}},
		{"3D PBC all", [3]int{8, 6, 5}, [3]int{2, 3, 1}, [3]float64{3e-9, 4e-9, 2e-9}},
		{"no PBC", [3]int{10, 7, 3}, [3]int{0, 0, 0}, [3]float64{2e-9, 3e-9, 5e-9}},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			t0 := time.Now()
			old := CalcDemagKernel(c.size, c.pbc, c.cell, 6)
			tOld := time.Since(t0)
			t0 = time.Now()
			fast, ok := CalcDemagKernelFast(c.size, c.pbc, c.cell, 6, 1<<30)
			tFast := time.Since(t0)
			if !ok {
				t.Fatal("fast kernel refused to run")
			}
			t.Logf("old %v, fast %v (%.1fx)", tOld, tFast, tOld.Seconds()/tFast.Seconds())
			compareKernels(t, "fast vs old", old, fast, oldKernelTol)
			if c.size[Y] == 1 && c.pbc[Y] > 0 {
				checkOddComponentsVanish(t, "fast", fast)
			}
		})
	}
}

func TestFastKernelFallback(t *testing.T) {
	if _, ok := CalcDemagKernelFast([3]int{16, 1, 2}, [3]int{1, 10, 0}, [3]float64{1e-9, 1e-9, 1e-9}, 6, 0); ok {
		t.Fatal("expected fallback (ok=false) with zero memory budget")
	}
}

// Full user case (16384x1x32, PBC 3,1000,0), fast method only — the old one takes hours.
// Run with: MUMAX_BIG_KERNEL=1 go test -run BigKernel -v -timeout 0 ./mag
func TestBigKernel(t *testing.T) {
	if os.Getenv("MUMAX_BIG_KERNEL") == "" {
		t.Skip("set MUMAX_BIG_KERNEL=1")
	}
	t0 := time.Now()
	_, ok := CalcDemagKernelFast([3]int{16384, 1, 32}, [3]int{3, 1000, 0}, [3]float64{5e-9, 5e-9, 2e-9}, 6, availableMemory()/2)
	t.Logf("ok=%v, %v", ok, time.Since(t0))
}

func TestKernelMethod(t *testing.T) {
	for in, want := range map[string]string{"off": "old", "OLD": "old", "": "old", "CPU": "cpu", "gpu": "gpu", " Auto ": "auto"} {
		if got, err := kernelMethod(in); err != nil || got != want {
			t.Errorf("kernelMethod(%q) = %q, %v; want %q", in, got, err, want)
		}
	}
	for _, bad := range []string{"true", "fast", "cuda"} {
		if _, err := kernelMethod(bad); err == nil {
			t.Errorf("kernelMethod(%q): expected error", bad)
		}
	}
}

func TestAutoUseGPU(t *testing.T) {
	defer func(g func([3]int, [3]int, [3]int, [3]float64, float64, float64) ([]float32, error), r func() int) {
		DemagKernelGPU, DemagKernelGPUFP64Ratio = g, r
	}(DemagKernelGPU, DemagKernelGPUFP64Ratio)

	DemagKernelGPU, DemagKernelGPUFP64Ratio = nil, nil
	if use, _ := autoUseGPU(1e12); use {
		t.Error("auto chose GPU without a GPU implementation")
	}
	DemagKernelGPU = func([3]int, [3]int, [3]int, [3]float64, float64, float64) ([]float32, error) { return nil, nil }
	for _, c := range []struct {
		ratio  int
		images float64
		want   bool
	}{{64, 1e12, false}, {32, 1e12, false}, {0, 1e12, false}, {2, 1e6, false}, {2, 1e9, true}} {
		ratio := c.ratio
		DemagKernelGPUFP64Ratio = func() int { return ratio }
		if use, why := autoUseGPU(c.images); use != c.want {
			t.Errorf("ratio %d, %g images: GPU=%v (%s), want %v", c.ratio, c.images, use, why, c.want)
		}
	}
}

// imageEvaluations must count exactly the (s, cell, image) evaluations of CalcDemagKernel's loops.
func TestImageEvaluations(t *testing.T) {
	for _, c := range []struct{ size, pbc [3]int }{
		{[3]int{16, 1, 4}, [3]int{3, 100, 0}}, {[3]int{8, 6, 5}, [3]int{2, 3, 1}}, {[3]int{10, 7, 3}, [3]int{0, 0, 0}}, {[3]int{9, 1, 1}, [3]int{4, 50, 0}},
	} {
		size := padSize(c.size, c.pbc)
		r1, r2 := kernelRanges(size, c.pbc)
		n := 0
		for z := r1[Z]; z <= r2[Z]; z++ {
			for y := r1[Y]; y <= r2[Y]; y++ {
				for x := r1[X]; x <= r2[X]; x++ {
					if wrap(x, size[X]) <= size[X]/2 && wrap(y, size[Y]) <= size[Y]/2 && wrap(z, size[Z]) <= size[Z]/2 {
						n += 3
					}
				}
			}
		}
		if got := imageEvaluations(c.size, c.pbc); got != float64(n) {
			t.Errorf("%v pbc %v: imageEvaluations = %g, loops = %d", c.size, c.pbc, got, n)
		}
	}
}
