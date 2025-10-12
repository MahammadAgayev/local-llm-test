package tensor

import "testing"


func TestSimpleSoftmax(t *testing.T) {
	vec := []float64{3,2,1}

	r := Softmax(vec)

	t.Log(r)
}

func TestMatrixMultiply(t *testing.T) {
	// Test matrix multiplication with given example
	// Matrix A: 3x3
	a := NewTensor([]float64{
		1, 2, 3,
		1, 2, 3,
		1, 1, 1,
	}, []int{3, 3})

	// Matrix B: 3x5
	b := NewTensor([]float64{
		1, 2, 3, 4, 5,
		1, 2, 3, 4, 5,
		1, 2, 3, 4, 5,
	}, []int{3, 5})

	// Expected result: 3x5
	expected := NewTensor([]float64{
		6, 12, 18, 24, 30,
		6, 12, 18, 24, 30,
		3, 6, 9, 12, 15,
	}, []int{3, 5})

	result, err := MatixMultiply(a, b)
	if err != nil {
		t.Fatalf("MatixMultiply failed: %v", err)
	}

	if result.shape[0] != expected.shape[0] || result.shape[1] != expected.shape[1] {
		t.Fatalf("Shape mismatch: got %v, expected %v", result.shape, expected.shape)
	}

	for i := 0; i < len(result.data); i++ {
		if result.data[i] != expected.data[i] {
			t.Fatalf("Value mismatch at index %d: got %f, expected %f", i, result.data[i], expected.data[i])
		}
	}
}
