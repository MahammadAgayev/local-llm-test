package tensor

import "testing"


func TestSimpleSoftmax(t *testing.T) {
	vec := []float64{3,2,1}

	r := Softmax(vec)

	t.Log(r)
}
