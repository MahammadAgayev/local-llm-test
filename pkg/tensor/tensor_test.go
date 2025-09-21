package tensor

import "testing"


func Test3DTensorIndexing(t *testing.T) {
	data := []float64{1, 2, 3, 4, 5, 6, 7, 8}
	shape := []int{2, 2, 2}  // 2x2x2 cube

	tensor := NewTensor(data, shape)

	tests := []struct{
		pos []int
		expected float64
		test string
	} {
		{
			pos: []int{0,0,0},
			expected: 1,
			test: "Edge: First element in tensor",
		},
		{
			pos: []int{1,1,1},
			expected: 8,
			test: "Edge: Last element in tensor",
		},
		{
			pos: []int{1, 1, 0},
			expected: 7,
			test: "Middle element in tensor",
		},
	}

	for _, tt := range tests {
		t.Run(tt.test, func(t *testing.T) {
			got := tensor.Value(tt.pos)

			if got != tt.expected {
				t.Errorf("Expected %f, got %f", tt.expected, got)
			}
		})
	}
}

