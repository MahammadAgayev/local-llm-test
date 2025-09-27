package tensor

import (
	"errors"
	"math"
)


func DotProduct(a []float64, b []float64) (float64, error) {
	if len(a) != len(b) {
		return 0, errors.New("DotProduct function matrixes should have equal length")
	}

	result := float64(0)
	for i := range a {
		result+= a[i] * b[i]
	}

	return result, nil
}

func Softmax(scores []float64) []float64 {
	result := make([]float64, len(scores))
	sum := float64(0)
	for i := range scores {
		result[i] = math.Exp(scores[i])
		sum+=result[i]
	}

	for i := range result {
		result[i] = result[i] / sum
	}

	return result
}
