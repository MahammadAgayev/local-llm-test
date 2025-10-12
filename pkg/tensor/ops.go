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

func MatixMultiply(a *Tensor, b *Tensor) (*Tensor, error) {
	if len(a.Shape()) != 2 {
		return nil, errors.New("Matrix multiplication only works for 2D matrixes")
	}

	if len(b.Shape()) != 2 {
		return nil, errors.New("Matrix multiplication only works for 2D matrixes")
	}

	if a.Shape()[1] != b.Shape()[0] {
		return nil, errors.New("For matrix multiplicaiton columns of first matrix(a) should match row of second matrix(b)")
	}

	transposedB, err := b.Transpose2D()

	if err != nil {
		return nil, err
	}

	newRow := a.Shape()[0]
	newColumn := b.Shape()[1]

	newData := make([]float64, newRow * newColumn)

	newTensor := NewTensor(newData, []int{newRow, newColumn})

	for row:=0; row < newRow; row++ {
		rowMatrix := a.GetRow(row)

		for column:=0; column < newColumn; column++ {

			columnMatrix := transposedB.GetRow(column)

			multiplied, err := DotProduct(rowMatrix, columnMatrix)

			if err != nil {
				return nil, err
			}

			newTensor.SetValue(multiplied, []int{row, column})
		}
	}

	return newTensor, nil
}


