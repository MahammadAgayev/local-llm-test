package tensor

import (
	"errors"
	"fmt"
	"strings"
)


type Tensor struct {
	data []float64
	shape []int
}

func NewTensor(data []float64, shape []int) *Tensor {
	return &Tensor{
		data: data,
		shape: shape,
	}
}

func (t *Tensor) Len() int {
	return len(t.data)
}

func (t *Tensor) Value(pos []int) float64 {
	// for index 0 we have multiply it with rest of the values
	// like index[0] * shape[1..N]

	idx := t.flatten(pos)

	return t.data[idx]
}

func (t *Tensor) GetRow(row int) []float64 {
	pos := make([]int, len(t.shape))
	pos[0] = row

	startIdx := t.flatten(pos)
	size := sublayer(0, t.shape)

	return t.data[startIdx:startIdx+size]
}

func (t *Tensor) DimSize(dim int) int {
	return t.shape[dim]
}

func (t *Tensor) Shape() []int {
	return t.shape
}

func (t *Tensor) flatten(pos []int) int {
	// for index 0 we have multiply it with rest of the values
	// like index[0] * shape[1..N]

	idx := flatten(pos, t.shape)

	return idx
}

func (t *Tensor) SetRow(row []float64, rowIdx int) error {
	volume := sublayer(1, t.shape)
	if len(row) != volume {
		return errors.New(fmt.Sprintf("Row size do not match, shape %d, adding %d", t.shape[1], len(row)))
	}

	pos := []int{rowIdx, 0}

	for i:=0; i<=volume; i++{
		pos[1]+=i

		idx := t.flatten(pos)

		t.data[idx] = row[i]
	}

	return nil
}

func (t *Tensor) SetValue(val float64, pos []int) {
	idx := t.flatten(pos)

	t.data[idx] = val
}

func (t *Tensor) Transpose2D() (*Tensor, error) {
	if len(t.shape) != 2 {
		return nil, errors.New("Cannot transpose tensor more 2 dim")
	}

	originalRow, originalColumn := t.shape[0], t.shape[1]
	newRow, newColumn := originalColumn, originalRow

	newData := make([]float64, originalRow * originalColumn)

	for row := 0; row < originalRow; row++ {
		for column := 0; column < originalColumn; column++ {

			newIdx := flatten([]int{column, row}, []int{newRow, newColumn})

			newData[newIdx] = t.Value([]int{row, column})
		}
	}

	newTensor := NewTensor(newData, []int{newRow, newColumn})

	return newTensor, nil
}

func (t *Tensor) Print() {
	fmt.Println("\n" + strings.Repeat("═", 60))
	fmt.Printf("│ Tensor Shape: %v\n", t.shape)
	fmt.Printf("│ Elements: %d\n", len(t.data))
	fmt.Println("├" + strings.Repeat("─", 58))
	fmt.Print(t.String())
	fmt.Println("\n" + strings.Repeat("═", 60) + "\n")
}

func (t *Tensor) String() string {
	return t.formatTensor(t.shape, 0, "")
}

func (t *Tensor) formatTensor(currentShape []int, offset int, indent string) string {
	if len(currentShape) == 1 {
		var elements []string
		for i := 0; i < currentShape[0]; i++ {
			elements = append(elements, fmt.Sprintf("%8.4f", t.data[offset+i]))
		}
		return "[" + strings.Join(elements, " ") + "]"
	}

	var result strings.Builder
	result.WriteString("[\n")
	
	stride := 1
	for i := 1; i < len(currentShape); i++ {
		stride *= currentShape[i]
	}
	
	for i := 0; i < currentShape[0]; i++ {
		result.WriteString(indent + "  ")
		childResult := t.formatTensor(currentShape[1:], offset+i*stride, indent+"  ")
		result.WriteString(childResult)
		if i < currentShape[0]-1 {
			result.WriteString(",")
		}
		result.WriteString("\n")
	}
	
	result.WriteString(indent + "]")
	return result.String()
}


