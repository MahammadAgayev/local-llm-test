package tensor


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
	size := t.subLayer(0)

	return t.data[startIdx:startIdx+size]
}

func (t *Tensor) flatten(pos []int) int {
	// for index 0 we have multiply it with rest of the values
	// like index[0] * shape[1..N]

	idx := 0
	for i, v := range pos {
		idx+= v * t.subLayer(i)
	}

	return idx
}

// Calculater the sublayer of given dimension, Can be optimized but not now
func (t *Tensor) subLayer(d int) int {
	layerVolume := 1
	for i := d + 1; i < len(t.shape); i++ {
		layerVolume *= t.shape[i]
	}

	return layerVolume
}


