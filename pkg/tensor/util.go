package tensor

func flatten(pos []int, shape []int) int {
	// for index 0 we have multiply it with rest of the values
	// like index[0] * shape[1..N]

	idx := 0
	for i, v := range pos {
		idx+= v * sublayer(i, shape)
	}

	return idx
}

// Calculater the sublayer of given dimension, Can be optimized but not now
func sublayer(d int, shape []int) int {
	layerVolume := 1
	for i := d + 1; i < len(shape); i++ {
		layerVolume *= shape[i]
	}

	return layerVolume
}
