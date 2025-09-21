package embedding

import "github.com/MahammadAgayev/local-llm-test/pkg/tensor"

type Embdedding struct {
	tensor tensor.Tensor
}

func (e *Embdedding) GetEmbedding(tokenId int) []float64 {
	return e.tensor.GetRow(tokenId)
}
