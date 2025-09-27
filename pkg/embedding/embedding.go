package embedding

import (
	"math/rand"

	"github.com/MahammadAgayev/local-llm-test/pkg/tensor"
)

type Embdedding struct {
	vector tensor.Tensor
	vocabSize int
	embeddingDim int
}

func NewEmbedding(vocabSize int, embeddingDim int) *Embdedding {
	size := vocabSize * embeddingDim

	arr := initArrayWithRand(size)
	return &Embdedding{
		vector: *tensor.NewTensor(arr, []int{ vocabSize, embeddingDim}),
		vocabSize: vocabSize,
		embeddingDim: embeddingDim,
	}
}

func initArrayWithRand(size int) []float64 {
	arr := make([]float64, size)

	for i := range arr {
		arr[i] = (rand.Float64() - 0.5) * 0.2
	}

	return arr
}

func (e *Embdedding) GetEmbedding(tokenId int) []float64 {
	return e.vector.GetRow(tokenId)
}

func (e *Embdedding) VocabSize() int {
	return e.vocabSize
}
