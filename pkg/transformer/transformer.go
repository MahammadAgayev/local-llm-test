package transformer

import (
	"github.com/MahammadAgayev/local-llm-test/pkg/tensor"
)

func ComputeAttentionSingle(embeddings *tensor.Tensor, attentionPosition []float64) ([]float64, error) {

	attentionScores := make([]float64, embeddings.DimSize(0))

	for i := range attentionScores {
		embedding := embeddings.GetRow(i)

		attentionScore, err := tensor.DotProduct(embedding, attentionPosition)
		if err != nil {
			return nil, err
		}

		attentionScores[i] = attentionScore
	}

	attentionWeights := tensor.Softmax(attentionScores)

	return attentionWeights, nil
}

func ComputeAttention(embeddings *tensor.Tensor) (*tensor.Tensor, error) {
	attentionWeights := make([]float64, 0)

	rowSize := embeddings.DimSize(0)

	for i:=0;i<rowSize; i++ {
		attentionPosition := embeddings.GetRow(i)
		attentionWeight, err := ComputeAttentionSingle(embeddings, attentionPosition)

		if err != nil {
			return nil, err
		}

		attentionWeights = append(attentionWeights,  attentionWeight...)

	}

	tensor := tensor.NewTensor(attentionWeights,[]int{ rowSize, rowSize })

	return tensor, nil
}


func ApplyAttention(embeddings *tensor.Tensor) (*tensor.Tensor, error) {

	attentionTensor, err := ComputeAttention(embeddings)

	if err != nil {
		return nil, err
	}

	newTensor, err := tensor.MatixMultiply(embeddings, attentionTensor)

	if err != nil {
		return nil, err
	}

	return newTensor, nil
}
