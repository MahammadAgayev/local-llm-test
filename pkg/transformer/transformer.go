package transformer

import (
	"github.com/MahammadAgayev/local-llm-test/pkg/embedding"
	"github.com/MahammadAgayev/local-llm-test/pkg/tensor"
)

//Calculating attention for while vocabulary is wrong, but how i should choose which i should calculate against, ????
// THIS IS NOT CORRECT
func ComputeAttention(embedding *embedding.Embdedding, tokenId int) ([]float64, error) {

	tokenEmbedding := embedding.GetEmbedding(tokenId)

	attentionScores := make([]float64, embedding.VocabSize())

	for i := 0; i < embedding.VocabSize(); i++ {
		attentionEmbedding := embedding.GetEmbedding(i)

		attentionScore, err := tensor.DotProduct(tokenEmbedding, attentionEmbedding)
		if err != nil {
			return nil, err
		}

		attentionScores[i] = attentionScore
	}

	attentionWeights := tensor.Softmax(attentionScores)

	return attentionWeights, nil
}
