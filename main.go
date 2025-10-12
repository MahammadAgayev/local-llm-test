package main

import (
	"fmt"

	"github.com/MahammadAgayev/local-llm-test/pkg/embedding"
	"github.com/MahammadAgayev/local-llm-test/pkg/transformer"
)

func main() {

	//simpleWord := "cat"

	positionEmbedding := embedding.NewEmbedding(3, 3)
	positionTensor := positionEmbedding.GetTensory()

	positionTensor.Print()

	newTensor, err := transformer.ApplyAttention(positionTensor)

	if err != nil {
		print(err)
		return
	}

	fmt.Printf("Applied attention.....")
	newTensor.Print()
}
