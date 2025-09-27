package embedding

import (
	"fmt"
	"testing"
)

func TestSimpleEmbedding(t *testing.T) {
	embedding := NewEmbedding(5, 3)

	row := embedding.GetEmbedding(0)

	fmt.Println(row)
}
