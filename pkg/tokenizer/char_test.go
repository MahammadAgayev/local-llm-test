package tokenizer

import(
	"testing"
	"fmt"
)


func TestSimpleEncodeDecodeWorks(t *testing.T) {
	c := NewCharTokenizer()

	words := []string{
		"hi",
		"hello",
		"hell",
	}

	for _, word := range words {
		t.Run(fmt.Sprintf("Testing %s", word), func(t *testing.T) {
			token := c.Encode(word)
			decoded := c.Decode(token)

			if decoded != word {
				t.Errorf("Expected %s, got %s", word, decoded)
			}
		})
	}
}

