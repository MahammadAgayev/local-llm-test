package tokenizer

import "strings"

type CharTokenizer struct {
	storage map[string]int
	reverseStorage map[int]string
	counter int
}

func NewCharTokenizer() *CharTokenizer {
	return &CharTokenizer{
		storage: make(map[string]int),
		reverseStorage: make(map[int]string),
		counter: 0,
	}
}

func (t *CharTokenizer) Encode(text string) []int {
	token := make([]int, 0, len(text))
	for _, char := range text {
		c := string(char)

		if _, exists := t.storage[c]; !exists {
			t.storage[c] = t.counter
			t.reverseStorage[t.counter] = c
			t.counter++
		}

		token = append(token, t.storage[c])
	}

	return token
}

func (t *CharTokenizer) Decode(tokens []int) string {
	var result strings.Builder

	for _, token := range tokens {
		if char, exists := t.reverseStorage[token]; exists{
			result.WriteString(char)
		} else {
			result.WriteString("-")
		}
	}

	return result.String()
}


