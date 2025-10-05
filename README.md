# local-llm-test

## Tensor
To simply put this multidimensional array represented by single dimensional array. The reason is be to able to plan with dimnesions of matrix dynamically.

## Embedding
The embedding is wrapper around a tensor representing the numerical information related to text. The embedding internall manages the Tensor.

## Attention
Thinking vector as geometric represention of the word, attention is effectively geometric relation between vector calculated using DotProduct. We later use softmax to calculate weights and probability distribution between vectors.

## Overall Process (Up to now)
Your process (corrected):

Generate token embeddings (one vector per token)
Generate position embeddings (one vector per position)
Sum them element-wise: combined_embed[i] = token_embed[i] + position_embed[i]

This gives you a tensor of shape [sequence_length, embedding_dim]


Compute attention weights using these combined embeddings

Output: attention matrix [sequence_length, sequence_length]

Use attention weights to compute new vectors (weighted combinations of the combined embeddings)

Output: new tensor of shape [sequence_length, embedding_dim]

