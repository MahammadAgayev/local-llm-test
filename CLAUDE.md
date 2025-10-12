# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Go-based educational implementation of transformer architecture components for local LLM experimentation. The project implements core building blocks from scratch: tensors, embeddings, attention mechanisms, and tokenization.

## Build and Test Commands

Build:
```bash
go build -o local-llm-test
```

Run tests:
```bash
go test ./...
```

Run tests for specific package:
```bash
go test ./pkg/tensor
go test ./pkg/embedding
go test ./pkg/tokenizer
go test ./pkg/transformer
```

Run tests with verbose output:
```bash
go test -v ./...
```

## Architecture

### Core Data Structure: Tensor
The `tensor` package (`pkg/tensor/`) provides the foundation:
- `Tensor` struct wraps a 1D `[]float64` array with multi-dimensional `shape []int`
- Uses row-major flattening: multi-dimensional indices are mapped to 1D positions
- Key operations: `Value(pos)`, `GetRow(row)`, `DotProduct`, `Softmax`

### Embedding Layer
The `embedding` package (`pkg/embedding/`) manages word representations:
- `Embdedding` (note the typo in the codebase) wraps a Tensor of shape `[vocabSize, embeddingDim]`
- Initialized with random values in range `[-0.1, 0.1]`
- `GetEmbedding(tokenId)` retrieves the vector for a given token

### Tokenization
The `tokenizer` package (`pkg/tokenizer/`) provides character-level tokenization:
- `CharTokenizer` builds vocabulary dynamically during encoding
- Maintains bidirectional mappings: `char <-> tokenId`
- Encodes text to `[]int`, decodes tokens back to strings

### Attention Mechanism
The `transformer` package (`pkg/transformer/`) implements attention:
- `ComputeAttentionSingle`: computes attention weights for one query position against all positions
- `ComputeAttention`: computes full attention matrix `[seq_len, seq_len]`
- Process: dot product → softmax → attention weights

### Data Flow
1. Text → CharTokenizer → token IDs
2. Token IDs → Embedding → vectors `[seq_len, embed_dim]`
3. Position embeddings added element-wise to token embeddings
4. Combined embeddings → Attention → attention matrix `[seq_len, seq_len]`
5. Attention weights × embeddings → context-aware representations

## Module Path
All imports use: `github.com/MahammadAgayev/local-llm-test`
