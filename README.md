# Nano LLM - Decoder-Only
This repository implements a minimal decoder-only Transformer, inspired by GPT architectures.

The goal of this project is to understand and implement the core concepts of a modern LLM from first priciples, without relying on high-level libraries.

## Blogs
- [Why GPTs Are Decoder-Only](https://imartinezcuevas.github.io/posts/why-gpts-are-decoder-only/)  
  Explains why modern GPT models use a decoder-only Transformer.
- [How GPTs processes tokens and KV Cache](https://imartinezcuevas.github.io/posts/gpt-flow-kv-cache/)
  How information flows inside GPT during training and inference, explaining why autoregressive generation becomes expensive and how KV cache is used as an optimization.

## Objectives

### 1. Embeddings
- [x] **TokenEmbedding**
  - Learnable lookup table `(vocab_size, d_model)`
  - Forward = direct indexing
  - Updated via backprop only
- [x] **PositionEmbedding**
  - Learned positional embeddings
  - Fixed max sequence length
  - Added to token embeddings

### 2. Attention
- [x] **Scaled Dot-Product Attention**
  - Proper scaling by `sqrt(d_k)`
  - Softmax + optional dropout
  - Boolean masking support
- [x] **Multi-Head Causal Self-Attention**
  - Q/K/V projections
  - Multi-head split
  - Causal (lower-triangular) mask
  - Optional padding mask
  - KV-cache support

### 3. Transformer Block
- [x] **TransformerBlock**
  - Causal multi-head self-attention
  - Residual connections
  - Layer normalization
  - Feedforward network (2-layer MLP with ReLU)
  - Dropout

### 4. Model
- [x] **MiniTransformer**
  - Decoder-only
  - Stack of configurable blocks
  - Token + positional embeddings
  - Final projection to vocab
  - KV-cache per layer

### 5. Text Generation
- [x] **Autoregressive generation**
  - Token-by-token decoding
  - KV-cache reuse
  - Temperature sampling
  - Top-k sampling
  - Top-p sampling
  - Greedy decoding (`temperature=0`)

### 6. Training
- [x] Training loop


