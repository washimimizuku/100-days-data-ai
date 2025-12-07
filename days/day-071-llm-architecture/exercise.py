"""
Day 71: LLM Architecture - Exercises

Practice implementing transformer components.
"""

import torch
import torch.nn as nn
import math


def exercise_1_scaled_dot_product_attention():
    """
    Exercise 1: Scaled Dot-Product Attention
    
    Implement the core attention mechanism:
    - Take Q, K, V matrices as input
    - Compute attention scores: QK^T / sqrt(d_k)
    - Apply softmax to get attention weights
    - Multiply weights by V to get output
    
    TODO: Implement scaled_dot_product_attention function
    TODO: Test with sample Q, K, V tensors
    TODO: Print attention weights and output
    """
    # TODO: Implement function
    # def scaled_dot_product_attention(Q, K, V):
    #     d_k = Q.size(-1)
    #     scores = ...
    #     attn_weights = ...
    #     output = ...
    #     return output, attn_weights
    
    # TODO: Test with sample data
    pass


def exercise_2_positional_encoding():
    """
    Exercise 2: Positional Encoding
    
    Create sinusoidal positional encodings:
    - PE(pos, 2i) = sin(pos / 10000^(2i/d))
    - PE(pos, 2i+1) = cos(pos / 10000^(2i/d))
    - Generate for max_len=100, embed_dim=512
    
    TODO: Implement create_positional_encoding function
    TODO: Visualize first 10 positions
    TODO: Check that encodings are unique for each position
    """
    # TODO: Implement function
    pass


def exercise_3_causal_mask():
    """
    Exercise 3: Causal Mask
    
    Generate causal attention mask for autoregressive models:
    - Create upper triangular matrix
    - Mask future positions (set to -inf or 0)
    - Test with different sequence lengths
    
    TODO: Implement create_causal_mask function
    TODO: Generate masks for seq_len = 4, 8, 16
    TODO: Visualize the mask pattern
    """
    # TODO: Implement function
    pass


def exercise_4_transformer_block():
    """
    Exercise 4: Transformer Block
    
    Build a complete transformer encoder block:
    - Multi-head self-attention
    - Add & Norm (residual connection + layer norm)
    - Feed-forward network
    - Add & Norm
    
    TODO: Implement TransformerBlock class
    TODO: Test with random input (batch=2, seq_len=10, embed_dim=512)
    TODO: Verify output shape matches input shape
    """
    # TODO: Implement class
    pass


def exercise_5_parameter_counting():
    """
    Exercise 5: Parameter Counting
    
    Calculate total parameters in a transformer model:
    - Embedding layer
    - Positional encoding (if learned)
    - N transformer blocks (attention + FFN)
    - Output layer
    
    TODO: Implement count_parameters function
    TODO: Calculate for GPT-2 small (12 layers, 768 dim, 12 heads)
    TODO: Calculate for BERT-base (12 layers, 768 dim, 12 heads)
    TODO: Compare with actual model sizes
    """
    # TODO: Implement function
    pass


if __name__ == "__main__":
    print("Day 71: LLM Architecture - Exercises\n")
    
    print("=" * 60)
    print("Exercise 1: Scaled Dot-Product Attention")
    print("=" * 60)
    # exercise_1_scaled_dot_product_attention()
    
    print("\n" + "=" * 60)
    print("Exercise 2: Positional Encoding")
    print("=" * 60)
    # exercise_2_positional_encoding()
    
    print("\n" + "=" * 60)
    print("Exercise 3: Causal Mask")
    print("=" * 60)
    # exercise_3_causal_mask()
    
    print("\n" + "=" * 60)
    print("Exercise 4: Transformer Block")
    print("=" * 60)
    # exercise_4_transformer_block()
    
    print("\n" + "=" * 60)
    print("Exercise 5: Parameter Counting")
    print("=" * 60)
    # exercise_5_parameter_counting()
