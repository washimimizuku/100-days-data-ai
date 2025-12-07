"""
Day 71: LLM Architecture - Solutions
"""

import torch
import torch.nn as nn
import math


def exercise_1_scaled_dot_product_attention():
    """Exercise 1: Scaled Dot-Product Attention"""
    
    def scaled_dot_product_attention(Q, K, V, mask=None):
        """Compute scaled dot-product attention"""
        d_k = Q.size(-1)
        
        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
        
        # Apply mask if provided
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # Apply softmax
        attn_weights = torch.softmax(scores, dim=-1)
        
        # Multiply by values
        output = torch.matmul(attn_weights, V)
        
        return output, attn_weights
    
    # Test with sample data
    batch_size, seq_len, d_k = 2, 4, 8
    Q = torch.randn(batch_size, seq_len, d_k)
    K = torch.randn(batch_size, seq_len, d_k)
    V = torch.randn(batch_size, seq_len, d_k)
    
    output, attn_weights = scaled_dot_product_attention(Q, K, V)
    
    print(f"Input shape: {Q.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Attention weights shape: {attn_weights.shape}")
    print(f"\nAttention weights (first batch, first query):")
    print(attn_weights[0, 0])
    print(f"Sum of weights: {attn_weights[0, 0].sum():.4f} (should be 1.0)")


def exercise_2_positional_encoding():
    """Exercise 2: Positional Encoding"""
    
    def create_positional_encoding(max_len, embed_dim):
        """Create sinusoidal positional encodings"""
        pe = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * 
                            -(math.log(10000.0) / embed_dim))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        return pe
    
    # Generate positional encodings
    max_len, embed_dim = 100, 512
    pe = create_positional_encoding(max_len, embed_dim)
    
    print(f"Positional encoding shape: {pe.shape}")
    print(f"\nFirst 10 positions, first 8 dimensions:")
    print(pe[:10, :8])
    
    # Check uniqueness
    print(f"\nAre all positions unique? {len(torch.unique(pe, dim=0)) == max_len}")
    
    # Visualize pattern
    print(f"\nPosition 0 (first 8 dims): {pe[0, :8]}")
    print(f"Position 1 (first 8 dims): {pe[1, :8]}")
    print(f"Position 50 (first 8 dims): {pe[50, :8]}")


def exercise_3_causal_mask():
    """Exercise 3: Causal Mask"""
    
    def create_causal_mask(seq_len):
        """Create causal mask for autoregressive attention"""
        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1)
        return mask == 0  # True where attention is allowed
    
    # Generate masks for different lengths
    for seq_len in [4, 8]:
        mask = create_causal_mask(seq_len)
        print(f"\nCausal mask for seq_len={seq_len}:")
        print(mask.int())
        print(f"Shape: {mask.shape}")
    
    # Demonstrate usage
    seq_len = 4
    mask = create_causal_mask(seq_len)
    scores = torch.randn(1, seq_len, seq_len)
    
    print(f"\nOriginal scores:")
    print(scores[0])
    
    masked_scores = scores.masked_fill(mask.unsqueeze(0) == 0, -1e9)
    print(f"\nMasked scores (future positions set to -inf):")
    print(masked_scores[0])


def exercise_4_transformer_block():
    """Exercise 4: Transformer Block"""
    
    class MultiHeadAttention(nn.Module):
        def __init__(self, embed_dim, num_heads):
            super().__init__()
            self.num_heads = num_heads
            self.head_dim = embed_dim // num_heads
            
            self.qkv = nn.Linear(embed_dim, embed_dim * 3)
            self.out = nn.Linear(embed_dim, embed_dim)
        
        def forward(self, x):
            batch_size, seq_len, embed_dim = x.shape
            
            # Project to Q, K, V
            qkv = self.qkv(x).reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)
            qkv = qkv.permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]
            
            # Attention
            scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
            attn = torch.softmax(scores, dim=-1)
            output = torch.matmul(attn, v)
            
            # Reshape and project
            output = output.transpose(1, 2).reshape(batch_size, seq_len, embed_dim)
            return self.out(output)
    
    class TransformerBlock(nn.Module):
        def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1):
            super().__init__()
            self.attention = MultiHeadAttention(embed_dim, num_heads)
            self.norm1 = nn.LayerNorm(embed_dim)
            self.ff = nn.Sequential(
                nn.Linear(embed_dim, ff_dim),
                nn.GELU(),
                nn.Linear(ff_dim, embed_dim)
            )
            self.norm2 = nn.LayerNorm(embed_dim)
            self.dropout = nn.Dropout(dropout)
        
        def forward(self, x):
            # Self-attention with residual
            attn_out = self.attention(self.norm1(x))
            x = x + self.dropout(attn_out)
            
            # Feed-forward with residual
            ff_out = self.ff(self.norm2(x))
            x = x + self.dropout(ff_out)
            
            return x
    
    # Test the block
    batch_size, seq_len, embed_dim = 2, 10, 512
    num_heads, ff_dim = 8, 2048
    
    block = TransformerBlock(embed_dim, num_heads, ff_dim)
    x = torch.randn(batch_size, seq_len, embed_dim)
    
    output = block(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Shapes match: {x.shape == output.shape}")
    
    # Count parameters
    total_params = sum(p.numel() for p in block.parameters())
    print(f"\nTotal parameters in block: {total_params:,}")


def exercise_5_parameter_counting():
    """Exercise 5: Parameter Counting"""
    
    def count_transformer_parameters(vocab_size, embed_dim, num_layers, num_heads, ff_dim):
        """Calculate total parameters in a transformer model"""
        
        # Token embeddings
        token_embed = vocab_size * embed_dim
        
        # Positional embeddings (learned)
        pos_embed = 512 * embed_dim  # Assuming max_len=512
        
        # Per transformer block
        # Multi-head attention: Q, K, V projections + output
        attn_params = 4 * (embed_dim * embed_dim)
        
        # Feed-forward network
        ff_params = (embed_dim * ff_dim) + (ff_dim * embed_dim)
        
        # Layer norms (2 per block)
        ln_params = 2 * (2 * embed_dim)  # gamma and beta for each
        
        block_params = attn_params + ff_params + ln_params
        total_blocks = num_layers * block_params
        
        # Output layer
        output_layer = embed_dim * vocab_size
        
        # Total
        total = token_embed + pos_embed + total_blocks + output_layer
        
        return {
            'token_embeddings': token_embed,
            'positional_embeddings': pos_embed,
            'transformer_blocks': total_blocks,
            'output_layer': output_layer,
            'total': total
        }
    
    # GPT-2 Small
    print("GPT-2 Small Configuration:")
    gpt2_params = count_transformer_parameters(
        vocab_size=50257,
        embed_dim=768,
        num_layers=12,
        num_heads=12,
        ff_dim=3072
    )
    print(f"Total parameters: {gpt2_params['total']:,}")
    print(f"Actual GPT-2 Small: ~124M parameters")
    print()
    
    # BERT-base
    print("BERT-base Configuration:")
    bert_params = count_transformer_parameters(
        vocab_size=30522,
        embed_dim=768,
        num_layers=12,
        num_heads=12,
        ff_dim=3072
    )
    print(f"Total parameters: {bert_params['total']:,}")
    print(f"Actual BERT-base: ~110M parameters")
    print()
    
    # Breakdown
    print("Parameter Breakdown (BERT-base):")
    for key, value in bert_params.items():
        if key != 'total':
            percentage = (value / bert_params['total']) * 100
            print(f"  {key}: {value:,} ({percentage:.1f}%)")


if __name__ == "__main__":
    print("Day 71: LLM Architecture - Solutions\n")
    
    print("=" * 60)
    print("Exercise 1: Scaled Dot-Product Attention")
    print("=" * 60)
    exercise_1_scaled_dot_product_attention()
    
    print("\n" + "=" * 60)
    print("Exercise 2: Positional Encoding")
    print("=" * 60)
    exercise_2_positional_encoding()
    
    print("\n" + "=" * 60)
    print("Exercise 3: Causal Mask")
    print("=" * 60)
    exercise_3_causal_mask()
    
    print("\n" + "=" * 60)
    print("Exercise 4: Transformer Block")
    print("=" * 60)
    exercise_4_transformer_block()
    
    print("\n" + "=" * 60)
    print("Exercise 5: Parameter Counting")
    print("=" * 60)
    exercise_5_parameter_counting()
