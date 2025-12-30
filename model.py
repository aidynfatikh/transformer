import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class InputEmbedding(nn.Module):
    def __init__(self, vocab_size, embd_dim, seq_length):
        super().__init__()

        self.tok_embd = nn.Embedding(vocab_size, embd_dim)
        self.pos_embd = nn.Embedding(seq_length, embd_dim)

    def forward(self, tokens):
        _, T = tokens.shape

        tok = self.tok_embd(tokens)
        pos = self.pos_embd(torch.arange(T, device=tokens.device))

        return tok + pos     

class Head(nn.Module):
    def __init__(self, hidden_size, head_size, dropout):
        super().__init__()

        self.W_Q = nn.Linear(hidden_size, head_size, bias=False) 
        self.W_K = nn.Linear(hidden_size, head_size, bias=False)
        self.W_V = nn.Linear(hidden_size, head_size, bias=False)
        self.d = nn.Dropout(dropout)
        
    def forward(self, x, mask=True):
        B, T, C = x.shape
        Q = self.W_Q(x) # [B, T, head_size]
        K = self.W_K(x) # [B, T, head_size]
        V = self.W_V(x) # [B, T, head_size]

        head_size = Q.shape[-1]

        scores = Q @ K.transpose(-2, -1) / np.sqrt(head_size) # [B, T, T]
        
        if mask:
            # Apply causal mask - prevent attention to future tokens
            mask = torch.tril(torch.ones(T, T, device=x.device))
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        scores = F.softmax(scores, dim=-1)
        scores = self.d(scores)
        scores = scores @ V

        return scores

class MultiHead(nn.Module):
    def __init__(self, hidden_size, n_heads, dropout):
        super().__init__()
        head_size = hidden_size // n_heads
        self.heads = nn.ModuleList([Head(hidden_size, head_size, dropout) for _ in range(n_heads)])
        self.proj = nn.Linear(hidden_size, hidden_size)

    def forward(self, x, mask=True):
        x = torch.cat([h(x, mask) for h in self.heads], dim=-1)
        x = self.proj(x)  # Apply projection
        return x
    

class FeedFoward(nn.Module):
    def __init__(self, hidden_size, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_size, 4 * hidden_size),
            nn.ReLU(),
            nn.Linear(4 * hidden_size, hidden_size),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)
    
class DecoderBlock(nn.Module):
    def __init__(self, hidden_size, n_heads, dropout):
        super().__init__()

        self.mla = MultiHead(hidden_size, n_heads, dropout)
        self.ln1 = nn.LayerNorm(hidden_size)
        self.ff = FeedFoward(hidden_size, dropout)
        self.ln2 = nn.LayerNorm(hidden_size)

    def forward(self, x):
        # Pre-norm with residual connections
        x = x + self.mla(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x
    

class CrossHead(nn.Module):
    def __init__(self, hidden_size, head_size, dropout):
        super().__init__()

        self.W_Q = nn.Linear(hidden_size, head_size, bias=False) 
        self.W_K = nn.Linear(hidden_size, head_size, bias=False)
        self.W_V = nn.Linear(hidden_size, head_size, bias=False)
        self.d = nn.Dropout(dropout)
        
    def forward(self, enc, dec):
        B, T, C = enc.shape
        Q = self.W_Q(dec) # [B, T, head_size]
        K = self.W_K(enc) # [B, T, head_size]
        V = self.W_V(enc) # [B, T, head_size]

        head_size = Q.shape[-1]

        scores = Q @ K.transpose(-2, -1) / np.sqrt(head_size) # [B, T, T]
        
        scores = F.softmax(scores, dim=-1)
        scores = self.d(scores)
        scores = scores @ V

        return scores

class CrossMultiHead(nn.Module):
    def __init__(self, hidden_size, n_heads, dropout):
        super().__init__()
        head_size = hidden_size // n_heads
        self.heads = nn.ModuleList([CrossHead(hidden_size, head_size, dropout) for _ in range(n_heads)])
        self.proj = nn.Linear(hidden_size, hidden_size)

    def forward(self, enc, dec):
        x = torch.cat([h(enc, dec) for h in self.heads], dim=-1)
        x = self.proj(x)
        return x

class EncoderBlock(nn.Module):
    def __init__(self, hidden_size, n_heads, dropout):
        super().__init__()

        self.mla = MultiHead(hidden_size, n_heads, dropout)
        self.ln1 = nn.LayerNorm(hidden_size)
        self.ff = FeedFoward(hidden_size, dropout)
        self.ln2 = nn.LayerNorm(hidden_size)

    def forward(self, x):
        x = x + self.mla(self.ln1(x), mask=False)
        x = x + self.ff(self.ln2(x))
        return x

