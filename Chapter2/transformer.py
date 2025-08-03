import torch
from torch import nn
import torch.nn.functional as F
import math

# 手写 Transformer
# 基础组件
# 1. Feed Forward Network
class FFN(nn.Module):
    def __init__(self, dim, hidden_dim, dropout_rate=0.1):
        super(FFN, self).__init__()
        self.w_1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w_2 = nn.Linear(dim, hidden_dim, bias=False)
        self.dropout = nn.Dropout(dropout_rate)
    def forward(self, X):
        return self.w_2(self.dropout(F.relu(self.w_1(X))))

# 2. MultiHead Attention
class MultiHeadAttention(nn.Module):
    def __init__(self, dim, heads, dropout_rate=0.1):
        super(MultiHeadAttention, self).__init__()
        self.hidden_dim = dim // heads
        self.heads = heads
        self.w_q = nn.Linear(dim, dim)
        self.w_k = nn.Linear(dim, dim)
        self.w_v = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout_rate)
        self.w_o = nn.Linear(dim, dim)
    def get_causal_mask(self, batch_size, seq_len):
        mask = torch.full(size=(1, seq_len, seq_len), fill_value=1)
        mask = 1 - torch.triu(mask, diagonal=1)
        mask = mask.unsqueeze(0).repeat((batch_size, self.heads, seq_len, seq_len))
        return mask
    def forward(self, Q_in, K_in, V_in, is_causal=False):
        batch_size, seq_len, _ = Q_in.shape
        Q, K, V = self.w_q(Q_in), self.w_k(K_in), self.w_v(V_in)
        Q = Q.reshape((batch_size, seq_len, self.heads, -1)).permute(0, 2, 1, 3)
        K = K.reshape((batch_size, seq_len, self.heads, -1)).permute(0, 2, 1, 3)
        V = V.reshape((batch_size, seq_len, self.heads, -1)).permute(0, 2, 1, 3)
        attention_weights = Q @ K.transpose(-1, -2) / math.sqrt(self.hidden_dim)
        if is_causal:
            mask = self.get_causal_mask(batch_size, seq_len)
            attention_weights = attention_weights.masked_fill(
                mask == 0,
                float('-inf')
            )
        attention_weights = torch.softmax(attention_weights, dim=-1)
        attention_weights = self.dropout(attention_weights)
        out = attention_weights @ V
        out = out.permute(0, 2, 1, 3).reshape((batch_size, seq_len, -1))
        out = self.w_o(out)

# Positional Encoding
class PositionalEmbedding(nn.Module):
    def __init__(self, max_len=100, dim=512):
        super(PositionalEmbedding, self).__init__()
        self.P = torch.zeros(size=(1, max_len, dim))
        pos_matrix = torch.arange(0, max_len, 1.0, dtype=torch.float32).reshape((-1, 1))
        pow_matrix = torch.pow(1000, torch.arange(0, dim, 2, dtype=torch.float32)/dim)
        matrix = pos_matrix / pow_matrix
        self.P[:, :, 0::2] = torch.sin(matrix)
        self.P[:, :, 1::2] = torch.cos(matrix)
    def forward(self, X):
        return X + self.P[:, :X.shape[0], :]

# 2. 复合组件库
# Attention Block
class AttentionBlock(nn.Module):
    def __init__(self, dim, heads, dropout_rate=0.1):
        super(AttentionBlock, self).__init__()
        self.attention = MultiHeadAttention(dim, heads, dropout_rate)
        self.norm = nn.LayerNorm(dim)
    def forward(self, Q_in, K_in, V_in, is_causal=False):
        _X = Q_in
        Out = self.attention(Q_in, K_in, V_in, is_causal)
        Out = self.norm(_X + Out)
        return Out

# FFN Block
class FFNBlock(nn.Module):
    def __init__(self, dim, hidden_dim, dropout_rate=0.1):
        super(FFNBlock, self).__init__()
        self.ffn = FFN(dim, hidden_dim, dropout_rate)
        self.norm = nn.LayerNorm(dim)
    def forward(self, X):
        _X = X
        Out = self.ffn(X)
        Out = self.norm(_X + Out)
        return Out

# Word Embedding
class WordEmbedding(nn.Module):
    def __init__(self, vocab_size, dim, max_len=1000):
        super(WordEmbedding, self).__init__()
        self.token_embedding = nn.Embedding(vocab_size, dim)
        self.positional_embedding = PositionalEmbedding(max_len, dim)
    def forward(self, X):
        Y = self.token_embedding(X)
        Y = self.positional_embedding(X)
        return Y

# 3. Encoder, Decoder Block
class TransformerEncoderBlock(nn.Module):
    def __init__(self, dim, heads, hidden_dim, dropout_rate=0.1):
        super(TransformerEncoderBlock, self).__init__()
        self.attention_block = AttentionBlock(dim, heads, dropout_rate)
        self.ffn_block = FFNBlock(dim, hidden_dim, dropout_rate)
    def forward(self, X):
        return self.ffn_block(self.attention_block(X, X, X))

class TransformerDecoderBlock(nn.Module):
    def __init__(self, dim, heads, hidden_dim, dropout_rate=0.1):
        super(TransformerDecoderBlock, self).__init__()
        self.self_attention_block = AttentionBlock(dim, heads, dropout_rate)
        self.cross_attention_block = AttentionBlock(dim, heads, dropout_rate)
        self.ffn_block = FFN(dim, hidden_dim, dropout_rate)
    def forward(self, X, K, V):
        Y = self.self_attention_block(X, X, X, is_causal=True)
        Y = self.cross_attention_block(Y, K, V)
        Y = self.ffn_block(Y)
        return Y

# 4. TransformerEncoder and Transformer Decoder
class TransformerEncoder(nn.Module):
    def __init__(self, dim, heads, hidden_dim, num_layers=6, dropout_rate=0.1):
        super(TransformerEncoder, self).__init__()
        self.encoder_blocks = nn.ModuleList(
           [
               TransformerEncoderBlock(dim, heads, hidden_dim, dropout_rate)
                for _ in range(num_layers)
           ]
        )
    def forward(self, X):
        for layer in self.encoder_blocks:
            X = layer(X)
        return X

class TransformerDecoder(nn.Module):
    def __init__(self, dim, heads, hidden_dim, num_layers=6, dropout_rate=0.1):
        super(TransformerDecoder, self).__init__()
        self.decoder_blocks = nn.ModuleList(
            [
                TransformerDecoderBlock(dim, heads, hidden_dim, dropout_rate)
                for _ in range(num_layers)
            ]
        )
    def forward(self, X, K, V):
        for layer in self.decoder_blocks:
            X = layer(X, K, V)
        return X

# Transformer
class Transformer(nn.Module):
    def __init__(self,
                 vocab_size_in,
                 vocab_size_out,
                 max_len,
                 dim,
                 heads,
                 hidden_dim,
                 num_layers,
                 dropout_rate=0.1):
        super(Transformer, self).__init__()
        self.word_embedding_in = WordEmbedding(vocab_size_in, dim, max_len)
        self.word_embedding_out = WordEmbedding(vocab_size_out, dim, max_len)
        self.encoder = TransformerEncoder(dim, heads, hidden_dim, num_layers, dropout_rate)
        self.decoder = TransformerDecoder(dim, heads, hidden_dim, num_layers, dropout_rate)
        self.w_o = nn.Linear(dim, vocab_size_out)
    def forward(self, encoder_in, decoder_in):
        encoder_in = self.word_embedding_in(encoder_in)
        decoder_in = self.word_embedding_out(decoder_in)
        encoder_out = self.encoder(encoder_in)
        decoder_out = self.decoder(decoder_in, encoder_out, encoder_out)
        decoder_out = self.w_o(decoder_out)
        return decoder_out