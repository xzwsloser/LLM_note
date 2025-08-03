import torch
from torch import nn
import math

# 实现缩放点积注意力
def attention(Q,K,V, dropout=0):
    dim = Q.shape[-1]
    # Q.size -> (batch_size, dim)
    # K.size -> (batch_size, dim)
    attention_weights = Q @ K.transpose(-1, -2) / math.sqrt(dim)
    attention_weights = torch.softmax(attention_weights, dim=-1)
    attention_weights = torch.dropout(attention_weights, dropout, train=True)
    output = attention_weights @ V
    return output, attention_weights

# 实现自注意力
class SelfAttention(nn.Module):
    def __init__(self, dim, dropout=0.1, **kwargs):
       super(SelfAttention, self).__init__(**kwargs)
       self.dim = dim
       self.w_q = nn.Linear(dim, dim)
       self.w_k = nn.Linear(dim, dim)
       self.w_v = nn.Linear(dim, dim)
       self.dropout = nn.Dropout(dropout)
       self.w_o = nn.Linear(dim, dim)
    def forward(self, X, attention_mask=None):
        # X.size -> (batch_size, seq_len, dim)
        # Q,K,V.size -> (batch_size, seq_len, dim)
        Q, K, V = self.w_q(X), self.w_k(X), self.w_v(X)
        attention_weights = Q @ K.transpose(-1, -2) / math.sqrt(self.dim)
        if attention_mask is not None:
            attention_weights = attention_weights.masked_fill(
                attention_mask == 0,
                float("-inf")
            )
        # attention_weights.size -> (batch_size, seq_len, seq_len)
        attention_weights = torch.softmax(attention_weights, dim=-1)
        attention_weights = self.dropout(attention_weights)
        print('======Attention Weights========')
        print(attention_weights)
        print('===============================')
        output = attention_weights @ V
        # output.size -> (batch_size, seq_len, dim)
        output = self.w_o(output)
        return output

def get_mask(batch_size, heads, seq_len):
    mask = torch.full(size=(1, seq_len, seq_len), fill_value=1)
    mask = 1 - torch.triu(mask, diagonal=1)
    mask = mask.unsqueeze(0).repeat((batch_size, heads, 1, 1))
    return mask

class MultiHeadAttention(nn.Module):
    def __init__(self, dim, heads, dropout_rate=0.1, **kwargs):
        super(MultiHeadAttention, self).__init__(**kwargs)
        self.hidden_dim = dim // heads
        self.heads = heads
        self.w_q = nn.Linear(dim, dim)
        self.w_k = nn.Linear(dim, dim)
        self.w_v = nn.Linear(dim, dim)
        self.w_o = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout_rate)
    def get_causal_mask(self, batch_size, seq_len):
        mask = torch.full(size=(1, seq_len, seq_len), fill_value=1)
        mask = 1 - torch.triu(mask, diagonal=1)
        mask = mask.unsqueeze(0).repeat((batch_size, self.heads, 1, 1))
        return mask
    def forward(self, X, is_causal=False):
        # X.shape -> (batch_size, seq_len, dim)
        batch_size, seq_len, _ = X.shape
        Q, K, V = self.w_q(X), self.w_k(X), self.w_v(X)
        # Q, K, V.shape -> (batch_size, seq_len, self.heads*self.hidden_dim)
        Q = Q.reshape((batch_size, seq_len, -1, self.hidden_dim)).transpose(-2, -3)
        K = K.reshape((batch_size, seq_len, -1, self.hidden_dim)).transpose(-2, -3)
        V = V.reshape((batch_size, seq_len, -1, self.hidden_dim)).transpose(-2, -3)
        # Q, K, V.shape -> (batch_size, heads, seq_len, hidden_dim)
        attention_weights = Q @ K.transpose(-1, -2) / math.sqrt(self.hidden_dim)
        # attention_weights.shape -> (batch_size, heads, seq_len, seq_len)
        if is_causal:
            mask = self.get_causal_mask(batch_size=batch_size, seq_len=seq_len)
            attention_weights = attention_weights.masked_fill(
                mask == 0,
                float('-inf')
            )
        attention_weights = torch.softmax(attention_weights, dim=-1)
        attention_weights = self.dropout(attention_weights)
        Out = attention_weights @ V
        # Out.shape -> (batch_size, heads, seq_len, dim)
        Out = Out.transpose(-2, -3).reshape(batch_size, seq_len, -1)
        # Out.shape -> (batch_size, seq_len, heads * dim)
        Out = self.w_o(Out)
        return Out

if __name__ == '__main__':
    Q = torch.rand(size=(3, 4))
    K = torch.rand(size=(3, 4))
    V = torch.rand(size=(3, 2))
    output, attention_weights = attention(Q, K, V)
    print(output)
    print('===========')
    print(attention_weights)
    print('==========')
    X = torch.rand(size=(3, 3, 4))
    sat = SelfAttention(dim=4, dropout=0)
    output = sat(X)
    print('========Output=========')
    print(output)
    print('=======================')
    mask = get_mask(batch_size=3, heads=2, seq_len=3)
    print(mask)
    print('========Mask===========')
    # output = sat(X, attention_mask=mask)
    print('=======================')
    print('==========Out==========')
    print(output)
    print('=======================')

    print('=============== MultiHeadAttention ================')
    mat = MultiHeadAttention(dim=128, heads=8, dropout_rate=0)
    X = torch.randn(size=(3, 4, 128))
    Out = mat(X, is_causal=True)
    print('Out.shape: ', Out.shape)
    print('===================================================')