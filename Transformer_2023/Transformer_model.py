import pandas as pd
from torch import nn
import torch
import matplotlib.pyplot as plt
from torch.utils import data
from torch.utils.data import DataLoader

class PositionEncoding(nn.Module):
    def __init__(self, input_dim, max_seq_len = 512):
        super().__init__()
        position = torch.arange(0, max_seq_len).unsqueeze(1)
        item = 1/10000**(torch.arange(0, input_dim, 2) / input_dim)
        tmp_pos = position * item
        pe = torch.zeros(max_seq_len, input_dim)
        pe[:, 0::2] = torch.sin(tmp_pos)
        pe[:, 1::2] = torch.cos(tmp_pos)

        # plt.matshow(pe)
        # plt.show()

        position_enconding = pe.unsqueeze(0)
        self.register_buffer("position_enconding", position_enconding, False)
    def forward(self, input):
        batch, seq_len, _ = input.shape
        pe = self.position_enconding
        return input + pe[:, :seq_len, :]

# if __name__ == "__main__":
#     pos = PositionEncoding(input_dim = 512, max_seq_len=  100)
#     print(list(pos.named_buffers()))

def attention(query, key, value, mask = None):
    input_dim = key.shape[-1]
    attention_score = torch.matmul(query, key.transpose(-2, -1)) / input_dim**0.5
    if mask is not None:
        attention_score = attention_score.masked_fill(mask, -1e9)
    attention_score = torch.softmax(attention_score, -1)
    return torch.matmul(attention_score, value)

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, input_dim, dropout = 0.1):
        super().__init__()
        self.query_liner = nn.Linear(input_dim, input_dim, bias=False)
        self.key_liner = nn.Linear(input_dim, input_dim, bias=False)
        self.value_liner = nn.Linear(input_dim, input_dim, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.num_heads = num_heads
        self.dim_Multi_Head = input_dim // num_heads
        self.input_dim = input_dim
        self.linear = nn.Linear(input_dim, input_dim, bias=False)
        self.mul_linear_q = nn.Linear(input_dim, self.dim_Multi_Head)
        self.mul_linear_k = nn.Linear(input_dim, self.dim_Multi_Head)
        self.mul_linear_v = nn.Linear(input_dim, self.dim_Multi_Head)

    def forward(self, query, key, value, mask = None):
        query = self.query_liner(query)
        key = self.key_liner(key)
        value = self.value_liner(value)
        output_list = []
        for i in range(self.num_heads):
            new_query = self.mul_linear_q(query)
            new_key = self.mul_linear_k(key)
            new_value = self.mul_linear_v(value)
            output = attention(new_query, new_key, new_value, mask)
            output_list.append(output)
        output = torch.cat(output_list, dim=-1)
        output = self.linear(output)
        final_output = self.dropout(output)
        return final_output

# if __name__ == "__main__":
#     atten = MultiHeadAttention(10, 50, 0.1)
#     input = torch.randn(5, 10, 50)
#     out = atten(input, input, input)
#     print(out.shape)

class FeedForward(nn.Module):
    def __init__(self, input_dim, middle_dim, dropout):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, middle_dim, bias=False),
            nn.ReLU(),
            nn.Linear(middle_dim, input_dim, bias=False),
            nn.Dropout(dropout)
        )
    def forward(self, input):
        return self.fc(input)

class EncoderLayer(nn.Module):
    def __init__(self, num_heads, input_dim, middle_dim, dropout = 0.1):
        super().__init__()
        self.multi_head = MultiHeadAttention(num_heads = num_heads, input_dim= input_dim, dropout= dropout)
        self.feedforward = FeedForward(input_dim= input_dim, middle_dim= middle_dim, dropout= dropout)
        self.norms = nn.ModuleList([nn.LayerNorm(input_dim) for i in range(2)])
        self.dropout = nn.Dropout(dropout)

    def forward(self, input, mask = None):
        multi_head_out = self.multi_head(input, input, input, mask)
        multi_head_out = self.norms[0](input + multi_head_out)
        feedforward_out = self.feedforward(multi_head_out)
        feedforward_out = self.norms[1](feedforward_out + multi_head_out)
        final_output = self.dropout(feedforward_out)
        return final_output

class Encoder(nn.Module):
    def __init__(self, input_dim, max_seq_len, num_layers, num_heads, middle_dim, dropout = 0.1):
        super().__init__()
        self.position_encoding = PositionEncoding(input_dim=input_dim, max_seq_len= max_seq_len)
        self.encoder = nn.ModuleList([EncoderLayer(num_heads= num_heads, input_dim= input_dim, middle_dim= middle_dim,
                                                  dropout= dropout) for i in range(num_layers)])
    def forward(self, input, src_mask = None):
        pos_enc_output = self.position_encoding(input)
        for layer in self.encoder:
            enc_output = layer(pos_enc_output, src_mask)
        return enc_output

class DecoderLayer(nn.Module):
    def __init__(self, num_heads, input_dim, middle_dim, max_seq_len, dropout = 0.1):
        super().__init__()
        self.masked_multi_head = MultiHeadAttention(num_heads= num_heads, input_dim= input_dim, dropout= dropout)
        self.attention = MultiHeadAttention(num_heads= num_heads, input_dim= input_dim, dropout= dropout)
        self.norms = nn.ModuleList([nn.LayerNorm(input_dim) for i in range(3)])
        self.feedforward = FeedForward(input_dim, middle_dim, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, target, enc_output, seq_mask = None):
        output = self.masked_multi_head(target, target, target, seq_mask)
        first_output = self.norms[0](output + target)
        output = self.attention(first_output, enc_output, enc_output)
        second_output = self.norms[1](output + first_output)
        output = self.feedforward(second_output)
        output = self.norms[2](output + second_output)
        final_output = self.dropout(output)
        return final_output

class Decoder(nn.Module):
    def __init__(self, input_dim, max_seq_len, num_layers, num_heads, middle_dim, dropout = 0.1):
        super().__init__()
        self.position_encoding = PositionEncoding(input_dim=input_dim, max_seq_len= max_seq_len)
        self.decoder_layers = nn.ModuleList([DecoderLayer(num_heads= num_heads, input_dim= input_dim, middle_dim= middle_dim, max_seq_len= max_seq_len,
                                                  dropout= dropout) for i in range(num_layers)])
    def forward(self, target, enc_outputkv, seq_mask = None):
        pos_enc_output = self.position_encoding(target)
        for layer in self.decoder_layers:
            dec_output = layer(pos_enc_output, enc_outputkv, seq_mask)
        return dec_output

class Transformer(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.inputlinear = nn.Linear(4, args.input_dim)
        self.inputlinear2 = nn.Linear(1, args.input_dim)

        self.encoder = Encoder(input_dim= args.input_dim,
                               max_seq_len= args.max_seq_len,
                               num_layers= args.num_layers,
                               num_heads= args.num_heads,
                               middle_dim= args.middle_dim,
                               dropout= args.dropout)
        self.decoder = Decoder(input_dim= args.input_dim,
                               max_seq_len= args.max_seq_len,
                               num_layers= args.num_layers,
                               num_heads= args.num_heads,
                               middle_dim= args.middle_dim,
                               dropout= args.dropout)
        self.linear = nn.Linear(args.input_dim, 1)

        self.flat = nn.Flatten()
        self.linear2 = nn.Linear(10, 1)
    def forward(self, src, target, src_mask = None, seq_mask = None):
        encinput = self.inputlinear(src)
        enc_output = self.encoder(encinput, src_mask)
        new_target = self.inputlinear2(target)
        dec_output = self.decoder(new_target, enc_output, seq_mask)
        output = self.linear(dec_output)
        output = self.linear2(self.flat(output))
        return output

def seq_mask(seq_len):
    return torch.tril(torch.ones(seq_len, seq_len))






