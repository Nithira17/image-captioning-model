import torch
import torch.nn  as nn
import math

class SinusoidalPositionalEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim//2
        emb = math.log(10000)/(half_dim-1)
        emb = torch.exp(torch.arange(half_dim, device=device)*-emb)
        emb = x[:, None]*emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        
        return emb
    
class Decoder(nn.Module):
    def __init__(self, num_emb, hidden_size=128, num_layers=3, num_heads=4):
        super(Decoder, self).__init__()

        self.embedding = nn.Embedding(num_emb, hidden_size)
        self.embedding.weight.data = 0.001*self.embedding.weight.data

        self.pos_emb = SinusoidalPositionalEmbedding(hidden_size)

        decoder_layer = nn.TransformerDecoderLayer(d_model=hidden_size, nhead=num_heads, dim_feedforward=hidden_size*4, dropout=0.0, batch_first=True)

        self.decoder_layers = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        self.fc_out = nn.Linear(hidden_size, num_emb)

    def forward(self, input_seq, encoder_output, input_padding_mask=None, encoder_padding_mask=None):
        input_embs = self.embedding(input_seq)
        bs, l, h = input_embs.shape

        seq_index = torch.arange(l, device=input_seq.device)
        pos_emb = self.pos_emb(seq_index).reshape(1, l, h).expand(bs, l, h)
        embs = input_embs + pos_emb
        causal_mask = torch.triu(torch.ones(l, l, device=input_seq.device), 1).bool()

        output = self.decoder_layers(tgt=embs, memory=encoder_output, tgt_mask=causal_mask, tgt_key_padding_mask=input_padding_mask, memory_key_padding_mask=encoder_padding_mask)

        return self.fc_out(output)