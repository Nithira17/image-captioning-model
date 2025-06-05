import torch
import torch.nn as nn
from models.utils import extract_patches

class VisionEncoder(nn.Module):
    def __init__(self, image_size, channels_in, patch_size=16, hidden_size=128, num_layers=3, num_heads=4):
        super(VisionEncoder, self).__init__()

        self.patch_size = patch_size
        self.fc_in = nn.Linear(channels_in*patch_size*patch_size, hidden_size)

        seq_length = (image_size//patch_size)**2
        self.pos_embedding = nn.Parameter(torch.empty(1, seq_length, hidden_size).normal_(std=0.02))

        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_size, nhead=num_heads, dim_feedforward=hidden_size*4, dropout=0.0, batch_first=True)
        self.encoder_layers = nn.TransformerEncoder(encoder_layer, num_layers)
        
    def forward(self, image):
        bs = image.shape[0]
        
        patch_seq = extract_patches(image, patch_size=self.patch_size)
        patch_emb = self.fc_in(patch_seq)
        
        embs = patch_emb + self.pos_embedding
        
        output = self.encoder_layers(embs)
        
        return output