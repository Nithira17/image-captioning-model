import torch.nn as nn
from models.encoder import VisionEncoder
from models.decoder import Decoder

class VisionEncoderDecoder(nn.Module):
    def __init__(self, image_size, channels_in, num_emb, patch_size=16, hidden_size=128, num_layers=(3, 3), num_heads=4):
        super(VisionEncoderDecoder, self).__init__()

        self.encoder = VisionEncoder(image_size=image_size, channels_in=channels_in, patch_size=patch_size, hidden_size=hidden_size, num_layers=num_layers[0], num_heads=num_heads)
        self.decoder = Decoder(num_emb=num_emb, hidden_size=hidden_size, num_layers=num_layers[1], num_heads=num_heads)

    def forward(self, input_image, target_seq, padding_mask):
        bool_padding_mask = padding_mask ==0
        encoded_seq = self.encoder(image=input_image)
        decoded_seq = self.decoder(input_seq=target_seq, encoder_output=encoded_seq, input_padding_mask=bool_padding_mask)

        return decoded_seq