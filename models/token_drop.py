import torch
import torch.nn as nn

class TokenDrop(nn.Module):
    def __init__(self, prob=0.1, blank_token=1, eos_token=102):
        self.prob = prob
        self.eos_token = eos_token
        self.blank_token = blank_token

    def __call__(self, sample):
        mask = torch.bernoulli(self.prob*torch.ones_like(sample)).long()

        can_drop = (~(sample == self.eos_token)).long()
        mask = mask*can_drop

        mask[:, 0] = torch.zeros_like(mask[:, 0]).long()

        replace_with = (self.blank_token*torch.ones_like(sample)).long()

        sample_out = (1-mask)*sample + mask*replace_with

        return sample_out