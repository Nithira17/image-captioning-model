import torch
import torch.nn as nn
import random

def extract_patches(image_tensor, patch_size=16):
    bs, c, h, w = image_tensor.size()
    unfold = torch.nn.Unfold(kernel_size=patch_size, stride=patch_size)
    unfolded = unfold(image_tensor)
    unfolded = unfolded.transpose(1, 2).reshape(bs, -1, c*patch_size*patch_size)

    return unfolded

class SampleCaption(nn.Module):
    def __call__(self, sample):
        rand_index = random.randint(0, len(sample)-1)
        return sample[rand_index] 