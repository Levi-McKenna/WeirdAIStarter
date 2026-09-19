import math

import torch
from torch.utils.data import Dataset


class LyricsDataset(Dataset):
    def __init__(self, tokens, block_size, stride):
        self.tokens = tokens
        self.stride = stride
        self.block_size = block_size

    def __len__(self):
        return math.ceil((len(self.tokens) - self.block_size) / self.stride)

    def __getitem__(self, index):
        # TODO:
        # Get the input/output token sequences
        # Calculate x by grabbing the sublist of tokens starting at the index up to the block_size
        # Calculate y by grabbing the sublist of tokens starting at index + 1 up to block_size + 1
        start = index * self.stride
        return torch.tensor(self.tokens[start:start+self.block_size]),torch.tensor(self.tokens[start+1:start+self.block_size+1])
