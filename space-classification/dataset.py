import torch
from torch.utils.data import Dataset
import numpy as np
import re

class SliceDataset(Dataset):
    """Custom PyTorch Dataset that takes a file containing Slices.

        Args:
                args        : all the args.
                data        : the list of slices
                char_list   : a vocabulary of all the unique characters in slices
                block_size  : maximum length of slices by characters.
                prop_list   : the list of properties, whose length should be equal to that of char_list.
        Returns:
                A custom PyTorch dataset for training the Prior.
    """
    def __init__(self, args, data, char_list, block_size, prop_list = None):

        chars = sorted(list(set(char_list)))
        data_size, vocab_size = len(data), len(chars)
        prop_size = len(prop_list)

        print('data has %d slices, %d unique characters.' % (data_size, vocab_size))
    
        self.stoi = { ch:i for i,ch in enumerate(chars) }
        self.itos = { i:ch for i,ch in enumerate(chars) }
        self.max_len = block_size
        self.vocab_size = vocab_size
        self.data = data
        self.debug = args.debug
        self.prop_list = prop_list
    
    def __len__(self):
        if self.debug:
            return math.ceil(len(self.data) / (self.max_len + 1))
        else:
            return len(self.data)

    def __getitem__(self, idx):
        slices, prop = self.data[idx], self.prop_list[idx]
        slices = slices.strip().split(" ")

        # For classification tasks, we don't need special tokens
        slices = slices[:self.max_len]  # Truncate to max_len if necessary
        slices += [str('<')] * (self.max_len - len(slices))
        dix = [self.stoi[s] for s in slices]

        x = torch.tensor(dix, dtype=torch.long)
        propt = torch.tensor([int(prop)], dtype=torch.long)
        return x, propt
        
        
        
