import numpy as np
import h5py
from torch.utils.data import Dataset


class FERDataSet(Dataset):

    def __init__(self, mode='Training', transformer=None):
        self.mode = mode
        self.transformer = transformer
        self.data = h5py.File('./data/FER2013/data.h5', 'r', driver='core')
        if self.mode == 'Training':
            self.train

    def __getitem__(self, index):
        pass

    def __len__(self):
        pass
