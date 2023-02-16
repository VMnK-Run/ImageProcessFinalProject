import numpy as np
import h5py
from torch.utils.data import Dataset
from PIL import Image


class FERDataSet(Dataset):

    def __init__(self, mode='Training', transform=None):
        self.mode = mode
        self.transform = transform
        self.datas = h5py.File('./data/FER2013/data.h5', 'r', driver='core')
        if self.mode == 'Training':
            self.features = self.datas['Training_features']
            self.labels = self.datas['Training_labels']
        if self.mode == 'PublicTest':
            self.features = self.datas['PublicTest_features']
            self.labels = self.datas['PublicTest_labels']
        if self.mode == 'PrivateTest':
            self.features = self.datas['PrivateTest_features']
            self.labels = self.datas['PrivateTest_labels']
        num = len(self.features)
        self.features = np.asarray(self.features).reshape((num, 48, 48))

    def __getitem__(self, index):
        feature, label = self.features[index], self.labels[index]
        # 转成 PIL Image
        img = feature[:, :, np.newaxis]
        img = np.concatenate((img, img, img), axis=2)
        img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.features)
