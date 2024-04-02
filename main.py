import scipy
import torch
from torch.utils.data import Dataset
import os

class Image_Dataset(Dataset):
    def __init__(self, annotations_file=None, img_dir=None, transform=None, target_transform=None):
        data_mat = scipy.io.loadmat('data/INSECTS/data.mat')
        self.embeddings_img = torch.from_numpy(data_mat['embeddings_img']).float()
        self.labels = torch.from_numpy(data_mat['labels']).long()
        self.species = data_mat['species']
        self.ids = data_mat['ids']

    def __len__(self):
        return len(self.embeddings_img)

    def __getitem__(self, idx):
        embedding = self.embeddings_img[idx]
        label = self.labels[idx]
        return embedding, label

data = Image_Dataset()
for i in range(10):
    print(data[i])
