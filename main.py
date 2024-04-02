import scipy
import torch
from torch.utils.data import Dataset
import os

class ImageDataset(Dataset):
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
    
class DNADataset(Dataset):
    def __init__(self, annotations_file=None, img_dir=None, transform=None, target_transform=None):
        data_mat = scipy.io.loadmat('data/INSECTS/data.mat')
        self.embeddings_dna = torch.from_numpy(data_mat['embeddings_dna']).float()
        self.labels = torch.from_numpy(data_mat['labels']).long()
        self.species = data_mat['species']
        self.ids = data_mat['ids']

    def __len__(self):
        return len(self.embeddings_dna)

    def __getitem__(self, idx):
        embedding = self.embeddings_dna[idx]
        label = self.labels[idx]
        return embedding, label

image_data = ImageDataset()
dna_data = DNADataset()