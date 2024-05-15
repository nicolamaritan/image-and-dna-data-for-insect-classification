import scipy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from pytorch_wavelets import DWT1DForward
import numpy as np
import os

torch.manual_seed(0)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

class ImageDNADataset(Dataset):
    def __init__(self, train=True):
        splits_mat = scipy.io.loadmat("data/INSECTS/splits.mat")
        train_loc = splits_mat["train_loc"]-1
        trainval_loc = splits_mat["trainval_loc"]-1
        test_seen_loc = splits_mat["test_seen_loc"]-1
        test_unseen_loc = splits_mat["test_unseen_loc"]-1
        val_seen_loc = splits_mat["val_seen_loc"]-1
        val_unseen_loc = splits_mat["val_unseen_loc"]-1
        
        assert len(trainval_loc[0] == 19420)
        assert len(test_seen_loc[0] == 4965)
        assert len(test_unseen_loc[0] == 8463)

        indeces = (
            trainval_loc
            if train
            else np.concatenate((test_seen_loc, test_unseen_loc), axis=1)
        )
        # indeces.shape is (1, |indeces|), so we extract the whole list using [0]
        indeces = indeces[0]

        data_mat = scipy.io.loadmat("data/INSECTS/data.mat")
        self.embeddings_img = torch.from_numpy(
            data_mat["embeddings_img"][indeces]
        ).float()
        self.embeddings_dna = torch.from_numpy(
            data_mat["embeddings_dna"][indeces]
        ).float()


        train_labels = data_mat["labels"][trainval_loc][0]
        train_labels_mapping = {label: i for i, label in enumerate(np.unique(train_labels))}
        train_labels_remapped = np.array([train_labels_mapping[label.item()] for label in train_labels])

        test_unseen_labels = data_mat["labels"][test_unseen_loc][0]
        test_unseen_labels_mapping = {label: i + 797 for i, label in enumerate(np.unique(test_unseen_labels))}
        test_unseen_labels_remapped = np.array([test_unseen_labels_mapping[label.item()] for label in test_unseen_labels])
        #test_unseen_labels_remapped += 797

        assert np.intersect1d(train_labels, test_unseen_labels).size == 0

        labels_mapping = train_labels_mapping | test_unseen_labels_mapping
        #print(labels_mapping)
        assert len(labels_mapping) == 1040

        print(np.unique(train_labels_remapped))
        print("culone")
        print(np.unique(test_unseen_labels_remapped))

        labels = data_mat["labels"][indeces]
        remapped_labels = np.array([labels_mapping[label.item()] for label in labels])

        self.remapped_labels = torch.from_numpy(remapped_labels).long()
        self.labels = torch.from_numpy(labels).long()

        if train:
            assert len(torch.unique(self.remapped_labels)) == 797
        else:
            assert len(torch.unique(self.remapped_labels)) == 1013

        # data_mat['G'] returns a ndarray of type uint16, therefore we convert into int16 before invoking from_numpy
        self.G = torch.from_numpy(data_mat["G"].astype(np.int16)).long()
        self.genera = torch.empty(self.labels.shape).long()
        for i in range(indeces.size):
            self.genera[i][0] = self.G[self.labels[i][0] - 1][0] - 1041

        if train:
            assert len(self.genera) == 19420
        else:
            assert len(self.genera) == 13428

        self.species = data_mat["species"][indeces]
        self.ids = data_mat["ids"][indeces]

    def __len__(self):
        return len(self.embeddings_dna)

    def __getitem__(self, idx):
        embedding = torch.cat(
            (self.embeddings_img[idx], self.embeddings_dna[idx])
        ).view(1, 1, -1)
        label = self.remapped_labels[idx].item()
        genera = self.genera[idx].item()

        dwt = DWT1DForward(wave="db6", J=1)
        yl, yh = dwt(embedding)
        embedding = torch.cat((yl, yh[0]), dim=0).squeeze(1)

        return embedding, label, genera


class Wavelet1DCNN(nn.Module):
    def __init__(self):
        super(Wavelet1DCNN, self).__init__()
        self.conv1 = nn.Conv1d(2, 16, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=3, stride=1, padding=1)
        self.fc11 = nn.Linear(32 * 319, 128)
        self.fc21 = nn.Linear(128, 797)
        self.fc12 = nn.Linear(32 * 319, 128)
        self.fc22 = nn.Linear(128, 368)

    def forward(self, x):
        # Convolutional layers
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)

        # Flatten
        x = x.view(-1, 32 * 319)
        x_species = x.clone()
        x_genera = x.clone()

        x_species = F.relu(self.fc11(x_species))
        x_species = F.relu(self.fc21(x_species))

        x_genera = F.relu(self.fc12(x_genera))
        x_genera = F.relu(self.fc22(x_genera))

        return x_species, x_genera


# Test the network with a random input tensor
model = Wavelet1DCNN()
model.to(device)
#print(f"Input shape: {image_dna_data[0][0].shape}")

training_set = ImageDNADataset(train=True)
test_set = ImageDNADataset(train=False)

batch_size = 32
training_loader = torch.utils.data.DataLoader(
    training_set, batch_size=batch_size, shuffle=True
)
test_loader = torch.utils.data.DataLoader(
    test_set, batch_size=batch_size, shuffle=False
)

inputs, labels, genera = next(iter(training_loader))
print(f"Training input batch: {inputs.shape}")
print(f"Training label batch: {labels.shape}")
print(f"Training genera batch: {genera.shape}")
inputs_test, labels_test, genera_test = next(iter(test_loader))
print(f"Test input batch: {inputs_test.shape}")
print(f"Test label batch: {labels_test.shape}")
print(f"Test genera batch: {genera.shape}")

# Report split sizes
print("Training set has {} instances".format(len(training_set)))
print("Test set has {} instances".format(len(test_set)))

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
epochs = 20


for epoch in range(epochs):  # loop over the dataset multiple times

    running_labels_loss = 0.0
    running_genera_loss = 0.0
    for i, data in enumerate(training_loader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels, genera = data
        inputs, labels, genera = inputs.to(device), labels.to(device), genera.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        labels_outputs, genera_outputs = model(inputs)
        labels_loss = criterion(labels_outputs, labels)
        genera_loss = criterion(genera_outputs, genera)
        total_loss = labels_loss + genera_loss
        total_loss.backward()
        optimizer.step()

        # print statistics
        running_labels_loss += labels_loss.item()
        running_genera_loss += genera_loss.item()
        print_step = 500
        if i % print_step == print_step - 1:
            print(f"[{epoch + 1}, {i + 1:5d}] labels loss: {running_labels_loss / print_step:.3f}")
            print(f"[{epoch + 1}, {i + 1:5d}] genera loss: {running_genera_loss / print_step:.3f}")
            running_labels_loss = 0.0
            running_genera_loss = 0.0

print("Finished Training")

correct_genera = 0
total_genera = 0
correct_labels = 0
total_labels = 0

with torch.no_grad():
    for data in test_loader:
        inputs, labels, genera = data
        inputs, labels, genera = inputs.to(device), labels.to(device), genera.to(device)
        
        labels_outputs, genera_outputs = model(inputs)
        
        _, predicted_genera = torch.max(genera_outputs.data, 1)
        _, predicted_labels = torch.max(labels_outputs.data, 1)
        
        total_genera += genera.size(0)
        total_labels += labels.size(0)
        
        correct_genera += (predicted_genera == genera).sum().item()
        correct_labels += (predicted_labels == labels).sum().item()


print(f"Genera: Accuracy of the network on the {len(test_set)} test inputs: {correct_genera / total_genera}")
print(f"Species: Accuracy of the network on the {len(test_set)} test inputs: {correct_labels / total_labels}")
