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
        train_loc = splits_mat["train_loc"]
        trainval_loc = splits_mat["trainval_loc"]
        test_seen_loc = splits_mat["test_seen_loc"]
        test_unseen_loc = splits_mat["test_unseen_loc"]
        val_seen_loc = splits_mat["val_seen_loc"]
        val_unseen_loc = splits_mat["val_unseen_loc"]

        indeces = (
            trainval_loc
            if train
            else np.concatenate((test_seen_loc, test_unseen_loc), axis=1)
        )
        # indeces.shape is (1, |indeces|), so we extract the whole list using [0]
        indeces = indeces[0]
        # Matlab indeces starts from 1
        indeces -= 1

        data_mat = scipy.io.loadmat("data/INSECTS/data.mat")
        self.embeddings_img = torch.from_numpy(
            data_mat["embeddings_img"][indeces]
        ).float()
        self.embeddings_dna = torch.from_numpy(
            data_mat["embeddings_dna"][indeces]
        ).float()
        self.labels = torch.from_numpy(data_mat["labels"][indeces]).long()
        # data_mat['G'] returns a ndarray of type uint16, therefore we convert into int16 before invoking from_numpy
        self.G = torch.from_numpy(data_mat["G"].astype(np.int16)).long()
        self.genera = torch.empty(self.labels.shape)
        for i in range(indeces.size):
            self.genera[i][0] = self.G[self.labels[i][0] - 1][0] - 1040

        self.species = data_mat["species"][indeces]
        self.ids = data_mat["ids"][indeces]

    def __len__(self):
        return len(self.embeddings_dna)

    def __getitem__(self, idx):
        embedding = torch.cat(
            (self.embeddings_img[idx], self.embeddings_dna[idx])
        ).view(1, 1, -1)
        label = self.labels[idx].item()

        dwt = DWT1DForward(wave="db6", J=1)
        yl, yh = dwt(embedding)
        embedding = torch.cat((yl, yh[0]), dim=0).squeeze(1)

        return embedding, label


image_dna_data = ImageDNADataset()


class Wavelet1DCNN(nn.Module):
    def __init__(self):
        super(Wavelet1DCNN, self).__init__()
        self.conv1 = nn.Conv1d(2, 16, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(32 * 319, 128)
        self.fc2 = nn.Linear(128, 1041)

    def forward(self, x):
        # Convolutional layers
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)

        # Flatten
        x = x.view(-1, 32 * 319)

        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# Test the network with a random input tensor
model = Wavelet1DCNN()
model.to(device)
print(f"Input shape: {image_dna_data[0][0].shape}")

training_set, test_set = torch.utils.data.random_split(image_dna_data, [0.8, 0.2])

training_set = ImageDNADataset(train=True)
test_set = ImageDNADataset(train=False)

batch_size = 32
training_loader = torch.utils.data.DataLoader(
    training_set, batch_size=batch_size, shuffle=True
)
test_loader = torch.utils.data.DataLoader(
    test_set, batch_size=batch_size, shuffle=False
)

inputs, labels = next(iter(training_loader))
print(f"Training input batch: {inputs.shape}")
print(f"Training label batch: {labels.shape}")
inputs_test, labels_test = next(iter(test_loader))
print(f"Test input batch: {inputs_test.shape}")
print(f"Test label batch: {labels_test.shape}")

# Report split sizes
print("Training set has {} instances".format(len(training_set)))
print("Test set has {} instances".format(len(test_set)))

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
epochs = 10

for epoch in range(epochs):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(training_loader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        print_step = 500
        if i % print_step == print_step - 1:
            print(f"[{epoch + 1}, {i + 1:5d}] loss: {running_loss / print_step:.3f}")
            running_loss = 0.0

print("Finished Training")

correct = 0
total = 0
# since we're not training, we don't need to calculate the gradients for our outputs
with torch.no_grad():
    for data in test_loader:
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        # calculate outputs by running images through the network
        outputs = model(inputs)
        # the class with the highest energy is what we choose as prediction
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"Accuracy of the network on the 10000 test inputs: {correct / total}")
