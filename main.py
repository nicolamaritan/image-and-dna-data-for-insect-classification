import scipy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

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

        # Remap train labels in [0, 796]
        train_labels = data_mat["labels"][trainval_loc][0]
        train_labels_mapping = {label: i for i, label in enumerate(np.unique(train_labels))}
        train_labels_remapped = np.array([train_labels_mapping[label.item()] for label in train_labels])

        # Remap test labels in [797, 1039]
        test_unseen_labels = data_mat["labels"][test_unseen_loc][0]
        test_unseen_labels_mapping = {label: i + 797 for i, label in enumerate(np.unique(test_unseen_labels))}
        test_unseen_labels_remapped = np.array([test_unseen_labels_mapping[label.item()] for label in test_unseen_labels])

        assert np.intersect1d(train_labels, test_unseen_labels).size == 0

        # Union of the two mappings, allows to full remap all the labels
        labels_mapping = train_labels_mapping | test_unseen_labels_mapping
        assert len(labels_mapping) == 1040

        labels = data_mat["labels"][indeces]  # Consider only train/test labels
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

        if (not train):
            # Compute genera of unseen species
            unseen_species_genera = []
            for i in test_unseen_loc[0]:
                unseen_species_genera.append(data_mat["G"][data_mat["labels"][i][0] - 1][0] - 1041)

            self.unseen_species_genera = np.array(unseen_species_genera)
            assert len(np.unique(self.unseen_species_genera)) == 134

            # Compute unseen species
            seen_species = []
            for i in test_seen_loc[0]:
                seen_species.append(labels_mapping[data_mat["labels"][i].item()])
            self.seen_species = np.array(seen_species)
            assert len(np.unique(self.seen_species)) == 770

        self.species = data_mat["species"][indeces]
        self.ids = data_mat["ids"][indeces]

    def __len__(self):
        return len(self.embeddings_dna)

    def __getitem__(self, idx):
        embedding_img = self.embeddings_img[idx]
        embedding_dna = self.embeddings_dna[idx]
        embedding = torch.cat(
            (self.embeddings_img[idx], self.embeddings_dna[idx])
        ).view(1, 1, -1)
        label = self.remapped_labels[idx].item()
        genera = self.genera[idx].item()

        return embedding_img.view(1, -1), embedding_dna.view(1, -1), label, genera


class CrossNet(nn.Module):
    def __init__(self):
        super(CrossNet, self).__init__()
        # Pre-core network
        # Image embedding dimensionality reduction
        self.img_fc1 = nn.Linear(2048, 1024)
        self.img_fc2 = nn.Linear(1024, 500)

        # Separate processing pipelines
        self.img_resblock1 = ResidualBlock1d(1, 4)
        self.img_resblock2 = ResidualBlock1d(4, 4)

        self.dna_resblock1 = ResidualBlock1d(1, 4)
        self.dna_resblock2 = ResidualBlock1d(4, 4)

        self.resblock1 = ResidualBlock1d(4, 4)
        self.resblock2 = ResidualBlock1d(4, 4)
        self.resblock3 = ResidualBlock1d(4, 4)
        self.resblock4 = ResidualBlock1d(4, 4)

        # Fully connected layers for classification
        self.fc_species_1 = nn.Linear(4*2548, 2048)
        self.fc_species_2 = nn.Linear(2048, 797)

        self.fc_genera_1 = nn.Linear(4*2548, 2048)
        self.fc_genera_2 = nn.Linear(2048, 368)

        # Dropout layers for regularization
        self.conv_dropout = nn.Dropout(0.2)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x_img, x_dna):
        # Reduce dimensionality of image embeddings
        #x_img = F.relu(self.img_fc1(x_img))
        #x_img = F.relu(self.img_fc2(x_img))

        x_img = self.img_resblock1(x_img)
        x_img = self.img_resblock2(x_img)

        x_dna = self.dna_resblock1(x_dna)
        x_dna = self.dna_resblock2(x_dna)

        x = torch.cat((x_img, x_dna), axis=2)

        # CrossNet core
        x = F.relu(self.resblock1(x))
        x = self.conv_dropout(F.relu(self.resblock2(x)))
        x = F.relu(self.resblock3(x))
        x = self.conv_dropout(F.relu(self.resblock4(x)))

        x = x.view(x.shape[0], 4*2548)
        #x = self.dropout(F.relu(self.fc1(x)))

        x_species = x.clone()
        x_genera = x.clone()

        # Dropout for regularization
        x_species = self.dropout(F.relu(self.fc_species_1(x_species)))
        x_species = self.fc_species_2(x_species)

        x_genera = self.dropout(F.relu(self.fc_genera_1(x_genera)))
        x_genera = self.fc_genera_2(x_genera)

        return x_species, x_genera

class ResidualBlock1d(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock1d, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm1d(out_channels)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += identity
        return out

# Test the network with a random input tensor
model = CrossNet()
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


inputs_img, inputs_dna, labels, genera = next(iter(training_loader))
print(f"Training input batch: {inputs_img.shape}, {inputs_dna.shape}")
print(f"Training label batch: {labels.shape}")
print(f"Training genera batch: {genera.shape}")
'''
inputs_test, labels_test, genera_test = next(iter(test_loader))
print(f"Test input batch: {inputs_test.shape}")
print(f"Test label batch: {labels_test.shape}")
print(f"Test genera batch: {genera.shape}")
'''

# Report split sizes
print("Training set has {} instances".format(len(training_set)))
print("Test set has {} instances".format(len(test_set)))

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
epochs = 5

for epoch in range(epochs):  # loop over the dataset multiple times

    running_labels_loss = 0.0
    running_genera_loss = 0.0
    for i, data in enumerate(training_loader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs_img, inputs_dna, labels, genera = data
        inputs_img, inputs_dna, labels, genera = inputs_img.to(device), inputs_dna.to(device), labels.to(device), genera.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        labels_outputs, genera_outputs = model(inputs_img, inputs_dna)
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
torch.save(model.state_dict(), "./model")
exit(0)

correct_genera = 0
total_genera = 0
correct_labels = 0
total_labels = 0

# Initialize lists to store accuracies
species_accuracies = []
genus_accuracies = []

# Define the range of thresholds to test
thresholds = np.linspace(0, 1, 2)  # Adjust the range and number of thresholds as needed

# Assuming `model`, `test_loader`, and `device` are already defined
with torch.no_grad():
    for threshold in thresholds:
        correct_labels = 0
        total_labels = 0
        correct_genera = 0
        total_genera = 0

        for data in test_loader:
            inputs_img, inputs_dna, labels, genera = data
            inputs_img, inputs_dna, labels, genera = inputs_img.to(device), inputs_dna.to(device), labels.to(device), genera.to(device)

            labels_outputs, genera_outputs = model(inputs_img, inputs_dna)

            labels_outputs = nn.Softmax(dim=1)(labels_outputs)
            genera_outputs = nn.Softmax(dim=1)(genera_outputs)

            predicted_labels_values, predicted_labels = torch.topk(labels_outputs.data, k=2, dim=1)
            _, predicted_genera = torch.max(genera_outputs.data, 1)

            differences = predicted_labels_values[:, 0] - predicted_labels_values[:, 1]
            genera_mask = differences <= threshold
            labels_mask = ~genera_mask

            correct_genera += (predicted_genera[genera_mask] == genera[genera_mask]).sum().item()
            total_genera += genera_mask.sum().item()

            correct_labels += (predicted_labels[labels_mask][:, 0] == labels[labels_mask]).sum().item()
            total_labels += labels_mask.sum().item()

        # Compute accuracies
        species_accuracy = correct_labels / total_labels if total_labels > 0 else 0
        genus_accuracy = correct_genera / total_genera if total_genera > 0 else 0

        print("-------------------------------------------------------------------------------")
        print(f"threshold: {threshold}")
        print(f"Genera: Accuracy of the network on the {len(test_set)} test inputs with total_genera={total_genera}: {genus_accuracy}")
        print(f"Species: Accuracy of the network on the {len(test_set)} test inputs with total_labels={total_labels}: {species_accuracy}")
        print("-------------------------------------------------------------------------------")

        species_accuracies.append(species_accuracy)
        genus_accuracies.append(genus_accuracy)

# Plotting the curve
plt.plot(species_accuracies, genus_accuracies, marker='o')
plt.xlabel('Species Accuracy')
plt.ylabel('Genus Accuracy')
plt.title('Curve: Species Accuracy vs Genus Accuracy')
plt.grid(True)
plt.show()