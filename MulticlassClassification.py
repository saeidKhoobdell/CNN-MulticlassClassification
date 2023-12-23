#%% packages
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import os
from typing import OrderedDict
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix
os.getcwd()

#%%
imgTest = plt.imread('train/akita/akita_0.jpg')
imgTest.shape
#%%
# %% transform and load data
transform = transforms.Compose([
    transforms.Resize((50,50)),
    transforms.RandomRotation(50),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize((0.5),(0.5))
]
)
#%%
# TODO: set up image transforms

# TODO: set up train and test datasets
trainset = torchvision.datasets.ImageFolder(root='train', transform=transform)
testset = torchvision.datasets.ImageFolder(root='test', transform=transform)

#%%
# TODO: set up data loaders
trainloader = DataLoader(trainset,batch_size=4,shuffle=True)
testloader = DataLoader(testset,batch_size=4,shuffle=True)
# %%


#%%
CLASSES = ['affenpinscher', 'akita', 'corgi']
NUM_CLASSES = len(CLASSES)

# TODO: set up model class
class ImageMulticlassClassificationNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1,6,3) #out 6 48
        self.pool = nn.MaxPool2d(2,2)  #out 6 24    
        self.conv2 = nn.Conv2d(6,16,3)  #out 16 22 and pool 16 11
        self.fc1 = nn.Linear(16 * 11 * 11, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64,3)
        self.sofmax = nn.LogSoftmax(dim=1)
        self.relu = nn.ReLU()
    def forward(self, X):
        x = self.conv1(X)
        x = self.relu(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)
        print(x.size())
        x = torch.flatten(x,1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.sofmax(x)
        return x

# input = torch.rand(1, 1, 50, 50) # BS, C, H, W
model = ImageMulticlassClassificationNet()      
# model(input).shape

# %% loss function and optimizer
# TODO: set up loss function and optimizer
# loss_fn = ...
criterion=  nn.CrossEntropyLoss()
# optimizer = ...
optimizer = torch.optim.Adam(model.parameters())

#%%
# %% training
NUM_EPOCHS = 10
for epoch in range(NUM_EPOCHS):
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        # TODO: define training loop
        optimizer.zero_grad()

        output = model(inputs)

        loss = criterion(output, labels)
        loss.backward()

        optimizer.step()

    print(f'Epoch {epoch}/{NUM_EPOCHS}, Loss: {loss.item():.4f}')


# %% test
y_test = []
y_test_hat = []
for i, data in enumerate(testloader, 0):
    inputs, y_test_temp = data
    with torch.no_grad():
        y_test_hat_temp = model(inputs).round()
    
    y_test.extend(y_test_temp.numpy())
    y_test_hat.extend(y_test_hat_temp.numpy())

# %%
acc = accuracy_score(y_test, np.argmax(y_test_hat, axis=1))
print(f'Accuracy: {acc*100:.2f} %')
# %% confusion matrix
confusion_matrix(y_test, np.argmax(y_test_hat, axis=1))
# %%
