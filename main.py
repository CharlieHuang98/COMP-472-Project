import os
import random
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from matplotlib import pyplot as plt
from tqdm import tqdm

# Hyper-parameters
num_epochs = 3
batch_size = 64
learning_rate = 0.001


# Device will determine whether to run the training on GPU or CPU.
use_cuda = torch.cuda.is_available()
# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if __name__ == '__main__':
    # Use transforms.compose method to reformat images for modeling and save to variable all_transforms for later use dataset has PILImage images of range [0, 1].  We transform them to Tensors of normalized range [-1, 1]
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    #give paths to train and test datasets
    ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
    test_dataset = torchvision.datasets.ImageFolder(root=ROOT_DIR + "/Face_Mask_Dataset/Test/", transform=transform)
    classes = ('WithoutMask', 'Clothmask', 'Surgicalmask', 'N95Mask')
    imgSize = 255
    train_data = []
    test_data = []


    def imshow(img):
        img = img / 2 + 0.5  # unnormalize
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        plt.show()

    def create_training_data():
        counter = 0
        rnd = random.randrange(0, 1001)
        for category in classes:  # cycle through categories
            path = os.path.join(ROOT_DIR + "/Face_Mask_Dataset/Train/", category)  # create path to categories
            class_num = classes.index(category)  # get the classification by index per category
            for img in tqdm(os.listdir(path)):  # iterate over each image per category
                try:
                    img_array = cv2.imread(os.path.join(path, img))  # convert to array
                    new_array = cv2.resize(img_array, (imgSize, imgSize)) # resize to normalize data size
                    train_data.append([new_array, class_num])  # add this to our training_data
                    counter += 1
                    if counter == rnd:
                        plt.imshow(new_array, cmap='gray')  # graph it
                        plt.show()

                except Exception as e:
                    pass


    create_training_data()
    train_dataset = train_data
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


    class CNN(nn.Module):
        def __init__(self):
            super(CNN, self).__init__()
            self.conv1 = nn.Conv2d(3, 6, 3)
            self.pool = nn.MaxPool2d(2, 2)
            self.conv2 = nn.Conv2d(6, 16, 5)
            self.fc1 = nn.Linear(59536, 120)
            self.fc2 = nn.Linear(120, 84)
            self.fc3 = nn.Linear(84, 10)

        def forward(self, x):
            x = x.permute(0, 3, 1, 2)
            x = self.pool(F.relu(self.conv1(x)))
            x = self.pool(F.relu(self.conv2(x)))
            x = x.view(x.size(0), -1)
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.fc3(x)
            return x

    model = CNN()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    optimizer.param_groups
    criterion = nn.CrossEntropyLoss()
    n_total_steps = len(train_loader)

    for epoch in range(num_epochs):

        for i, (images, labels) in enumerate(train_loader):
            # origin shape: [6, 3, 5, 5] = 6, 3, 25
            # input_layer: 3 input channels, 6 output channels, 5 kernel size
            images = images.to(device)
            labels = labels.to(device)
            # Forward pass
            images = images.float()
            outputs = model(images)
            loss = criterion(outputs, labels)
            # Backward and optimize
            optimizer.zero_grad()
            torch.autograd.set_detect_anomaly(True)
            loss.backward(retain_graph=True)
            optimizer.step()

            if (i + 1) % 2000 == 0:
                print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{n_total_steps}], Loss: {loss.item():.4f}')

    print('Finished Training')
    PATH = './cnn.pth'
    torch.save(model.state_dict(), PATH)

    with torch.no_grad():
        n_correct = 0
        n_samples = 0
        n_class_correct = [0 for i in range(10)]
        n_class_samples = [0 for i in range(10)]

        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            # max returns (value ,index)
            _, predicted = torch.max(outputs, 1)
            n_samples += labels.size(0)
            n_correct += (predicted == labels).sum().item()

            for i in range(batch_size):
                label = labels[i]
                pred = predicted[i]

                if (label == pred):
                    n_class_correct[label] += 1
                n_class_samples[label] += 1

        acc = 100.0 * n_correct / n_samples
        print(f'Accuracy of the network: {acc} %')

        for i in range(10):
            acc = 100.0 * n_class_correct[i] / n_class_samples[i]
            print(f'Accuracy of {classes[i]}: {acc} %')


