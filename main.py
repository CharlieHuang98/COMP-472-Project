import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import numpy as np
from torch import optim
import torch.nn.functional as F
import shutil
import os
import cv2
from tqdm import tqdm

batch_size = 64
num_classes = 10
learning_rate = 0.001
num_epochs = 20

# Device will determine whether to run the training on GPU or CPU.
use_cuda = torch.cuda.is_available()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if __name__ == '__main__':
    # Use transforms.compose method to reformat images for modeling,
    # and save to variable all_transforms for later use
    all_transforms = transforms.Compose([transforms.Resize((32, 32)), transforms.ToTensor(), transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])])
    dataDir = "insert folder path"
    categories = ["WithoutMask", "Clothmask", " Surgicalmask", "N95Mask"]
    imgSize = 100
    training_data = []
    test_data = []


    def save_ckp(state, is_best, checkpoint_path, best_model_path):
        f_path = checkpoint_path
        # save checkpoint data to the path given, checkpoint_path
        torch.save(state, f_path)
        # if it is a best model, min validation loss
        if is_best:
            best_fpath = best_model_path
            # copy that checkpoint file to best path given, best_model_path
            shutil.copyfile(f_path, best_fpath)

    def load_ckp(checkpoint_fpath, model, optimizer):
        # load check point
        checkpoint = torch.load(checkpoint_fpath)
        # initialize state_dict from checkpoint to model
        model.load_state_dict(checkpoint['state_dict'])
        # initialize optimizer from checkpoint to optimizer
        optimizer.load_state_dict(checkpoint['optimizer'])
        # initialize valid_loss_min from checkpoint to valid_loss_min
        valid_loss_min = checkpoint['valid_loss_min']
        # return model, optimizer, epoch value, min validation loss
        return model, optimizer, checkpoint['epoch'], valid_loss_min.item()

    def create_training_data():
        for category in categories:  # cycle through categories
            path = os.path.join(dataDir, category)  # create path to categories
            class_num = categories.index(category)  # get the classification by index per category
            for img in tqdm(os.listdir(path)):  # iterate over each image per category
                try:
                    img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)  # convert to array
                    new_array = cv2.resize(img_array, (imgSize, imgSize))  # resize to normalize data size
                    training_data.append([new_array, class_num])  # add this to our training_data
                except Exception as e:
                    pass

    train_loader = torch.utils.data.DataLoader(dataset=training_data, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_data, batch_size=batch_size, shuffle=True)

    # Creating a CNN class
    class ConvNeuralNet(nn.Module):
        #  Determine what layers and their order in CNN object
        def __init__(self, num_classes):
            super(ConvNeuralNet, self).__init__()
            self.conv_layer1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3)
            self.conv_layer2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3)
            self.max_pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.conv_layer3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3)
            self.conv_layer4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3)
            self.max_pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.fc1 = nn.Linear(1600, 128)
            self.relu1 = nn.ReLU()
            self.fc2 = nn.Linear(128, num_classes)

        # Progresses data across layers
        def forward(self, x):
            out = self.conv_layer1(x)
            out = self.conv_layer2(out)
            out = self.max_pool1(out)
            out = self.conv_layer3(out)
            out = self.conv_layer4(out)
            out = self.max_pool2(out)
            out = out.reshape(out.size(0), -1)
            out = self.fc1(out)
            out = self.relu1(out)
            out = self.fc2(out)
            return out


        def train(self, path_checkpoint, path_bestcheckpoint):
            model = ConvNeuralNet(num_classes)

            # Set Loss function with criterion
            criterion = nn.CrossEntropyLoss()

            # Set optimizer with optimizer
            optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=0.005, momentum=0.9)
            total_step = len(train_loader)

            # We use the pre-defined number of epochs to determine how many iterations to train the network on
            for epoch in range(num_epochs):
                # Load in the data in batches using the train_loader object
                for i, (images, labels) in enumerate(train_loader):
                    # Move tensors to the configured device
                    images = images.to(device)
                    labels = labels.to(device)

                    # Forward pass
                    outputs = model(images)
                    loss = criterion(outputs, labels)

                    # Backward and optimize
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch + 1, num_epochs, loss.item()))

                # create checkpoint variable and add important data
                checkpoint = {
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                }

                # save checkpoint
                save_ckp(checkpoint, False, path_checkpoint, path_bestcheckpoint)

            with torch.no_grad():
                correct = 0
                total = 0
                for images, labels in train_loader:
                    images = images.to(device)
                    labels = labels.to(device)
                    outputs = model(images)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

                print('Accuracy of the network on the {} train images: {} %'.format(50000, 100 * correct / total))


    trained_model = ConvNeuralNet.train("./checkpoint/current_checkpoint.pt", "./best_model/best_model.pt")




