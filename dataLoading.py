import numpy as np
import os
import cv2
import random
from tqdm import tqdm

dataDir ="insert folder path"
categories = ["WithoutMask", "ClothMask", "SurgicalMask", "N95Mask"]
imgSize = 255
training_data = []


def create_training_data():
    for category in categories:  # cycle through categories

        path = os.path.join(dataDir,category)  # create path to categories
        class_num = categories.index(category)  # get the classification by index per category

        for img in tqdm(os.listdir(path)):  # iterate over each image per category
            try:
                img_array = cv2.imread(os.path.join(path,img))  # convert to array
                new_array = cv2.resize(img_array, (imgSize, imgSize))  # resize to normalize data size
                training_data.append([new_array, class_num])  # add this to our training_data
            except Exception as e:
                pass


random.shuffle(training_data) # shuffle images in order to optimize the training.
