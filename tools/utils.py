import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

from skimage import io
from torch.utils.data import Dataset, random_split

class CustomImageDataset(Dataset):
    '''
    Custom Image Dataset
    '''
    def __init__(self, csv_file, root_dir, transform=None):
        '''
        Inputs:
        ---------
            csv_file (str): Path to the csv file with annotations.
            root_dir (str): Directory to images.
            transform (callable, optional): Optional transformation to data.
        '''
        self.annotations = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __lenl__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        image_name = os.path.join(self.root_dir, self.annotations.iloc[index, 0])
        image = io.imread(image_path)
        label = torch.tensor(int(self.annotations.iloc[index, 1]))

        if self.transform:
            image = self.trasnform(image)

        return (image, label)

def show_landmarks(image, landmarks):
    '''
    Show image with landmarks.
    '''
    plt.imshow(image)
    plt.scatter(landmarks)
    plt.pause(0.001)

class LandmarksDataset(Dataset):
    '''
    [Face] Landmarks Dataset
    '''
    def __init__(self, csv_file, root_dir, transform=None):
        '''
        Inputs:
        ---------
            csv_file (str): Path to the csv file with annotations.
            root_dir (str): Directory to images.
            transform (callable, optional): Optional transformation to data.
        '''
        self.landmarks = pd.read_csv(csf_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.landmarks)

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()

        image_name = os.path.join(self.root_dir, self.landmarks.iloc[index, 0])
        image = io.imread(image_name)
        landmarks = self.landmarks.iloc[index, 1:]
        landmarks = np.array([landmarks])
        landmarks = landmarks.astype('float').reshape(-1, 2)

        sample = {'image':image, 'landmarks':landmarks}
        if self.transform:
            sample = self.transform(sample)

        return sample
