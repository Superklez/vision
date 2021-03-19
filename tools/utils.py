import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Used for training
import torch
import time
import copy

# Used for loading custom datasets
import cv2
from PIL import  Image
from skimage import io
from torch.utils.data import Dataset
from torchvision import transforms

# For reading DICOM files
import pydicom
from pydicom.pixel_data_handlers.util import apply_voi_lut

def train_model(model, criterion, optimizer, scheduler=None, epochs=10):
    '''
    Train model.

    Inputs:
    ----------
        model:
        criterion:
        optimizer:
        scheduler (optional):
        epochs (int): Number of epochs for training.

    Outputs:
    ----------
        model: Trained model of input model.

    Recommended use:
    ----------
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        losses = {'train':[], 'val':[]}
        accuracies = {'train':[], 'val':[]}

        dataloaders = {'train':train_loader, 'val':val_loader}
        dataset_sizes = {phase: len(dataloaders[phase].dataset) for phase in ['train', 'val']}

        model = Net()
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=5)

        model = train_model(model, criterion, optimizer, scheduler, epochs=20)
    '''
    start_time = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0

    for epoch in range(epochs):
        epoch += 1
        print('Epoch {}/{}'.format(epoch, epochs))
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()

            elif phase == 'val':
                model.eval()

            running_loss = 0
            running_corrects = 0

            for b, (inputs, labels) in enumerate(dataloaders[phase]):
                inputs = inputs.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                    _, preds = torch.max(outputs, 1)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            if phase == 'train':
                if scheduler:
                    scheduler.step(loss)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            losses[phase].append(epoch_loss)
            accuracies[phase].append(epoch_acc)

            print(f'{phase.capitalize()} Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc * 100:.2f}%')

            if phase == 'val' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - start_time

    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best Validation Acc: {best_acc * 100:.2f}')

    return model

###############################################################################
### LOADING CUSTOM DATASETS ###################################################
###############################################################################

class TorchvisionDataset(Dataset):
    '''
    Load custom image data using torchvision transformations.
    For image classification only.
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

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        image_name = os.path.join(self.root_dir, self.annotations.iloc[index, 0])
        image = io.imread(image_name)
        #label = torch.tensor(int(self.annotations.iloc[index, 1]))
        label = int(self.annotations.iloc[index, 1])

        if self.transform:
            image = self.trasnform(image)

        return image, label

class AlbumentationsDataset(Dataset):
    '''
    Load custom image data using albumentations transformations.
    For image classification only.
    '''
    def __init__(self, csv_file, root_dir, transform=None):
        self.annotations = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        image_name = os.path.join(self.root_dir, self.annotations.iloc[index, 0])
        image = cv2.imread(image_name)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        label = int(self.annotations.iloc[index, 1])

        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']

        return image, label

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

###############################################################################
### VISUALIZATION #############################################################
###############################################################################

def imshow(image, title=None, normalize=False):
    '''
    Imshow for Tensor. Ideal for displaying training images.

    Recommended use:
    ----------
        inputs, classes = next(iter(dataloaders['train']))
        out = torchvision.utils.make_grid(inputs)
        imshow(out, title=[class_names[x] for x in classes])
    '''
    image = image.numpy().transpose((1, 2, 0))

    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    if normalize:
        image = std * image + mean

    image = np.clip(image, 0, 1)

    plt.imshow(image)

    if title is not None:
        plt.title(title)

    plt.pause(0.001)

def visualize_model(model, num_images=6):
    '''
    Display predictions for few images.

    Inputs:
    ----------
        model:
        num_images (int): Number of images to display.

    Output:
    ----------
        Displayed images with predictions as labels.

    Recommended use:
    ----------
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        class_names = dataset['train'].classes

        visualize_model(model)
    '''
    was_training = model.training
    model.eval()
    images_so_far = 0
    fig = plt.figure()

    with torch.no_grad():
        for b, (inputs, labels) in enumerate(dataloaders['val']):
            #inputs = inputs.to(device)
            #labels = labels.to(device)
            optimizer.zero_grad()

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            for i in range(inputs.size()[0]):
                images_so_far += 1

                ax = plt.subplot(num_images // 2, 2, images_so_far)
                ax.axis('off')
                ax.set_title('Predicted: {}'.format(class_names[preds[i]]))

                imshow(inputs.data[i]) #inputs.cpu().data[i]

                if images_so_far == num_images:
                    model.train(mode=was_training)
                    return

        model.train(mode=was_training)

def show_landmarks(image, landmarks):
    '''
    Show image with landmarks.
    '''
    plt.imshow(image)
    plt.scatter(landmarks)
    plt.pause(0.001)

###############################################################################
### GET DATASET INFORMATION ###################################################
###############################################################################

def get_mean_std(dataloader):
    '''
    Determine the mean and standard deviation per channel.

    Inputs:
    ----------
        dataloader:

    Outputs:
    ----------
        mean (list): List of mean values for RGB channels.
        std (list): List of standard deviation values for RGB channels.
    '''

    # VAR[X] = E[X**2] - E[X]**2
    # STD[X] = sqrt(VAR[X])
    channels_sum = 0
    channels_squared_sum = 0
    num_batches = 0

    for data, _ in dataloader:
        channels_sum += torch.mean(data, dim=[0, 2, 3])
        channels_squared_sum += torch.mean(data**2, dim=[0, 2, 3])
        num_batches += 1

    mean = channels_sum / num_batches
    std = torch.sqrt(channels_squared_sum / num_batches - mean ** 2)

    return mean, std

def read_xray(path, voi_lut = True, fix_monochrome = True):
    dicom = pydicom.read_file(path)

    # VOI LUT (if available by DICOM device) is used to transform raw DICOM data to "human-friendly" view
    if voi_lut:
        data = apply_voi_lut(dicom.pixel_array, dicom)
    else:
        data = dicom.pixel_array

    # depending on this value, X-ray may look inverted - fix that:
    if fix_monochrome and dicom.PhotometricInterpretation == "MONOCHROME1":
        data = np.amax(data) - data

    data = data - np.min(data)
    data = data / np.max(data)
    data = (data * 255).astype(np.uint8)

    return data
