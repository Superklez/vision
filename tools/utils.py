import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

from PIL import  Image
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

def plot_image(image, boxes):

    im = np.array(image)
    height, width, _ = im.shape

    fig, ax = plt.subplots()
    ax.imshow(im)

    for box in boxes:
        box = box[2:]
        assert len(box) == 4, 'box must only contain x, y, w, h dimensions'

        x0 = box[0] - box[2] / 2
        y0 = box[1] - box[3] / 2

        rect = patches.Rectangle(
            (x0 * width, y0 * height),
            box[2] * width,
            box[3] * height,
            linewidth = 1,
            edgecolor = 'r',
            facecolor = 'none'
        )

        ax.add_patch(rect)

    plt.show()

def get_bboxes(model, loader, prob_threshold=0.2, iou_threshold=0.4, bbox_format='midpoint'):

    all_pred_boxes = []
    all_true_boxes = []

    model.eval()
    train_idx = 0

    for b, data in enumerate(loader):
        images, labels = data
        #labels = labels.to(device)

        with torch.no_grad():
            predictions = model(images)

        batch_size = images.shape[0]
        pred_bboxes = cellboxes_to_boxes(predictions)
        true_bboxes = cellboxes_to_boxes(labels)

        for idx in range(batch_size):
            nms_boxes = non_max_suppression(
                pred_bboxes[idx], prob_threshold, iou_threshold, bbox_format
            )

            for nms_box in nms_boxes:
                all_pred_boxes.append([train_idx] + nms_box)

            for box in true_bboxes[idx]:
                if box[1] > prob_threshold:
                    all_true_boxes.append([train_idx] + box)

            train_idx += 1

    model.train()
    return all_pred_boxes, all_true_boxes

def convert_cellboxes(predictions, S=7, C=20):

    batch_size = predictions.shape[0]
    bboxes1 = predictions[..., C+1:C+5]
    bboxes2 = predictions[..., C+6:C+10]

    scores = torch.cat(
        (predictions[..., C].unsqueeze(0), predictions[..., C+5].unsqueeze(0)), dim=0
    )

    best_bbox = scores.argmax(0).unsqueeze(-1)
    best_bboxes = torch.mul(bboxes1, (1 - best_bbox)) + torch.mul(bboxes2, best_bbox)

    cell_indices = torch.arange(S).repeat(batch_size, S, 1).unsqueeze(-1)

    x = 1 / S * (best_bboxes[..., :1] + cell_indices)
    y = 1 / S * (best_bboxes[..., 1:2] + cell_indices.permute(0, 2, 1, 3))
    wy = 1 / S * best_bboes[..., 2:4]

    converted_bboxes = torch.cat((x, y, wy), dim=-1)
    predicted_class = predictions[..., :C].argmax(-1).unsqueeze(-1)
    best_confidence = torch.max(predictions[..., C], predictions[..., C+5]).unsqueeze(-1)
    converted_preds = torch.cat((predicted_class, best_confidence, converted_bboxes), dim=-1)

    return converted_preds

def cellboxes_to_boxes(out, S=7):

    converted_pred = convert_cellboxes(out).reshape(out.shape[0], S * S, -1)
    converted_pred[..., 0] = converted_pred[..., 0].long()
    all_bboxes = []

    for ex_idx in range(out.shape[0]):
        bboxes = []

        for bbox_idx in range(S * S):
            bboxes.append([x.item() for x in converted_pred[ex_idx, bbox_idx, :]])
        all_bboxes.append(bboxes)

    return all_bboxes
