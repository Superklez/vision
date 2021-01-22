import os
import pandas as pd
import torch

from PIL import Image
from skimage import io
from torch.utils.data import Dataset

class VOCDataset(Dataset):

    def __init__(self, csv_file, img_dir, label_dir, S=7, B=2, C=20, transform=None):
        super().__init__()
        self.annotations = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.transform = transform
        self.S = S
        self.B = B
        self.C = C

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        label_path = os.path.join(self.label_dir, self.annotations.iloc[index, 1])
        boxes = [] # technically bboxes

        with open(label_path) as f:
            for label in f.readlines():
                class_label, x, y, w, h = [
                    float(num) if float(num) != int(float(num)) else int(num)
                    for num in label.replace('\n', '').split()
                ]
                boxes.append([class_label, x, y, w, h])

        image_path = os.path.join(self.img_dir, self.annotations.iloc[index, 0])
        image = Image.open(image_path)
        boxes = torch.tensor(boxes)

        if self.transform:
            image, boxes = self.transform(image, boxes)

        label_matrix = torch.zeros((self.S, self.S, self.C + 5 * self.B)) # but only first 5 bboxes are used
        for box in boxes:
            class_label, x, y, w, h = box.tolist()
            class_label = int(class_label)
            #assert (type(class_label) == int)

            i = int(self.S * y)
            j = int(self.S * x)

            x_cell = self.S * x - j
            y_cell = self.S * y - i

            w_cell = self.S * w
            h_cell = self.S * h

            if label_matrix[i, j, self.C] == 0:
                label_matrix[i, j, self.C] == 1
                box_coordinates = torch.tensor(
                    [x_cell, y_cell, w_cell, h_cell]
                )
                label_matrix[i, j, self.C+1:self.C+5] = box_coordinates
                label_matrix[i, j, class_label] = 1

        return image, label_matrix
