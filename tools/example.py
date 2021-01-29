from torch.utils.data import DataLoader, random_split
from torchvision.transforms import transforms
from utils import TorchvisionDataset, AlbumentationsDataset

import albumentations as A
from albumentations.pytorch import ToTensorV2

###############################################################################
### TORCHVISION ###############################################################
###############################################################################

#transform = transforms.Compose([
#    #transforms.ToPILImage(),
#    transforms.Resize(256),
#    transforms.RandomCrop(224),
#    transforms.ToTensor(),
#    transforms.Normalize(
#        mean = [0.485, 0.456, 0.406],
#        std = [0.229, 0.224, 0.225]
#    )
#])

torchvision_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomCrop(224),
    #transforms.RandomHorizontalFlip(0.5),
    #transforms.RandomVerticalFlip(0.5),
    transforms.ToTensor(),
    transforms.Normalize(
        mean = [0.485, 0.456, 0.406],
        std = [0.229, 0.224, 0.225]
    )
])

torchvision_dataset = TorchvisionDataset(csv_file='train.csv', root_dir='train', transform=torchvision_transform)
train_data, val_data = random_split(dataset, [num_train_samples, num_val_samples])

train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
val_loader = DataLoader(val_data, batch_size=64, shuffle=True)

###############################################################################
### ALBUMENTATIONS ############################################################
###############################################################################

albumentations_transform = A.Compose([
    A.Resize(256, 256),
    A.RandomCrop(244, 244),
    #A.HorizontalFlip(),
    A.Normalize(
        mean = [0.485, 0.456, 0.406],
        std = [0.229, 0.224, 0.225]
    ),
    ToTensorV2()
])

albumentations_dataset = AlbumentationsDataset(csv_file='train.csv', root_dir='train', transform=albumentations_transform)
train_data, val_data = random_split(dataset, [num_train_samples, num_val_samples])

train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
val_loader = DataLoader(val_data, batch_size=64, shuffle=True)

###############################################################################
### OTHER STUFF ###############################################################
###############################################################################

#train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
#val_loader = DataLoader(val_data, batch_size=64, shuffle=True)

#val_split = 0.2
#dataset_size = len(dataset)
#indices = list(range(dataset_size))

#val_len = int(np.floor(dataset_size * val_split))
#val_idx = np.random.choice(indices, size=test_len, replace=False)
#train_idx = list(set(indices) - set(val_idx))

#train_sampler = SubsetRandomSampler(train_idx)
#val_sampler = SubsetRandomSampler(val_idx)
