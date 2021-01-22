from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from utils import CustomImageDataset

normalize = transforms.Normalize(
    [0.485, 0.456, 0.406],
    [0.229, 0.224, 0.225]
)

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(256),
    transforms.RandomCrop(224),
    transforms.ToTensor(),
    normalize
])

train_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(256),
    transforms.RandomCrop(224),
    transforms.RandomRotation(10),
    #transforms.RandomHorizontalFlip(0.5),
    #transforms.RandomVerticalFlip(0.5),
    transforms.ToTensor(),
    normalize
])

test_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(224),
    #transforms.Resize(256),
    #transforms.CenterCrop(224),
    transforms.ToTensor(),
    normalize
])

dataset = CustomImageDataset(csv_file='train.csv', root_dir='train', transform=transform)
train_data, val_data = random_split(dataset, [num_train_samples, num_val_samples])

train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
val_loader = DataLoader(val_data, batch_size=64, shuffle=True)

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
