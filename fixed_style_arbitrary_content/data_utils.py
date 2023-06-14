import torchvision
from torch.utils.data import DataLoader
from torchvision.transforms import transforms


def get_data_set(file_path, image_size):
    data_transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.mul(255))
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    dataset = torchvision.datasets.ImageFolder(file_path, transform=data_transform)

    return dataset


def get_data_loader_from_file(file_path, image_size, batch_size):
    dataset = get_data_set(file_path, image_size)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return data_loader


def get_data_loader_from_data_set(dataset, batch_size):
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return data_loader
