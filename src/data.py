import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import matplotlib.pyplot as plt
from torchvision.transforms import ToPILImage
tpi = ToPILImage(mode="RGB")


class ModifiedMNISTDataset(Dataset):
    """Modified MNIST dataset."""

    def __init__(self, x_pkl_file, y_csv_file, to_rgb = False, transform=None):
        """
        Args:
            x_pkl_file (string): Path to the pickle file with images.
            y_csv_file (string): Path to the csv file with labels.
            to_rgb (bool): Whether to expand to 3 channels for pretrained models
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.images = pd.read_pickle(x_pkl_file)
        self.labels = pd.read_csv(y_csv_file)["Label"].to_list()
        self.to_rgb = to_rgb
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        label = torch.LongTensor([self.labels[idx]]).squeeze()
        image = torch.Tensor(self.images[idx]/256)
        # add channels (torchvision expext (H x W, C) for torch tensors)
        if self.to_rgb:
            image = image.unsqueeze(-1).repeat(1,1,3)
        if self.transform:
            image = self.transform(tpi(image))
        if self.to_rgb:
            # convert to (C x H x W) for model input
            image = image.transpose(0,2).transpose(1,2)
        return image, label


def imshow(inp, title=None):
    """Imshow for Tensor."""
    plt.imshow(inp, cmap="gray_r")
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated