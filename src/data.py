import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import matplotlib.pyplot as plt
from torchvision.transforms import ToPILImage
import numpy as np
tpi = ToPILImage()


class ModifiedMNISTDataset(Dataset):
    """Modified MNIST dataset."""

    def __init__(self, x: np.ndarray, y: np.ndarray,
                 to_rgb: bool = False,
                 transform=None):
        self.images = x
        self.labels = y
        self.to_rgb = to_rgb
        self.transform = transform

    @classmethod
    def from_files(cls, x_pkl_file, y_csv_file,
                   to_rgb: bool = False,
                   transform=None):
        """Constructor from files

        Included as separate constructor since pickling after train-test split
        does not work due to memory error. Therefore, we do the train/test split
        in memory and instantiate the Dataset object directly from the arrays.

        Args:
            x_pkl_file (string): Path to the pickle file with images.
            y_csv_file (string): Path to the csv file with labels.
            to_rgb (bool): Whether to expand to 3 channels for pretrained models
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        x = pd.read_pickle(x_pkl_file)
        y =  pd.read_csv(y_csv_file)["Label"].to_numpy()
        return cls(x, y, to_rgb, transform)

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
        elif self.to_rgb:
            # convert to (C x H x W) for model input
            image = image.transpose(0,2).transpose(1,2)
        return image, label


def imshow(inp, title=None):
    """Imshow for Tensor."""
    plt.imshow(inp, cmap="gray_r")
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated


def train_test_split(x: np.ndarray, y: pd.DataFrame,
                     y_label_col="Label", y_id_col="Id", split = 0.9):
    train_idx = []
    valid_idx = []
    for label in y[y_label_col].unique():
        label_subset = y.loc[y[y_label_col] == label]
        s = int(len(label_subset) * split)
        train_idx += label_subset[y_id_col].to_list()[:s]
        valid_idx += label_subset[y_id_col].to_list()[s:]

    train_x = x[train_idx]
    valid_x = x[valid_idx]
    train_y = y[y_label_col].iloc[train_idx].to_list()
    valid_y = y[y_label_col].iloc[valid_idx].to_list()

    return train_x, train_y, valid_x, valid_y