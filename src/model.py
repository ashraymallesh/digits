import torchvision
from torch import nn
import torch.nn.functional as F

def pretrained_resnet18(num_classes, pretrained=True):
    model = torchvision.models.resnet18(pretrained=pretrained)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    return model

def pretrained_resnet50(num_classes, pretrained=True):
    model = torchvision.models.resnet50(pretrained=pretrained)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    return model

def pretrained_mobilenet(num_classes):
    model = torchvision.models.mobilenet_v2(pretrained=True)
    model.classifier = nn.Sequential(
        nn.Dropout(0.2),
        nn.Linear(model.last_channel, num_classes))
    return model


class SimpleCNN(nn.Module):
    def __init__(self, input_dim: int, num_classes: int, apply_log_softmax: bool = False):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1_input_dim = int(((input_dim-4)/2-4)/2) ** 2 * 50
        self.fc1 = nn.Linear(self.fc1_input_dim, 500)
        self.fc2 = nn.Linear(500, num_classes)
        self.apply_log_softmax = apply_log_softmax

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, self.fc1_input_dim)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        if self.apply_log_softmax:
            x = F.log_softmax(x, dim=1)
        return x