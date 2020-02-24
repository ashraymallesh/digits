

# Modified MNIST Handwritten Digit Recognition

In this mini-project the goal is to perform an image analysis prediction challenge. The task is based upon the MNIST dataset (https://en.wikipedia.org/wiki/MNIST_database). The original MNIST contains handwritten numeric digits from 0-9 and the goal is to classify which digit is present in an image. Here, you will be working with a Modified MNIST dataset that we have constructed. In this modified dataset, the images contain three digits, and the goal is to output the digit in the image with the highest numeric value.

### [Project 3 Spec](miniproject3_spec.pdf)

## Experiments:
- **Model Architecture**: Simple CNN, ResNet50, MobileNet, ShuffleNet
- **Model Size**: ResNet18, ResNet50, ResNet101, WideResNet50
- **fast.ai Data Augmentation**: transformations include jitter, perspective warp, squish, skew, tilt, totate, zoom, lighting, contrast at training, evaluating effect of augmentation on ResNet18
- **Image Resizing**: evaluating effect of image resizing - 128x128, 64x64, 256x256 (with data augmentation)
- **Model Pretraining**: ResNet18 with/without pretraining on imagenet


### Kaggle Competition Leaderboard (11th Place - Top 10% - Group 32)
https://www.kaggle.com/c/modified-mnist/leaderboard


