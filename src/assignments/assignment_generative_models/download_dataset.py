"""
Download Fashion-MNIST using PyTorch and convert to numpy format
Reference: https://github.com/zalandoresearch/fashion-mnist
"""
import torch
from torchvision import datasets
from torchvision.transforms import ToTensor
import numpy as np
import os

print("Downloading Fashion-MNIST using PyTorch...")

# Download Fashion-MNIST with ToTensor transform
# This follows the official documentation from zalandoresearch/fashion-mnist
train_dataset = datasets.FashionMNIST(root='/tmp/fashion_mnist', train=True, download=True, transform=ToTensor())
test_dataset = datasets.FashionMNIST(root='/tmp/fashion_mnist', train=False, download=True, transform=ToTensor())

# Convert to numpy arrays
# When using ToTensor transform, we need to access the underlying data differently
# The dataset returns tensors, so we iterate through them
print("Converting to numpy arrays...")
x_train = np.array([img.numpy() for img, _ in train_dataset])
y_train = np.array([label for _, label in train_dataset])
x_test = np.array([img.numpy() for img, _ in test_dataset])
y_test = np.array([label for _, label in test_dataset])

# Convert from (N, 1, 28, 28) to (N, 28, 28) and scale back to [0, 255]
x_train = (x_train.squeeze() * 255).astype(np.uint8)
x_test = (x_test.squeeze() * 255).astype(np.uint8)

print(f"Train data shape: {x_train.shape}")
print(f"Test data shape: {x_test.shape}")

# Save to the keras cache directory
cache_dir = os.path.expanduser('~/.keras/datasets')
os.makedirs(cache_dir, exist_ok=True)

np.savez(os.path.join(cache_dir, 'fashion_mnist.npz'),
         x_train=x_train, y_train=y_train,
         x_test=x_test, y_test=y_test)

print(f"Fashion-MNIST saved to {cache_dir}/fashion_mnist.npz")
