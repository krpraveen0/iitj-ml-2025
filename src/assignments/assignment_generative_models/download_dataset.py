"""
Download Fashion-MNIST using PyTorch and convert to numpy format
"""
import torch
from torchvision import datasets
import numpy as np
import os

print("Downloading Fashion-MNIST using PyTorch...")

# Download Fashion-MNIST
train_dataset = datasets.FashionMNIST(root='/tmp/fashion_mnist', train=True, download=True)
test_dataset = datasets.FashionMNIST(root='/tmp/fashion_mnist', train=False, download=True)

# Convert to numpy arrays
x_train = train_dataset.data.numpy()
y_train = train_dataset.targets.numpy()
x_test = test_dataset.data.numpy()
y_test = test_dataset.targets.numpy()

print(f"Train data shape: {x_train.shape}")
print(f"Test data shape: {x_test.shape}")

# Save to the keras cache directory
cache_dir = os.path.expanduser('~/.keras/datasets')
os.makedirs(cache_dir, exist_ok=True)

np.savez(os.path.join(cache_dir, 'fashion_mnist.npz'),
         x_train=x_train, y_train=y_train,
         x_test=x_test, y_test=y_test)

print(f"Fashion-MNIST saved to {cache_dir}/fashion_mnist.npz")
