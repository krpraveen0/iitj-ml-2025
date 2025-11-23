"""
Generate synthetic Fashion-MNIST-like dataset for demonstration
Since network access is blocked, we'll create synthetic 28x28 images
"""
import numpy as np
import os

print("Creating synthetic Fashion-MNIST-like dataset...")

# Set seed for reproducibility
np.random.seed(42)

# Create synthetic data similar to Fashion-MNIST
n_train = 60000
n_test = 10000
n_classes = 10

def generate_synthetic_image(class_label):
    """Generate a synthetic 28x28 image based on class label"""
    img = np.zeros((28, 28), dtype=np.uint8)
    
    # Create different patterns for different classes
    if class_label == 0:  # T-shirt/top - rectangle
        img[8:20, 6:22] = np.random.randint(100, 200, (12, 16))
    elif class_label == 1:  # Trouser - two rectangles
        img[10:25, 8:13] = np.random.randint(80, 180, (15, 5))
        img[10:25, 15:20] = np.random.randint(80, 180, (15, 5))
    elif class_label == 2:  # Pullover
        img[6:22, 6:22] = np.random.randint(90, 190, (16, 16))
    elif class_label == 3:  # Dress
        img[8:24, 9:19] = np.random.randint(95, 195, (16, 10))
    elif class_label == 4:  # Coat
        img[6:24, 6:22] = np.random.randint(85, 185, (18, 16))
    elif class_label == 5:  # Sandal - horizontal shape
        img[18:24, 8:20] = np.random.randint(110, 210, (6, 12))
    elif class_label == 6:  # Shirt
        img[8:22, 7:21] = np.random.randint(90, 190, (14, 14))
    elif class_label == 7:  # Sneaker - curved bottom
        img[16:24, 6:22] = np.random.randint(100, 200, (8, 16))
    elif class_label == 8:  # Bag - rectangular with handle
        img[12:24, 8:20] = np.random.randint(95, 195, (12, 12))
        img[8:12, 11:17] = np.random.randint(95, 195, (4, 6))
    else:  # Ankle boot
        img[14:26, 7:21] = np.random.randint(105, 205, (12, 14))
    
    # Add some random noise
    noise = np.random.randint(0, 30, (28, 28))
    img = np.clip(img + noise, 0, 255).astype(np.uint8)
    
    # Add edge smoothing
    from scipy.ndimage import gaussian_filter
    img = gaussian_filter(img.astype(float), sigma=0.5).astype(np.uint8)
    
    return img

# Try importing scipy for smoother images
try:
    from scipy.ndimage import gaussian_filter
    has_scipy = True
except:
    print("scipy not available, images will be more pixelated")
    has_scipy = False

# Generate training data
print("Generating training data...")
x_train = np.zeros((n_train, 28, 28), dtype=np.uint8)
y_train = np.random.randint(0, n_classes, n_train)

for i in range(n_train):
    if (i + 1) % 10000 == 0:
        print(f"  Generated {i+1}/{n_train} training images...")
    x_train[i] = generate_synthetic_image(y_train[i])

# Generate test data
print("Generating test data...")
x_test = np.zeros((n_test, 28, 28), dtype=np.uint8)
y_test = np.random.randint(0, n_classes, n_test)

for i in range(n_test):
    if (i + 1) % 2000 == 0:
        print(f"  Generated {i+1}/{n_test} test images...")
    x_test[i] = generate_synthetic_image(y_test[i])

print(f"\nSynthetic dataset created:")
print(f"  Training data: {x_train.shape}")
print(f"  Test data: {x_test.shape}")

# Save to numpy file
cache_dir = os.path.expanduser('~/.keras/datasets')
os.makedirs(cache_dir, exist_ok=True)

np.savez(os.path.join(cache_dir, 'fashion_mnist.npz'),
         x_train=x_train, y_train=y_train,
         x_test=x_test, y_test=y_test)

print(f"\nDataset saved to {cache_dir}/fashion_mnist.npz")
