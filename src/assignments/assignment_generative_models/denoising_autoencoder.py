"""
Denoising Autoencoder Implementation on Fashion-MNIST
Part (a) of Generative Models Assignment

This script implements a Denoising Convolutional Autoencoder that learns to
reconstruct clean images from noisy inputs.
"""

import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import fashion_mnist
import os

# Set random seed for reproducibility
np.random.seed(42)
import tensorflow as tf
tf.random.set_seed(42)

# Create outputs directory
OUTPUT_DIR = 'outputs'
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("Loading Fashion-MNIST dataset...")
# Load Fashion-MNIST dataset from local cache
cache_path = os.path.expanduser('~/.keras/datasets/fashion_mnist.npz')
data = np.load(cache_path)
x_train, y_train = data['x_train'], data['y_train']
x_test, y_test = data['x_test'], data['y_test']

# Normalize pixel values to [0, 1]
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# Reshape to include channel dimension (28, 28, 1)
x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)

print(f"Training set shape: {x_train.shape}")
print(f"Test set shape: {x_test.shape}")

# Use subset for faster training (first 10000 samples)
x_train = x_train[:10000]
# Split training data into train (80%) and validation (20%)
split_idx = int(0.8 * len(x_train))
x_train_clean = x_train[:split_idx]
x_val_clean = x_train[split_idx:]

print(f"Train clean shape: {x_train_clean.shape}")
print(f"Validation clean shape: {x_val_clean.shape}")

# Add Gaussian noise to create noisy versions
noise_factor = 0.5
x_train_noisy = x_train_clean + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_train_clean.shape)
x_val_noisy = x_val_clean + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_val_clean.shape)
x_test_noisy = x_test + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_test.shape)

# Clip values to [0, 1] range
x_train_noisy = np.clip(x_train_noisy, 0.0, 1.0)
x_val_noisy = np.clip(x_val_noisy, 0.0, 1.0)
x_test_noisy = np.clip(x_test_noisy, 0.0, 1.0)

print("\nBuilding Denoising Autoencoder architecture...")

# Define latent dimension
latent_dim = 32

# Build Encoder
encoder_input = layers.Input(shape=(28, 28, 1), name='encoder_input')

# Encoder layers
x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(encoder_input)
x = layers.MaxPooling2D((2, 2), padding='same')(x)  # 14x14
x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
x = layers.MaxPooling2D((2, 2), padding='same')(x)  # 7x7
x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
x = layers.MaxPooling2D((2, 2), padding='same')(x)  # 4x4 (rounded up)

# Flatten to latent vector
x = layers.Flatten()(x)
encoder_output = layers.Dense(latent_dim, activation='relu', name='latent_vector')(x)

encoder = models.Model(encoder_input, encoder_output, name='encoder')
encoder.summary()

# Build Decoder
decoder_input = layers.Input(shape=(latent_dim,), name='decoder_input')

# Dense layer to reshape
x = layers.Dense(4 * 4 * 128, activation='relu')(decoder_input)
x = layers.Reshape((4, 4, 128))(x)

# Decoder layers
x = layers.Conv2DTranspose(128, (3, 3), activation='relu', padding='same')(x)
x = layers.UpSampling2D((2, 2))(x)  # 8x8
x = layers.Conv2DTranspose(64, (3, 3), activation='relu', padding='same')(x)
x = layers.UpSampling2D((2, 2))(x)  # 16x16
x = layers.Conv2DTranspose(32, (3, 3), activation='relu', padding='same')(x)
x = layers.UpSampling2D((2, 2))(x)  # 32x32

# Crop to 28x28 and output
x = layers.Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)
decoder_output = layers.Cropping2D(cropping=((2, 2), (2, 2)))(x)  # 28x28

decoder = models.Model(decoder_input, decoder_output, name='decoder')
decoder.summary()

# Combine Encoder and Decoder into Autoencoder
autoencoder_input = layers.Input(shape=(28, 28, 1))
encoded = encoder(autoencoder_input)
decoded = decoder(encoded)
autoencoder = models.Model(autoencoder_input, decoded, name='autoencoder')

# Compile the autoencoder
autoencoder.compile(optimizer='adam', loss='binary_crossentropy', metrics=['mse'])
autoencoder.summary()

print("\nTraining the Denoising Autoencoder...")
# Train the autoencoder
epochs = 5
batch_size = 128

history = autoencoder.fit(
    x_train_noisy, x_train_clean,
    epochs=epochs,
    batch_size=batch_size,
    validation_data=(x_val_noisy, x_val_clean),
    shuffle=True,
    verbose=1
)

print("\nEvaluating on test set...")
# Evaluate on test set
test_loss, test_mse = autoencoder.evaluate(x_test_noisy, x_test, verbose=0)
print(f"Test Loss (Binary Cross-Entropy): {test_loss:.4f}")
print(f"Test MSE: {test_mse:.4f}")

# Save training history
with open(os.path.join(OUTPUT_DIR, 'dae_training_log.txt'), 'w') as f:
    f.write("Denoising Autoencoder Training Log\n")
    f.write("="*50 + "\n\n")
    f.write(f"Architecture:\n")
    f.write(f"  Latent Dimension: {latent_dim}\n")
    f.write(f"  Noise Factor: {noise_factor}\n")
    f.write(f"  Epochs: {epochs}\n")
    f.write(f"  Batch Size: {batch_size}\n\n")
    f.write(f"Test Results:\n")
    f.write(f"  Test Loss: {test_loss:.4f}\n")
    f.write(f"  Test MSE: {test_mse:.4f}\n\n")
    f.write(f"Training History:\n")
    for epoch in range(epochs):
        f.write(f"  Epoch {epoch+1}: loss={history.history['loss'][epoch]:.4f}, "
                f"val_loss={history.history['val_loss'][epoch]:.4f}\n")

print("\nGenerating visualizations...")

# Plot training and validation loss curves
plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Training Loss', linewidth=2)
plt.plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Loss (Binary Cross-Entropy)', fontsize=12)
plt.title('Denoising Autoencoder: Training and Validation Loss', fontsize=14, fontweight='bold')
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'dae_loss_curves.png'), dpi=300, bbox_inches='tight')
plt.close()
print("Saved: dae_loss_curves.png")

# Plot MSE curves
plt.figure(figsize=(10, 6))
plt.plot(history.history['mse'], label='Training MSE', linewidth=2)
plt.plot(history.history['val_mse'], label='Validation MSE', linewidth=2)
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Mean Squared Error', fontsize=12)
plt.title('Denoising Autoencoder: Training and Validation MSE', fontsize=14, fontweight='bold')
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'dae_mse_curves.png'), dpi=300, bbox_inches='tight')
plt.close()
print("Saved: dae_mse_curves.png")

# Generate reconstructions for visualization
n_images = 10
reconstructed_images = autoencoder.predict(x_test_noisy[:n_images], verbose=0)

# Create visualization showing noisy input, original, and reconstructed
fig, axes = plt.subplots(3, n_images, figsize=(20, 6))

for i in range(n_images):
    # Noisy input
    axes[0, i].imshow(x_test_noisy[i].squeeze(), cmap='gray')
    axes[0, i].axis('off')
    if i == 0:
        axes[0, i].set_title('Noisy Input', fontsize=10, fontweight='bold')
    
    # Original clean image
    axes[1, i].imshow(x_test[i].squeeze(), cmap='gray')
    axes[1, i].axis('off')
    if i == 0:
        axes[1, i].set_title('Original Clean', fontsize=10, fontweight='bold')
    
    # Reconstructed (denoised) image
    axes[2, i].imshow(reconstructed_images[i].squeeze(), cmap='gray')
    axes[2, i].axis('off')
    if i == 0:
        axes[2, i].set_title('Denoised Output', fontsize=10, fontweight='bold')

plt.suptitle('Denoising Autoencoder Results: Noisy Input → Original → Denoised Output', 
             fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'dae_reconstruction_comparison.png'), dpi=300, bbox_inches='tight')
plt.close()
print("Saved: dae_reconstruction_comparison.png")

# Save model
autoencoder.save(os.path.join(OUTPUT_DIR, 'denoising_autoencoder.h5'))
encoder.save(os.path.join(OUTPUT_DIR, 'encoder.h5'))
decoder.save(os.path.join(OUTPUT_DIR, 'decoder.h5'))
print("\nModels saved successfully!")

# Bonus: Latent Space Interpolation
print("\nGenerating latent space interpolation (Bonus)...")

# Find indices for different classes (e.g., 0: T-shirt/top, 7: Sneaker)
class_0_idx = np.where(y_test == 0)[0][0]  # T-shirt
class_7_idx = np.where(y_test == 7)[0][0]  # Sneaker

# Get latent vectors
z_tshirt = encoder.predict(np.expand_dims(x_test[class_0_idx], 0), verbose=0)
z_sneaker = encoder.predict(np.expand_dims(x_test[class_7_idx], 0), verbose=0)

# Generate 10 interpolated images
n_interpolations = 10
interpolated_images = []

for alpha in np.linspace(0, 1, n_interpolations):
    z_interp = alpha * z_sneaker + (1 - alpha) * z_tshirt
    img_interp = decoder.predict(z_interp, verbose=0)
    interpolated_images.append(img_interp[0])

# Visualize interpolation
fig, axes = plt.subplots(1, n_interpolations, figsize=(20, 2))
for i, img in enumerate(interpolated_images):
    axes[i].imshow(img.squeeze(), cmap='gray')
    axes[i].axis('off')
    axes[i].set_title(f'{i/9:.1f}', fontsize=8)

plt.suptitle('Latent Space Interpolation: T-shirt → Sneaker', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'dae_latent_interpolation.png'), dpi=300, bbox_inches='tight')
plt.close()
print("Saved: dae_latent_interpolation.png")

print("\n" + "="*50)
print("Denoising Autoencoder training completed!")
print("All outputs saved to:", OUTPUT_DIR)
print("="*50)
