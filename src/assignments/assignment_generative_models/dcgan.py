"""
Deep Convolutional GAN (DCGAN) Implementation on Fashion-MNIST
Part (b) of Generative Models Assignment

This script implements a DCGAN with custom training loop to generate
synthetic Fashion-MNIST images.
"""

import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import fashion_mnist
import os
import tensorflow as tf

# Set random seed for reproducibility
np.random.seed(42)
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

# Normalize pixel values to [-1, 1] (standard for GANs with tanh)
x_train = (x_train.astype('float32') - 127.5) / 127.5

# Use subset for faster training (first 10000 samples)
x_train = x_train[:10000]

# Reshape to include channel dimension
x_train = np.expand_dims(x_train, -1)

print(f"Training set shape: {x_train.shape}")
print(f"Pixel value range: [{x_train.min():.2f}, {x_train.max():.2f}]")

# Hyperparameters
latent_dim = 100
img_shape = (28, 28, 1)
epochs = 20
batch_size = 128
sample_interval = 10  # Save generated images every N epochs

print("\nBuilding DCGAN architecture...")

# Build Discriminator
def build_discriminator():
    """
    Discriminator: Binary classifier that distinguishes real from fake images
    Input: (28, 28, 1) image
    Output: Single probability (real/fake)
    """
    model = models.Sequential(name='discriminator')
    
    # Input: 28x28x1
    model.add(layers.Conv2D(64, (3, 3), strides=2, padding='same', input_shape=img_shape))
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.Dropout(0.3))
    
    # 14x14x64
    model.add(layers.Conv2D(128, (3, 3), strides=2, padding='same'))
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.Dropout(0.3))
    
    # 7x7x128
    model.add(layers.Conv2D(256, (3, 3), strides=2, padding='same'))
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.Dropout(0.3))
    
    # Flatten and output
    model.add(layers.Flatten())
    model.add(layers.Dense(1, activation='sigmoid'))
    
    return model

# Build Generator
def build_generator():
    """
    Generator: Creates fake images from random noise
    Input: 100-dimensional noise vector
    Output: (28, 28, 1) fake image
    """
    model = models.Sequential(name='generator')
    
    # Foundation for 7x7 image
    model.add(layers.Dense(7 * 7 * 256, input_dim=latent_dim))
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.Reshape((7, 7, 256)))
    
    # Upsample to 14x14
    model.add(layers.Conv2DTranspose(128, (4, 4), strides=2, padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU(alpha=0.2))
    
    # Upsample to 28x28
    model.add(layers.Conv2DTranspose(64, (4, 4), strides=2, padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU(alpha=0.2))
    
    # Output layer: 28x28x1 with tanh activation (values in [-1, 1])
    model.add(layers.Conv2D(1, (3, 3), activation='tanh', padding='same'))
    
    return model

# Create discriminator and compile
discriminator = build_discriminator()
discriminator.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

discriminator.summary()

# Create generator
generator = build_generator()
generator.summary()

# Create combined model (Generator + Discriminator)
# For training the generator, we freeze the discriminator
discriminator.trainable = False

gan_input = layers.Input(shape=(latent_dim,))
generated_image = generator(gan_input)
gan_output = discriminator(generated_image)

combined = models.Model(gan_input, gan_output, name='combined_gan')
combined.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5),
    loss='binary_crossentropy'
)

combined.summary()

print("\nStarting DCGAN training...")

# Training history
d_losses = []
g_losses = []
d_accuracies = []

# Fixed noise for consistent visualization
fixed_noise = np.random.normal(0, 1, (25, latent_dim))

# Custom training loop
for epoch in range(epochs):
    # Select a random batch of real images
    idx = np.random.randint(0, x_train.shape[0], batch_size)
    real_images = x_train[idx]
    
    # Generate fake images
    noise = np.random.normal(0, 1, (batch_size, latent_dim))
    fake_images = generator.predict(noise, verbose=0)
    
    # Labels for real and fake images
    real_labels = np.ones((batch_size, 1))
    fake_labels = np.zeros((batch_size, 1))
    
    # Add label smoothing (helps GAN stability)
    real_labels += 0.05 * np.random.random(real_labels.shape)
    fake_labels += 0.05 * np.random.random(fake_labels.shape)
    
    # Train Discriminator
    discriminator.trainable = True
    d_loss_real, d_acc_real = discriminator.train_on_batch(real_images, real_labels)
    d_loss_fake, d_acc_fake = discriminator.train_on_batch(fake_images, fake_labels)
    d_loss = 0.5 * (d_loss_real + d_loss_fake)
    d_acc = 0.5 * (d_acc_real + d_acc_fake)
    
    # Train Generator (via combined model)
    discriminator.trainable = False
    noise = np.random.normal(0, 1, (batch_size, latent_dim))
    # We want generator to fool discriminator, so we label fake images as real
    misleading_labels = np.ones((batch_size, 1))
    g_loss = combined.train_on_batch(noise, misleading_labels)
    
    # Store losses
    d_losses.append(d_loss)
    g_losses.append(g_loss)
    d_accuracies.append(d_acc)
    
    # Print progress
    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch + 1}/{epochs} | D Loss: {d_loss:.4f}, D Acc: {d_acc:.4f} | G Loss: {g_loss:.4f}")
    
    # Save generated images at intervals
    if (epoch + 1) % sample_interval == 0:
        # Generate images from fixed noise
        generated_images = generator.predict(fixed_noise, verbose=0)
        
        # Rescale images to [0, 1] for visualization
        generated_images = 0.5 * generated_images + 0.5
        
        # Create 5x5 grid
        fig, axes = plt.subplots(5, 5, figsize=(10, 10))
        for i in range(5):
            for j in range(5):
                idx = i * 5 + j
                axes[i, j].imshow(generated_images[idx].squeeze(), cmap='gray')
                axes[i, j].axis('off')
        
        plt.suptitle(f'Generated Images at Epoch {epoch + 1}', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, f'dcgan_epoch_{epoch + 1:03d}.png'), dpi=200, bbox_inches='tight')
        plt.close()

print("\nTraining completed!")

# Save final models
generator.save(os.path.join(OUTPUT_DIR, 'dcgan_generator.h5'))
discriminator.save(os.path.join(OUTPUT_DIR, 'dcgan_discriminator.h5'))
print("Models saved!")

# Save training history
with open(os.path.join(OUTPUT_DIR, 'dcgan_training_log.txt'), 'w') as f:
    f.write("DCGAN Training Log\n")
    f.write("="*50 + "\n\n")
    f.write(f"Architecture:\n")
    f.write(f"  Latent Dimension: {latent_dim}\n")
    f.write(f"  Image Shape: {img_shape}\n")
    f.write(f"  Epochs: {epochs}\n")
    f.write(f"  Batch Size: {batch_size}\n\n")
    f.write(f"Training History (every 10 epochs):\n")
    for epoch in range(0, epochs, 10):
        f.write(f"  Epoch {epoch+1}: D_loss={d_losses[epoch]:.4f}, "
                f"D_acc={d_accuracies[epoch]:.4f}, G_loss={g_losses[epoch]:.4f}\n")

print("\nGenerating final visualizations...")

# Plot loss curves
plt.figure(figsize=(12, 6))
plt.plot(d_losses, label='Discriminator Loss', linewidth=1.5, alpha=0.8)
plt.plot(g_losses, label='Generator Loss', linewidth=1.5, alpha=0.8)
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Loss', fontsize=12)
plt.title('DCGAN: Discriminator and Generator Loss Over Time', fontsize=14, fontweight='bold')
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'dcgan_loss_curves.png'), dpi=300, bbox_inches='tight')
plt.close()
print("Saved: dcgan_loss_curves.png")

# Plot discriminator accuracy
plt.figure(figsize=(12, 6))
plt.plot(d_accuracies, label='Discriminator Accuracy', linewidth=1.5, color='green', alpha=0.8)
plt.axhline(y=0.5, color='r', linestyle='--', label='Random Guess (0.5)', linewidth=1)
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Accuracy', fontsize=12)
plt.title('DCGAN: Discriminator Accuracy Over Time', fontsize=14, fontweight='bold')
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'dcgan_accuracy_curve.png'), dpi=300, bbox_inches='tight')
plt.close()
print("Saved: dcgan_accuracy_curve.png")

# Generate final set of images
print("\nGenerating final synthetic images...")
final_noise = np.random.normal(0, 1, (25, latent_dim))
final_generated = generator.predict(final_noise, verbose=0)
final_generated = 0.5 * final_generated + 0.5

fig, axes = plt.subplots(5, 5, figsize=(10, 10))
for i in range(5):
    for j in range(5):
        idx = i * 5 + j
        axes[i, j].imshow(final_generated[idx].squeeze(), cmap='gray')
        axes[i, j].axis('off')

plt.suptitle('Final Generated Fashion-MNIST Images', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'dcgan_final_generated.png'), dpi=300, bbox_inches='tight')
plt.close()
print("Saved: dcgan_final_generated.png")

# Create comparison with real images
real_sample = x_train[:25]
real_sample = 0.5 * real_sample + 0.5  # Rescale to [0, 1]

fig, axes = plt.subplots(2, 5, figsize=(12, 5))

# Top row: Real images
for i in range(5):
    axes[0, i].imshow(real_sample[i].squeeze(), cmap='gray')
    axes[0, i].axis('off')
    if i == 0:
        axes[0, i].set_title('Real Images', fontsize=10, fontweight='bold', loc='left')

# Bottom row: Generated images
for i in range(5):
    axes[1, i].imshow(final_generated[i].squeeze(), cmap='gray')
    axes[1, i].axis('off')
    if i == 0:
        axes[1, i].set_title('Generated Images', fontsize=10, fontweight='bold', loc='left')

plt.suptitle('Real vs Generated Fashion-MNIST Images', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'dcgan_real_vs_fake.png'), dpi=300, bbox_inches='tight')
plt.close()
print("Saved: dcgan_real_vs_fake.png")

print("\n" + "="*50)
print("DCGAN training completed!")
print("All outputs saved to:", OUTPUT_DIR)
print("="*50)
