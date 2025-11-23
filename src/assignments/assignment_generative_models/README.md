# Generative Models Assignment: Autoencoders & GANs

This directory contains the implementation of two fundamental deep generative models on Fashion-MNIST dataset.

**Dataset Reference**: [Fashion-MNIST by Zalando Research](https://github.com/zalandoresearch/fashion-mnist)

## Contents

### Python Scripts
- `denoising_autoencoder.py` - Part (a): Denoising Convolutional Autoencoder implementation
- `dcgan.py` - Part (b): Deep Convolutional GAN implementation
- `generate_synthetic_data.py` - Helper script to generate synthetic Fashion-MNIST-like dataset
- `download_dataset.py` - Alternative dataset download script

### Documentation
- `report.tex` - LaTeX source for the technical report
- `requirements.txt` - Python package dependencies

### Output Directory
- `outputs/` - Contains all generated images, models, and training logs

## Requirements

Install dependencies:
```bash
pip install -r requirements.txt
```

Required packages:
- tensorflow>=2.10.0
- numpy>=1.21.0
- matplotlib>=3.5.0
- scipy (for synthetic data generation)

## Usage

### 1. Generate Synthetic Dataset (if needed)
```bash
python generate_synthetic_data.py
```

### 2. Run Denoising Autoencoder
```bash
python denoising_autoencoder.py
```

This will:
- Train a denoising autoencoder for 5 epochs
- Generate loss curves and reconstruction comparisons
- Save models and visualizations to `outputs/`

### 3. Run DCGAN
```bash
python dcgan.py
```

This will:
- Train a DCGAN for 20 epochs
- Generate loss curves and synthetic images
- Save models and visualizations to `outputs/`

### 4. Compile Report
```bash
pdflatex report.tex
```

Or upload `report.tex` to Overleaf and compile online.

## Architecture Details

### Denoising Autoencoder
- **Encoder**: Conv2D layers with MaxPooling → 32-dim latent vector
- **Decoder**: Dense + Conv2DTranspose with UpSampling
- **Loss**: Binary Cross-Entropy
- **Optimizer**: Adam

### DCGAN
- **Generator**: Dense → Conv2DTranspose layers with BatchNorm
- **Discriminator**: Conv2D layers with LeakyReLU and Dropout
- **Loss**: Binary Cross-Entropy (adversarial)
- **Optimizer**: Adam (lr=0.0002, beta_1=0.5)

## Results

### Denoising Autoencoder
- Successfully removes Gaussian noise from images
- Test Loss: 0.3333
- Test MSE: 0.0039
- Includes latent space interpolation visualization

### DCGAN
- Generates diverse Fashion-MNIST-like images
- Shows progressive improvement over training epochs
- Achieves stable adversarial training

## Output Files

Generated in `outputs/` directory:

**Denoising Autoencoder:**
- `dae_loss_curves.png` - Training/validation loss
- `dae_mse_curves.png` - MSE curves
- `dae_reconstruction_comparison.png` - Noisy → Clean → Denoised
- `dae_latent_interpolation.png` - Latent space interpolation
- `dae_training_log.txt` - Training metrics
- `denoising_autoencoder.h5` - Trained model
- `encoder.h5` - Encoder model
- `decoder.h5` - Decoder model

**DCGAN:**
- `dcgan_loss_curves.png` - D and G loss over time
- `dcgan_accuracy_curve.png` - Discriminator accuracy
- `dcgan_epoch_XXX.png` - Generated images at different epochs
- `dcgan_final_generated.png` - Final generated samples
- `dcgan_real_vs_fake.png` - Real vs generated comparison
- `dcgan_training_log.txt` - Training metrics
- `dcgan_generator.h5` - Trained generator
- `dcgan_discriminator.h5` - Trained discriminator

## Notes

- This implementation uses a synthetic Fashion-MNIST-like dataset generated locally due to network restrictions
- Training uses reduced dataset size (10,000 samples) and epochs for faster execution
- For production use, increase epochs and use full dataset
- The report is in LaTeX format and can be compiled on Overleaf

## References

1. Fashion-MNIST: Xiao et al., 2017
2. DCGAN: Radford et al., 2015
3. Denoising Autoencoders: Vincent et al., 2008
4. TensorFlow/Keras Documentation
