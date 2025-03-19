# MNIST GAN (Generative Adversarial Network)

A PyTorch implementation of a GAN (Generative Adversarial Network) trained on the MNIST dataset to generate handwritten digits.

## Project Overview

This project implements a GAN architecture to generate realistic handwritten digits by learning from the MNIST dataset. The network consists of two main components:

- **Generator**: Creates synthetic images from random noise
- **Discriminator**: Distinguishes between real and generated images

## Technical Details

### Architecture

#### Generator
- Input: Random noise vector (64-dimensional)
- Architecture: Series of transposed convolution layers
- Output: 28x28 grayscale images
- Uses BatchNorm and ReLU activations
- Final layer uses Tanh activation

#### Discriminator
- Input: 28x28 grayscale images
- Architecture: Series of convolutional layers
- Output: Single value (real/fake prediction)
- Uses LeakyReLU and BatchNorm
- Final layer uses sigmoid activation

### Hyperparameters

- Batch size: 128
- Noise dimension: 64
- Learning rate: 0.002
- Beta1: 0.5
- Beta2: 0.99
- Training epochs: 25

### Data Augmentation

The training data is augmented using:
- Random rotation (-20° to +20°)
- Normalization via ToTensor transform

## Training

The model is trained adversarially where:
- The discriminator tries to maximize its ability to identify real and fake images
- The generator tries to minimize its detection by creating more realistic images
- Both networks use Adam optimizer with custom beta parameters
- Training runs for 25 epochs on the full MNIST dataset

## Results

The training progress can be monitored through:
- Discriminator loss (D_loss)
- Generator loss (G_loss)
- Visual inspection of generated samples after each epoch

## Requirements

- PyTorch
- torchvision
- matplotlib
- numpy
- tqdm

## Model Artifacts

The trained generator model is saved as `generator_model.pth` and can be loaded to generate new digit images without retraining.

## Usage

To generate new images using the trained model:

```python
# Load the trained generator
G = Generator(noise_dim=64)
G.load_state_dict(torch.load('generator_model.pth'))
G.eval()

# Generate images
with torch.no_grad():
    noise = torch.randn(16, 64)  # Generate 16 images
    fake_images = G(noise)
