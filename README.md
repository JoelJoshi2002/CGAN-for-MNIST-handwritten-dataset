# GAN for MNIST Handwritten Digits

This project implements a Generative Adversarial Network (GAN) using PyTorch to generate images of handwritten digits similar to those in the MNIST dataset. The GAN consists of two neural networks: a generator that creates fake images from random noise and a discriminator that distinguishes between real and fake images. Through adversarial training, the generator improves its ability to produce realistic images.

# Project Overview

The goal of this project is to demonstrate the basic concepts of GANs by generating new handwritten digits that resemble the MNIST dataset. The MNIST dataset contains 60,000 training images and 10,000 testing images of handwritten digits (0-9), each 28x28 pixels in size.
## Key Features:

### Generator: A fully connected neural network that transforms a 64-dimensional noise vector into a 28x28 image.
### Discriminator: A fully connected neural network that classifies images as real or fake.
### Training: The GAN is trained for 20 epochs using the Adam optimizer and Binary Cross-Entropy loss.
### Visualization: Generated images are saved after each epoch to monitor progress.


# Requirements
To run this project, you need the following dependencies:

Python 3.8+
PyTorch 1.8+
Torchvision
NumPy
Matplotlib
TQDM (for progress bars)

You can install the required packages using:
pip install torch torchvision numpy matplotlib tqdm


# Setup

Clone the repository:
git clone https://github.com/JoelJoshi2002/CGAN-for-MNIST-handwritten-dataset
cd GAN-MNIST


Download the MNIST dataset:The dataset will be automatically downloaded when you run the code for the first time.

Run the training script:
python train_gan.py


This will train the GAN for 20 epochs and save generated images after each epoch in the images/ directory.

# Project Structure

GAN-MNIST/
│
├── data/                   # MNIST dataset (downloaded automatically)
├── images/                 # Generated images saved after each epoch
├── train_gan.py            # Main training script
├── generator.pth           # Saved generator model weights
├── discriminator.pth       # Saved discriminator model weights
└── README.md               # Project documentation


# Usage
Training the GAN

Run train_gan.py to start training.
The script will:
Load the MNIST dataset.
Initialize the generator and discriminator.
Train both networks using adversarial loss.
Save generated images after each epoch.
Save the final model weights.



# Generating New Images

After training, you can load the saved generator model to create new images:generator = Generator().to(device)
generator.load_state_dict(torch.load('generator.pth'))
noise = torch.randn(16, noise_dim, device=device)
fake_images = generator(noise)




# Results
After training for 20 epochs, the generator produces digit-like images. Below is a sample of generated images from the final epoch:


# Challenges and Solutions

Mode Collapse: The generator sometimes produces similar images. Increasing the noise dimension or using advanced GAN techniques like WGAN can help.
Training Instability: GANs can be tricky to train. Adjust learning rates or use gradient clipping if losses oscillate too much.
Image Quality: For better results, consider using convolutional layers or training for more epochs.


Future Work

Convolutional GAN (DCGAN): Replace fully connected layers with convolutional layers for better image quality.
Hyperparameter Tuning: Experiment with different learning rates, batch sizes, or noise dimensions.
Advanced Architectures: Explore Wasserstein GAN (WGAN) or Conditional GANs for more control over generated outputs.


# References

PyTorch-GAN GitHub Repository
DCGAN Tutorial — PyTorch Tutorials
Generating MNIST Digit Images using Vanilla GAN with PyTorch
MNIST GAN Training - YouTube


# License
This project is licensed under the MIT License. See the LICENSE file for details.
