# VGG-16 Inspired CNN for CIFAR-100 (PyTorch)

## Overview
This project implements a VGG-16 inspired convolutional neural network using PyTorch and trains it on the CIFAR-100 dataset. The main objective was to reproduce key design ideas from the VGG architecture and explore how training choices affect optimisation and performance in deep convolutional networks.

The final model achieves **~66% test accuracy** on CIFAR-100.

---

## Model Architecture
The network follows the core principles of the VGG-16 design:

- Stacked 3×3 convolutional layers  
- Increasing feature depth (64 → 128 → 256 → 512)  
- Max-pooling between convolutional blocks  
- Fully connected layers for classification  

To improve training stability, **Batch Normalisation** is applied after each convolutional layer.  
The architecture is adapted for **32×32 input images**, unlike the original VGG model which was designed for ImageNet.

---

## Training Setup
- **Optimizer:** SGD with momentum (0.9)  
- **Initial learning rate:** 0.1  
- **Weight decay:** 5e-4  
- **Batch size:** 256  
- **Epochs:** up to 100  

---

## Learning Rate Scheduling
A **MultiStep learning rate scheduler** is used. Learning rate drop points were chosen based on observed plateaus in validation accuracy. After each drop, further performance improvements were observed.

This approach mirrors the staged learning rate reduction strategy commonly used in VGG-style training.

---

## Data Augmentation
To improve generalisation, the following data augmentation techniques are applied during training:

- Random cropping with padding  
- Random horizontal flipping  

Validation and test sets are kept **deterministic with no augmentation** to ensure fair evaluation.

---

## Results
- **Best test accuracy:** ~66%  
- Training loss continues to decrease while validation accuracy eventually plateaus, highlighting the optimisation limitations of deep plain CNNs without skip connections.
