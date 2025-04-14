# ECE570

# BYOL CIFAR-10 Project

## Overview
This project reimplements the Bootstrap Your Own Latent (BYOL) method for self-supervised representation learning on the CIFAR-10 dataset. In addition to reimplementing the original unsupervised BYOL pipeline, we extend the approach to a semi-supervised setting by incorporating a small fraction (10%) of labeled data. Furthermore, we compare the performance of BYOL against two popular contrastive methods—SimCLR and MoCo—to illustrate the advantages of a negative-sample-free strategy. A short demo video (under 5 minutes) is provided to explain the main features and design of the project.

## Code Structure
The project is organized into three primary Colab cells:
- **Cell 1: Unsupervised BYOL on CIFAR-10**  
  In this cell, we set up a data augmentation pipeline that produces two distinct augmented views of each input image. We modify a ResNet-18 to act as our encoder and build two MLP-based modules: a projector and a predictor. The online network’s predictions are compared (via mean-squared error loss) to the target network’s outputs, where the target network is updated using an exponential moving average.
  
- **Cell 2: Semi-Supervised BYOL**  
  Building on the unsupervised BYOL setup, this cell introduces a classifier head attached to the encoder to leverage a small subset of labeled CIFAR-10 images. The training loop optimizes a combined loss—unsupervised MSE loss for feature alignment plus a supervised cross-entropy loss—demonstrating that BYOL’s learned features can be effectively fine-tuned even with limited label availability.
  
- **Cell 3: Comparison with SimCLR and MoCo**  
  The final cell benchmarks BYOL against SimCLR and MoCo. SimCLR uses an NT-Xent loss with a concatenation of two augmented views, while MoCo employs a momentum-based key encoder and a dynamic queue to manage negative samples. Their respective loss curves are compared to show that BYOL achieves rapid convergence with much lower loss values.

## Environment Setup
To ensure you have all required dependencies, run the following command in your Colab notebook:
```bash
!pip freeze > requirements.txt
