# CNN-Based Image Classifier: Panda, Cat, and Dog

![OpenCV Logo](https://opencv.org/wp-content/uploads/2021/06/OpenCV_logo_black_.png)

## Table of Contents
1. [Introduction](#introduction)
2. [Purpose](#purpose)
3. [Dataset](#dataset)
4. [Model Architecture](#model-architecture)
5. [Training Pipeline](#training-pipeline)
6. [Results and Performance Metrics](#results-and-performance-metrics)

---

### 1. Introduction

This project implements a **3-class image classifier** using a **Convolutional Neural Network (CNN)**. The goal was to classify images of **pandas**, **cats**, and **dogs**. The dataset consists of **3000 images** (1000 per class) and was split into an **80:20 ratio** for training and validation. The target validation accuracy was set at **85%**, and the final model achieved an accuracy of **91.33%**.

---

### 2. Purpose

The project serves a dual purpose:
1. **Building and Training a Neural Network from Scratch**  
   The project involves designing a CNN architecture, training it on real-world data, and analyzing its results.
2. **Reinforcing a Complete Training Pipeline**  
   The project follows a robust training pipeline inspired by OpenCV University, addressing common errors such as:
   - Input/output shape mismatches.
   - Loss function and optimizer compatibility.
   - Image transformation issues.
   - Misaligned plotting curves.

---

### 3. Dataset

The dataset, obtained from [Kaggle](https://www.kaggle.com/ashishsaxena2209/animal-image-datasetdog-cat-and-panda), consists of:
- **3000 images** (1000 per class: Panda, Cat, Dog).
- Images resized to **224x224** for consistency.
- Data split into **80% training** and **20% validation**.

#### Preprocessing and Augmentation
To enhance generalization, the following techniques were applied:
- Random rotation.
- Horizontal flipping.
- Random affine transfromation.
- Image Normalization

---

### 4. Model Architecture

The CNN model comprises:
1. **Convolutional Layers**: For feature extraction.
2. **MaxPooling Layers**: To reduce spatial dimensions.
3. **Dropout Layers**: To prevent overfitting.
4. **Fully Connected Layers**: For classification.

#### Key Highlights:
- **Input**: `(Batch Size, 3, 224, 224)`  
- **Output**: 3 class scores (Panda, Cat, Dog).  
- **Activation Functions**: ReLU for hidden layers, Softmax for output.  
- **Regularization**: Dropout and batch normalization.  

For a detailed breakdown of the architecture, refer to [Training_from_scratch.ipynb](https://github.com/04092000f/Image-Classifier-from-Scratch/blob/main/Training_from_scratch.ipynb).

---

### 5. Training Pipeline

The training process involved:
- Experimentation with:
  - Number of layers and filters.
  - Optimizers and learning rate schedulers.
  - Regularization techniques like data augmentation, dropout, and batch normalization.
  - Number of epochs.
- Validation accuracy monitored to ensure the model generalizes well.

Explore the training pipeline in the following notebooks:
1. [Data_and_Pipeline_Check.ipynb](https://github.com/04092000f/Image-Classifier-from-Scratch/blob/main/Data_and_Pipeline_Check.ipynb)
2. [Training_from_scratch.ipynb](https://github.com/04092000f/Image-Classifier-from-Scratch/blob/main/Training_from_scratch.ipynb)

---

### 6. Results and Performance Metrics

#### Validation Accuracy
The final validation accuracy achieved was **91.33%**, surpassing the target of 85%.

![Accuracy](https://github.com/04092000f/Image-Classifier-from-Scratch/blob/main/visuals/final_result.png)

#### Confusion Matrix
The confusion matrix indicates strong performance, with minimal misclassifications.

![Confusion Matrix](https://github.com/04092000f/Image-Classifier-from-Scratch/blob/main/visuals/confusion_matrix.png)

#### Sample Predictions
Here are some predictions made by the model:

![Sample Predictions](https://github.com/04092000f/Image-Classifier-from-Scratch/blob/main/visuals/sample_predictions.png)

---
