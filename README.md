# CNN-Based Image Classifier: Panda, Cat, and Dog


## Table of Contents
1. [Introduction](#1-introduction)
2. [Dataset](#2-dataset)
3. [Model Architecture](#3-model-architecture)
4. [Training Pipeline](#4-training-pipeline)
5. [Results and Performance Metrics](#5-results-and-performance-metrics)
6. [Model Inference](#6-model-inference)


---

### 1. Introduction

- This project implements a **multi-class image classifier** using a **Convolutional Neural Network (CNN)**. The goal was to classify images of **pandas**, **cats**, and **dogs**. The dataset consists of **3000 images** (1000 per class) and was split into an **80:20 ratio** for training and validation. The final trained model achieved an accuracy of **91.33%**.

- The project had the following purpose:
1. **Building and Training a Neural Network from Scratch**  
   The project involves designing a CNN architecture, training it on real-world data, and analyzing its results.
2. **Reinforcing a Complete Training Pipeline**  
   The project follows a robust training pipeline, addressing common errors such as:
   - Input/output shape mismatches.
   - Loss function and optimizer compatibility.
   - Image transformation issues.
   - Misaligned plotting curves.

---

### 2. Dataset

The dataset, obtained from [Kaggle](https://www.kaggle.com/ashishsaxena2209/animal-image-datasetdog-cat-and-panda), consists of:
- **3000 images** (1000 per class: Panda, Cat, Dog).
- Images resized to **224x224** for consistency.
- Data split into **80% training** and **20% validation**.
- Some data samples are visualized below<br>
   - For **Cats**:<br>
     <img src='https://github.com/04092000f/CatDogPandaClassifier/blob/main/visuals/cat.png' width='200' height='200'><img src='https://github.com/04092000f/CatDogPandaClassifier/blob/main/visuals/cat1.png' width='200' height='200'><img src='https://github.com/04092000f/CatDogPandaClassifier/blob/main/visuals/cat2.png' width='200' height='200'><img src='https://github.com/04092000f/CatDogPandaClassifier/blob/main/visuals/cat3.png' width='200' height='200'><br><br>
   - For **Dogs**:<br>
     <img src='https://github.com/04092000f/CatDogPandaClassifier/blob/main/visuals/dog.png' width='200' height='200'><img src='https://github.com/04092000f/CatDogPandaClassifier/blob/main/visuals/dog1.png' width='200' height='200'><img src='https://github.com/04092000f/CatDogPandaClassifier/blob/main/visuals/dog2.png' width='200' height='200'><img src='https://github.com/04092000f/CatDogPandaClassifier/blob/main/visuals/dog3.png' width='200' height='200'><br><br>
   - For **Pandas**:<br>
     <img src='https://github.com/04092000f/CatDogPandaClassifier/blob/main/visuals/panda.png' width='200' height='200'><img src='https://github.com/04092000f/CatDogPandaClassifier/blob/main/visuals/panda1.png' width='200' height='200'><img src='https://github.com/04092000f/CatDogPandaClassifier/blob/main/visuals/panda2.png' width='200' height='200'><img src='https://github.com/04092000f/CatDogPandaClassifier/blob/main/visuals/panda3.png' width='200' height='200'><br><br>

#### Preprocessing and Augmentation
To enhance generalization, the following augmentations were applied:
- Random rotation.
- Horizontal flipping.
- Random affine transfromation.
- Image Normalization

---

### 3. Model Architecture

For a detailed breakdown of the architecture, refer to [Training_from_scratch.ipynb](https://github.com/04092000f/Image-Classifier-from-Scratch/blob/main/Training_from_scratch.ipynb).

#### Key Highlights:
- **Input**: `(Batch Size, 3, 224, 224)`  
- **Output**: `3-class` scores (Panda, Cat, Dog).  
- **Activation Functions**: `ReLU` for hidden layers, `Softmax` for output.  
- **Regularization**: Dropout and batch normalization.  

---

### 4. Training Pipeline

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

### 5. Results and Performance Metrics

#### Validation Accuracy
The final validation accuracy achieved was **91.33%**.

![Accuracy](https://github.com/04092000f/Image-Classifier-from-Scratch/blob/main/visuals/final_result.png)

#### Confusion Matrix
The confusion matrix of the classification model is given below.

![Confusion Matrix](https://github.com/04092000f/Image-Classifier-from-Scratch/blob/main/visuals/confusion_matrix.png)

---

### 6. Model Inference
Here are some predictions made by the model:

![Sample Predictions](https://github.com/04092000f/Image-Classifier-from-Scratch/blob/main/visuals/sample_predictions.png)

---
