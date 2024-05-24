# Parkinson's Disease Detection Project
An exploratory project aimed at testing non-invasive methods for early-stage Parkinson's disease diagnosis.

Table of Contents
Introduction
Project Overview
Model Architecture
Platform
Accuracy
Dataset

This project focuses on predicting Parkinson's disease based on spiral drawings. Utilizing advanced deep learning techniques, we aim to diagnose Parkinson's disease in its early stages through non-invasive methods. Our primary objective is to improve diagnostic accuracy using convolutional neural networks (CNNs) to analyze the spiral images.

# Project Overview
Parkinson's disease is a progressive neurodegenerative disorder that affects movement control. Early diagnosis is crucial for effective management and treatment. Traditional diagnostic methods are often invasive and uncomfortable for patients. This project explores a non-invasive alternative using machine learning models to analyze patients' spiral drawings, which can reflect motor impairments characteristic of Parkinson's disease.

# Model Architecture
The model used in this project is a Convolutional Neural Network (CNN) designed to classify spiral images for Parkinson's disease detection. The architecture includes:

Input Layer: Accepts 128x128 grayscale images.
Convolutional Layers: Four layers with varying filter sizes and activation functions to extract features.
Max Pooling Layers: Following each convolutional layer to reduce spatial dimensions.
Dropout Layers: To prevent overfitting by randomly setting a fraction of input units to zero.
Fully Connected Layers: Two dense layers for final classification.
Output Layer: A softmax layer that outputs probabilities for two classes (Parkinson's or not).
Regularization is applied using L2 regularizers to prevent overfitting. The model is compiled with the Adam optimizer and categorical cross-entropy loss.

# Platform
The entire model training and evaluation process is conducted on Google Colab, leveraging its powerful GPU capabilities for efficient computation.

# Accuracy
The model achieved an accuracy of 85.91% on the validation dataset, demonstrating its effectiveness in early-stage Parkinson's disease diagnosis.

<img width="506" alt="accuracy" src="https://github.com/ujjawal-yadav/Parkinson-Prediction/assets/81307555/d59fe930-5ed6-4494-beb2-aa7d4bd79aee">


# Dataset
The dataset comprises spiral drawings collected from individuals with and without Parkinson's disease. The images are preprocessed and resized to 128x128 pixels for input into the CNN model.


