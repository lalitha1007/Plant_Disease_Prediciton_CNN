# Plant_Disease_Prediciton_CNN

# 🌿 Plant Disease Detection Using Convolutional Neural Network (CNN)

> Accurate, Fast, and Automated Plant Disease Detection for Precision Agriculture

![Plant Disease Detection](https://img.shields.io/badge/Deep%20Learning-CNN-blue) 
![Python](https://img.shields.io/badge/Made%20with-Python-brightgreen) 
![TensorFlow](https://img.shields.io/badge/Framework-TensorFlow-orange)  
![License](https://img.shields.io/badge/License-MIT-yellow) 
![Accuracy](https://img.shields.io/badge/Accuracy-96.77%25-success)

---

## 📌 Overview

The health of plants directly impacts global food security. Manual disease detection by experts is time-consuming, expensive, and prone to errors.  
This project introduces a Deep Learning-based approach to detect and classify plant leaf diseases using Convolutional Neural Networks (CNNs), leveraging the PlantVillage Dataset from Kaggle.

Our trained CNN model achieved an accuracy of 96.77%, making it a promising tool for farmers, agronomists, and researchers in precision agriculture.

---

## 🚀 Features

- Automated Detection – No expert intervention needed.
- High Accuracy – 96.77% on validation data.
- Multiple Disease Classes – Trained on diverse plant leaf diseases.
- Fast Inference – Ideal for real-time deployment.
- Dataset Preprocessing – Clean, labeled, and expert-verified images.

---

## 🧠 How It Works

1. Data Collection – PlantVillage dataset from Kaggle.
2. Image Preprocessing – Cropping, noise removal, normalization.
3. Label Verification – Agricultural experts validate disease labels.
4. Model Training – CNN built and trained using TensorFlow/Keras.
5. Prediction – Model classifies leaf images into disease categories.

---

## 📂 Dataset

- Source: [Kaggle - PlantVillage Dataset](https://www.kaggle.com/datasets/emmarex/plantdisease)
- Classes: Multiple diseases + healthy leaves.
- Preprocessing Steps:
  - Removed low-quality images (< 500px).
  - Cropped around the region of interest (leaf area).
  - Removed duplicate images.
  - Normalized image sizes.

---

## 🏗 Model Architecture

- Type: Deep Convolutional Neural Network (CNN)
- Layers:
  - Convolutional layers with ReLU activation.
  - Pooling layers for downsampling.
  - Fully connected layers with dropout for regularization.
- Activation Function: ReLU
- Optimizer: Adam
- Loss Function: Categorical Crossentropy
- Framework: TensorFlow/Keras

---

## 📊 Results

| Metric       | Value    |
|--------------|----------|
| Accuracy     | 96.77%   |
| Dataset Size | 54,000+ images |
| Model Type   | CNN      |
| Training     | TensorFlow (GPU-enabled) |


---

## 💡 Use Cases

- 🌱 Farmers – Quick diagnosis to prevent crop loss.
- 🔬 Researchers – Training data for agricultural AI models.
- 📱 Mobile Apps – Integration for real-time leaf scanning.

---

## ⚙️ Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/Plant-Disease-Detection.git

# Navigate to the project folder
cd Plant-Disease-Detection

# Install dependencies
pip install -r requirements.txt
