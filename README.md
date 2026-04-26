# 🧬 Malaria Cell Detection (HOG + SVM)

This project is a machine learning web application that detects whether a blood cell is **infected (Parasitized)** or **uninfected** using classical computer vision techniques.

The model is based on:

- **HOG (Histogram of Oriented Gradients)** → feature extraction  
- **SVM (Support Vector Machine)** → classification  
- **Flask** → backend API  

---

## 🎯 Project Goal

The main objective of this project is to explore and understand the differences between **classical machine learning** and **deep learning** approaches for image classification.

This application was developed as part of a learning process to:

- Understand how traditional computer vision techniques work  
- Build a complete ML pipeline from scratch  
- Compare performance with modern deep learning models (YOLO)  

---

## 🔍 Approach 1: HOG + SVM (This Project)

This project uses a classical pipeline:

- Extract features using **HOG (Histogram of Oriented Gradients)**  
- Classify using **Support Vector Machine (SVM)**  

### ✅ Advantages
- Lightweight (no GPU required)  
- Fast inference  
- Easy to interpret  
- Works well on small datasets  

### ❌ Limitations
- Requires manual feature engineering  
- Sensitive to preprocessing  
- Limited performance on complex patterns  

---

## 🤖 Approach 2: YOLO (Deep Learning)

A YOLO-based model is used for comparison.

YOLO (**You Only Look Once**) is a deep learning model for object detection.

### ✅ Advantages
- Automatically learns features  
- High accuracy on complex data  
- Real-time detection capability  

### ❌ Limitations
- Requires more data  
- Needs GPU for training  
- Higher computational cost  

---

## ⚖️ Comparison Objective

The goal is to compare both approaches based on:

- Accuracy  
- Inference speed  
- Model complexity  
- Ease of deployment  
- Performance on real-world data  

---

## 🧠 Learning Outcomes

- Image preprocessing using OpenCV  
- Feature extraction with HOG  
- Machine learning modeling (SVM)  
- Model evaluation (accuracy, confusion matrix, ROC)  
- Backend development using Flask  
- Understanding differences between ML and Deep Learning  
- Practical comparison between HOG+SVM and YOLO  

---

## 🚀 Features

- Upload a cell image  
- Automatic HOG feature extraction  
- Predict infected vs uninfected  
- Fast and lightweight inference  

---

## 📊 Dataset

- **Malaria Cell Images Dataset**

Classes:
- Parasitized  
- Uninfected  

---

## 🧠 Model Pipeline

Input Image
↓
Resize (fixed size)
↓
Convert to Grayscale
↓
Extract HOG Features
↓
Standard Scaling
↓
SVM Classifier (Calibrated)
↓
Prediction (Infected / Uninfected)


---

## 🧪 Technologies Used

- Python
- OpenCV
- Scikit-image (HOG)
- Scikit-learn (SVM)
- Flask (API)

---

## ⚙️ Library Versions (IMPORTANT)

This model **must use the same versions** to work correctly:
numpy==2.0.2
opencv-python==4.13.0.92
joblib==1.5.3
matplotlib==3.10.0
seaborn==0.13.2
scikit-learn==1.6.1
scikit-image==0.25.2
tqdm==4.67.3
flask
flask-cors


---

## 📦 Installation

Install dependencies:

```bash
pip install numpy==2.0.2 opencv-python==4.13.0.92 joblib==1.5.3 matplotlib==3.10.0 seaborn==0.13.2 scikit-learn==1.6.1 scikit-image==0.25.2 tqdm==4.67.3 flask flask-cors
