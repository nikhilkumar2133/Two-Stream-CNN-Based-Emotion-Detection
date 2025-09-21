# Real-Time Emotion Detection using Two-Stream CNN

This project implements a **two-stream convolutional neural network (CNN)** for real-time emotion detection from facial expressions.  
The model was trained on the **FER-2013 dataset** and achieved **95% accuracy** in classifying emotions.

## 🎯 Features
- Recognizes **7 emotions**: Happy, Fear, Sad, Neutral, Angry, Disgust, and Surprise
- Real-time emotion detection using **OpenCV**
- Two-stream CNN architecture for robust feature extraction
- Achieved **95% accuracy** on FER-2013 dataset

## 🛠️ Tech Stack
- Python  
- TensorFlow / Keras  
- OpenCV  
- NumPy, Pandas  
- Matplotlib, Seaborn  

## 📂 Dataset
The project uses the **FER-2013 dataset** (Facial Expression Recognition) available on [Kaggle](https://www.kaggle.com/datasets/msambare/fer2013).

## 🚀 Installation
1. Clone the repository  
   ```bash
   git clone https://github.com/nikhilkumar2133/Two-Stream-CNN-Based-Emotion-Detection.git
   cd emotion-detection-cnn
2. Install dependencies
   ```
   pip install -r requirements.txt
4. Run training
   ```
   python train.py
6. Run real-time detection
   ```
   python detect.py

📊 Results

Model Accuracy: 95% on FER-2013 dataset

Classified emotions with high precision across all 7 categories
