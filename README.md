# 🖼️ Image Classification using Transfer Learning  
### ⚙️ VGG16 & MobileNetV2

---

## 📌 Overview
This project implements an **image classification pipeline using transfer learning** with two industry-standard convolutional neural networks:

- **VGG16** – accuracy-focused, high-capacity CNN  
- **MobileNetV2** – lightweight, mobile-optimized CNN  

Both models use **ImageNet pre-trained weights** as feature extractors and are extended with custom classification heads.  
The project is designed and executed in **Google Colab**, with datasets stored in **Google Drive**.

---

## ✨ Key Features
- 🔁 Transfer learning with **VGG16** and **MobileNetV2**
- 🧠 Modular model pipeline for architecture comparison
- 🖼️ Image preprocessing and augmentation
- ❄️ Frozen base models to reduce overfitting
- 📊 Quantitative evaluation using standard ML metrics
- 📈 Visualization of training and validation performance

---

## 🧰 Tech Stack
- 🐍 **Python**
- 🧠 **TensorFlow / Keras**
- 📐 **NumPy**
- 📊 **Matplotlib & Seaborn**
- 📉 **Scikit-learn**
- ☁️ **Google Colab + Google Drive**

---

## 🧠 Models Used

### 🔹 VGG16
- Deep convolutional neural network
- High representational capacity
- Suitable for accuracy-driven experiments
- Higher memory and compute cost

### 🔹 MobileNetV2
- Lightweight architecture using depthwise separable convolutions
- Optimized for speed and low-resource environments
- Suitable for mobile and edge deployment

> Both models are initialized with `include_top=False` and **ImageNet weights**.

---

## 🏗️ Model Architecture (Common Head)
- Base Model: **VGG16 / MobileNetV2** (frozen)
- Global Average Pooling
- Dense layer (ReLU)
- Dropout (regularization)
- Dense output layer (Softmax)

---

## 📂 Dataset Structure
The dataset must follow the structure below:
dataset/
│── train/
│ ├── class_1/
│ ├── class_2/
│ └── ...
│
│── val/
│ ├── class_1/
│ ├── class_


📌 Each folder name is automatically treated as a **class label**.

---

## ⚙️ Training Configuration

| Parameter | Value |
|---------|-------|
| 🖼️ Image Size | 224 × 224 |
| 📦 Batch Size | 200 |
| ⚡ Optimizer | Adam |
| 🎯 Loss Function | Categorical Crossentropy |
| 📈 Metrics | Accuracy |
| 🔄 Data Augmentation | Enabled (train only) |

---

## 📊 Evaluation Metrics
- ✅ Accuracy
- 📏 Precision
- 🔁 Recall
- 🧮 F1-Score
- 📉 Confusion Matrix

Predictions are evaluated using **`sklearn.metrics`** for objective comparison.

---

## ▶️ How to Run
1. Open the notebook in **Google Colab**
2. Mount **Google Drive**
3. Verify dataset paths
4. Run cells sequentially:
   1. Load dataset
   2. Initialize models
   3. Train VGG16
   4. Train MobileNetV2
   5. Evaluate and visualize results

---

## 📤 Outputs
- 🧠 Trained VGG16-based classifier
- ⚡ Trained MobileNetV2-based classifier
- 📈 Accuracy & loss plots
- 🔥 Confusion matrix heatmaps
- 🧾 Classification reports for both models

---

## ⚖️ Comparative Insight
- **VGG16** → better representational power, higher resource usage
- **MobileNetV2** → faster training and inference, lower memory footprint

📌 The project demonstrates **accuracy vs efficiency trade-offs** in real-world ML systems.

---

## 🚀 Future Enhancements
- 🔓 Fine-tune upper convolutional layers
- 📉 Add learning-rate scheduling
- 💾 Save and reload trained models
- 🖼️ Add single-image inference script
- ⏱️ Benchmark inference latency and model size

---

## 🎯 Use Cases
- 🎓 Academic transfer learning experiments
- 🧪 CNN architecture comparison
- 📌 Baseline for vision-based ML projects
- 📱 Edge vs cloud deployment analysis

