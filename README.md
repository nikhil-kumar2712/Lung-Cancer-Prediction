# 🩺 Lung Cancer Detection

This project aims to detect **lung cancer** using two complementary approaches:  

1. **Deep Learning on CT-Scan Images** – to classify different lung cancer types from medical images. The dataset can be found here [Link](https://www.kaggle.com/datasets/mohamedhanyyy/chest-ctscan-images?resource=download).
2. **Traditional Machine Learning on Patient Survey Data** – to predict lung cancer risk based on patient attributes.  

By combining imaging data with clinical data, the project demonstrates a hybrid AI system for improved diagnosis and risk prediction.

---

## 📂 Dataset Description

### 1️⃣ CT-Scan Images Dataset  

The image dataset contains CT-Scan images from four categories:

- **Adenocarcinoma**: most common form of lung cancer (≈30–40% of NSCLC cases).  
- **Large Cell Carcinoma**: ≈10–15% of NSCLC cases.  
- **Squamous Cell Carcinoma**: ≈30% of NSCLC cases, generally linked to smoking.  
- **Normal**: CT-Scan images of healthy lung tissue.  

All images are in **JPG/PNG** format and resized appropriately for the model.

**Split:**  
- Training set – 70%  
- Testing set – 20%  
- Validation set – 10%  

### 2️⃣ Patient Survey Dataset (Tabular Data)  

This dataset contains patient-level features such as:  
- Age  
- Gender  
- Smoking history  
- Air pollution exposure  
- Genetic risk  
- Symptoms (coughing, fatigue, etc.)  

This tabular data is used to train classical ML algorithms to predict lung cancer presence or risk.

---

## 🛠️ Technologies Used

### Deep Learning:
- TensorFlow, Keras  
- Transfer Learning Models: ResNet50, VGG16, ResNet101, VGG19, DenseNet201, EfficientNetB4, MobileNetV2  
- Data Augmentation: `ImageDataGenerator`  
- Image Processing: PIL, OpenCV  
- Visualization: Matplotlib, Seaborn  

### Traditional ML:
- Scikit-Learn (Logistic Regression, Random Forest, SVM, KNN)  
- Data Preprocessing: Pandas, NumPy  
- Evaluation Metrics: Accuracy, Confusion Matrix, ROC-AUC  

---

## 🧠 Model Architectures

### A. Deep Learning (CT-Scan Images)
- Convolutional layers for feature extraction  
- Max Pooling layers for dimensionality reduction  
- Batch Normalization for faster convergence  
- Dense layers for classification into four categories  
- Transfer learning with pre-trained weights (ResNet, VGG, DenseNet, EfficientNet, MobileNet)

**Training Parameters:**  
- Optimizer: Adam (learning rate = 0.001)  
- Batch size: 16  
- Epochs: 50  

### B. Traditional Machine Learning (Patient Survey Data)
- Preprocessing: missing values handled, categorical features encoded, feature scaling  
- Algorithms evaluated: Logistic Regression, Random Forest, SVM, KNN  
- **Best Model:** Logistic Regression achieved **98% accuracy** on the test set  

---

## 📊 Results

### Deep Learning (CT-Scan Images)
- Achieved **95.2% accuracy** on the testing set  
- Training & validation accuracies and losses visualized per epoch
- ![Accuracy and Loss](https://raw.githubusercontent.com/BarriBarri20/Lung-cancer-detection-model-training/main/accuracy_epochs.png)

### Traditional Machine Learning (Patient Survey Data)
- Logistic Regression achieved **98% accuracy** on the testing set  
- Outperformed other classical ML models  
- Provides a fast, interpretable way to predict lung cancer risk based on patient attributes  

---

## 📝 Checkpoints
- Two checkpoints were saved during the training process. These checkpoints can be used to resume training or to evaluate the model on new data.

---

## 🌟 Conclusion
This project combines **computer vision** (deep learning on CT-Scan images) with **machine learning on tabular patient data** to create a comprehensive lung cancer detection system. It highlights the potential of integrating image-based and survey-based models for early diagnosis and decision support in healthcare.

---

## 🚀 How to Run

1. Clone the repository  
   ```bash
   git clone https://github.com/your-username/Lung-Cancer-Detection.git
   cd Lung-Cancer-Detection


