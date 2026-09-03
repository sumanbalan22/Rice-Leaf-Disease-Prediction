
# 🌾 Rice Leaf Disease Prediction using Deep Learning

## 📌 Project Overview

Rice Leaf Disease Prediction is a **Deep Learning-based image classification project** developed to identify diseases in rice leaves from uploaded images.

The project uses **Transfer Learning with MobileNetV2** to classify rice leaf images into different disease categories. A **Streamlit web application** is used to provide an easy-to-use interface where users can upload a rice leaf image and get the predicted disease.

## 🎯 Objectives

* Detect rice leaf diseases automatically from images.
* Build an image classification model using Deep Learning.
* Compare different Deep Learning models and select an effective model.
* Develop a simple web interface using Streamlit.
* Help demonstrate how AI can be applied to agricultural disease detection.

## 🧠 Machine Learning / Deep Learning Approach

The project follows these major steps:

1. **Dataset Collection**

   * Rice leaf images are collected and organized according to disease classes.

2. **Data Preprocessing**

   * Resize images to the required input size.
   * Normalize pixel values.
   * Apply image preprocessing and augmentation where required.

3. **Model Training**

   * Multiple Deep Learning models are evaluated.
   * Transfer Learning with **MobileNetV2** is used for the final prediction model.

4. **Model Evaluation**

   * Models are compared using classification performance metrics.
   * The best-performing model is selected for deployment.

5. **Deployment**

   * The trained MobileNetV2 model is integrated into a **Streamlit application**.
   * Users can upload an image and receive a disease prediction.

## 🏗️ System Workflow

```text
Rice Leaf Image
       ↓
Image Upload
       ↓
Image Preprocessing
       ↓
MobileNetV2 Model
       ↓
Feature Extraction
       ↓
Disease Classification
       ↓
Predicted Disease
       ↓
Streamlit Web Interface
```

## 🤖 Model Used

### MobileNetV2

**MobileNetV2** is a lightweight Convolutional Neural Network architecture designed for efficient image classification.

In this project, MobileNetV2 is used through **Transfer Learning**. The pretrained network is used to extract useful image features, and the model is adapted for rice leaf disease classification.

### Why MobileNetV2?

* Lightweight architecture
* Faster inference
* Suitable for deployment
* Lower computational requirements
* Effective for image classification
* Suitable for CPU-based applications

## 🛠️ Technologies Used

### Programming Language

* Python

### Deep Learning

* TensorFlow
* Keras
* MobileNetV2

### Data Processing

* NumPy
* Pandas

### Image Processing

* OpenCV
* PIL

### Machine Learning

* Scikit-learn

### Visualization

* Matplotlib
* Seaborn

### Deployment

* Streamlit

### Development Tools

* Jupyter Notebook
* Google Colab
* VS Code
* Git
* GitHub

## 📂 Project Structure

```text
Rice-Leaf-Disease-Prediction/
│
├── app.py
├── train.ipynb
├── Rice_Leaf_Model_Comparison (1).ipynb
├── rice_leaf_mobilenetv2.h5
├── class_names.npy
├── requirements.txt
├── README.md
└── pyvenv.cfg
```

## 📄 File Description

| File                                   | Description                                  |
| -------------------------------------- | -------------------------------------------- |
| `app.py`                               | Streamlit application for disease prediction |
| `train.ipynb`                          | Model training and preprocessing notebook    |
| `Rice_Leaf_Model_Comparison (1).ipynb` | Comparison of different Deep Learning models |
| `rice_leaf_mobilenetv2.h5`             | Trained MobileNetV2 model                    |
| `class_names.npy`                      | Stored disease class names                   |
| `requirements.txt`                     | Required Python libraries                    |
| `README.md`                            | Project documentation                        |

## 🚀 How to Run the Project

### 1. Clone the Repository

```bash
git clone https://github.com/sumanbalan22/Rice-Leaf-Disease-Prediction.git
```

### 2. Open the Project

```bash
cd Rice-Leaf-Disease-Prediction
```

### 3. Create a Virtual Environment

```bash
python -m venv venv
```

### 4. Activate the Virtual Environment

**Windows:**

```bash
venv\Scripts\activate
```

### 5. Install Required Libraries

```bash
pip install -r requirements.txt
```

### 6. Run the Streamlit Application

```bash
streamlit run app.py
```

The application will open in your browser.

## 💻 Application Features

* Upload rice leaf image
* Automatic image preprocessing
* Deep Learning-based disease prediction
* MobileNetV2 model inference
* Simple Streamlit user interface
* Easy-to-understand prediction output

## 📊 Model Comparison

Different Deep Learning architectures were evaluated during the project.

The model comparison notebook is included in the repository:

`Rice_Leaf_Model_Comparison (1).ipynb`

The final application uses **MobileNetV2** for prediction.

## 🔍 Explainable AI

The project can use **Grad-CAM (Gradient-weighted Class Activation Mapping)** to help visualize the important regions of the input leaf image that contribute to the model's prediction.

This makes the prediction more interpretable instead of treating the model as a complete black box.

## 🌱 Real-World Application

This type of system can be used as a supporting tool for:

* Farmers
* Agricultural researchers
* Crop monitoring systems
* Smart agriculture applications
* Early disease identification

The model should be considered an **AI-based prediction/support system**, not a replacement for professional agricultural diagnosis.

## 📈 Future Improvements

* Add more rice disease classes.
* Increase the size and diversity of the dataset.
* Improve model accuracy through hyperparameter tuning.
* Add real-time camera-based detection.
* Deploy the application to a cloud platform.
* Add disease treatment/recommendation information.
* Develop a mobile application.
* Improve explainability using Grad-CAM.

## 👨‍💻 Author

**Suman Balan**

M.Sc. Applied Data Science

### Skills Demonstrated

`Python` `SQL` `Machine Learning` `Deep Learning` `TensorFlow` `Keras` `Computer Vision` `OpenCV` `Pandas` `NumPy` `Scikit-learn` `Streamlit` `Power BI` `Tableau`

## ⭐ Project

If you find this project useful, consider giving the repository a ⭐ on GitHub.
