# FaceAgePrediction
This model is coded in python language. Trained with more than 10k human images. Accuracy score is over %90. 
# Face Age Prediction

A computer vision project that predicts a person's age from a facial image using Python and deep learning techniques.

---

## 🔍 Overview

This project implements an end-to-end **face age estimation** pipeline. Given an input image, the system detects the face, preprocesses it, and predicts the person's age using a trained neural network model.

The focus of the project is on:

* Robust model design
* Reproducible experiments
* Clear project structure

---

## 🧠 Methodology

1. **Face Detection**

   * Face region is detected from the input image using a computer vision-based approach.


2. **Age Prediction**

   * A FaceNet-based feature extraction model followed by a regression head predicts the age as a continuous value

---

## 🗂 Dataset

* The model is trained on **10k+ facial images** collected from publicly available datasets.
* The dataset is not included in this repository due to size limitations.
* It is definitely safe in ethics. All used images are taken with permission. 

📌 Dataset source: 

--- https://drive.google.com/drive/folders/1rcSGjbKg0DItreaFRj1iiPU9LUu5cZyy?usp=sharing

🏗 Model Details

Framework: PyTorch

Backbone Architecture: InceptionResNetV1 (FaceNet-based)

Pretrained Weights: FaceNet (facial feature extraction)

Task Formulation: Regression (continuous age prediction)

Architecture Overview

Face Feature Extraction

A pre-trained InceptionResNetV1 model is used to extract high-level facial embeddings from input images.

Embedding Processing

The extracted facial embeddings represent discriminative facial features relevant to age estimation.

Age Regression Head

Fully connected layers map facial embeddings to a single continuous age value.

Training Details

Image Preprocessing:

Face cropping and alignment

Resizing and normalization using torchvision.transforms


---

## 📊 Results

* The trained model achieves reasonable accuracy on unseen test images.

---

## 🚀 How to Run

```bash
pip install -r requirements.txt
python src/BigSystem.py --image path/to/image.jpg
```

---

## 🛠 Technologies Used

* Python
* OpenCV
* TensorFlow / PyTorch
* NumPy
* Matplotlib
  
📘 Project Guide

A detailed step-by-step guide explaining the system workflow, model architecture, and usage instructions is provided as a PowerPoint presentation.

📎 Guide: See guide/Guide.pptx
---

## 📄 License

This project is licensed under the MIT License.

---

## 👤 Author

**Ali Barış Tunalı**
