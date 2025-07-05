# Violence & Weapon Detection in Surveillance Videos

This project detects **violent actions** and **weaponized individuals** in surveillance footage using two approaches:

* 🔍 Machine Learning (**CNN**)
* ⚙️ Deep Learning (**MobileNetV2 + LSTM**)

It classifies activities into:

* **Normal**
* **Violence**
* **Weaponized**

---

## 📂 Dataset

The project uses the **Surveillance Camera Violence Detection (SCVD)** dataset.

You can download the dataset directly from your notebook using **kagglehub**:

```python
import kagglehub

# Download the latest version
path = kagglehub.dataset_download("toluwaniaremu/smartcity-cctv-violence-detection-dataset-scvd")

print("Path to dataset files:", path)
```

> ⚠️ **The dataset is not included in this repository** due to its size.

**Dataset Folder Structure:**

```
Dataset/
├── Normal/
│   ├── train/
│   └── test/
├── Violence/
│   ├── train/
│   └── test/
└── Weaponized/
    ├── train/
    └── test/
```

---

## 🔍 Machine Learning Model (CNN)

### 🛠️ Model Architecture

* **Conv2D & MaxPooling:** Extract spatial features from individual frames.
* **Fully Connected Layers:** Perform binary classification (**Violent** or **Normal**).
* **Dropout:** Reduces overfitting.

### ✔️ Highlights

* Trained on individual frames.
* Binary classification (Normal vs Violence).
* Achieved \~80% accuracy on test data.

---

## ⚙️ Deep Learning Model (MobileNetV2 + LSTM)

### 🛠️ Model Architecture

* **MobileNetV2:** Pre-trained on ImageNet for spatial feature extraction.
* **TimeDistributed Layers:** Apply MobileNetV2 across all frames in a clip.
* **LSTM Layers:** Capture temporal patterns across video frames.
* **Fully Connected Layers:** Classify the entire video sequence.

### ✔️ Highlights

* Processes sequences of **15 frames** per video.
* Designed for recognizing motion-based activities.
* Achieved \~98% accuracy on test clips.

---

## 🔧 Repository Contents

| Model | Architecture       | Task                          | Accuracy |
| ----- | ------------------ | ----------------------------- | -------- |
| ML    | Custom CNN         | Frame-level classification    | \~80%    |
| DL    | MobileNetV2 + LSTM | Sequence-level classification | \~98%    |

Pre-trained models (`.keras` files) are included. You'll need to download the dataset and run the provided code to retrain or fine-tune.

---

## 🛠️ Tools & Libraries

* TensorFlow, Keras
* OpenCV
* NumPy, Pandas, Scikit-learn
* Matplotlib
* Jupyter Notebook, Google Colab
* Microsoft Azure, Office 365 (for collaboration and deployment)

---

## 📊 Evaluation Metrics

* Accuracy
* Precision & Recall
* Confusion Matrix
* Training/Validation Loss & Accuracy Curves

---

## 🔮 Future Work

* Deploy real-time detection on CCTV feeds.
* Expand the deep learning model to handle "Weaponized" as a separate class.
* Deploy on Azure or edge devices.
* Improve generalization across different datasets.

---

## 🙏 Acknowledgments

* Kaggle Dataset Contributors
* TensorFlow & OpenCV communities
