Violence & Weapon Detection in Surveillance Videos
This project focuses on detecting violent actions and weaponized individuals in video surveillance footage using two approaches:
🔍 Machine Learning (ML) with CNN
⚙️ Deep Learning (DL) with MobileNetV2 + LSTM

The system classifies actions into:

Normal

Violence

Weaponized

📂 Dataset
The dataset used is the Surveillance Camera Violence Detection (SCVD) dataset from Kaggle:
🔗 Surveillance Camera Violence Detection - Kaggle

⚠️ The dataset is not included in this repository due to its large size. Please download it directly from Kaggle.

Folder Structure (after download):
bash
Copy
Edit
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
🔍 Machine Learning Approach: Custom CNN
🛠 Model Architecture:
Conv2D Layers: Extract basic spatial features from individual video frames.

MaxPooling: Reduces dimensionality.

Flatten + Dense: Fully connected layers classify the frame as Violent or Normal.

Dropout: Prevents overfitting.

⚙️ Workflow:
Preprocess video frames: resized, normalized.

Each frame independently classified as violent or normal.

Binary Classification - does not distinguish "Weaponized" as a separate class.

Best Accuracy: ~80% on frame-level classification.

✅ Strengths:
Lightweight and fast for real-time frame classification.

Simpler to train on smaller datasets.

⚙️ Deep Learning Approach: MobileNetV2 + LSTM
🛠 Model Architecture:
MobileNetV2: Pre-trained on ImageNet, extracts efficient spatial features from each frame (low computation cost).

TimeDistributed Layer: Applies MobileNetV2 to each frame in the video clip.

LSTM Layers: Captures temporal relationships and motion patterns across the 15 frames.

Dense Output: Performs binary classification (Violent vs Normal) by combining spatial and temporal features.

⚙️ Workflow:
Preprocess video clips into 15 frames each.

MobileNetV2 extracts features for each frame.

LSTM learns patterns over time (e.g., repetitive violent actions).

Final Sigmoid layer predicts if the clip contains violence.

Best Accuracy: ~98% on clip-level classification.

✅ Strengths:
Handles motion-based patterns, not just single frames.

Much higher accuracy on real-world scenarios where violence involves sequences of actions.

🔧 Models Included in the Repository
Model Type	Architecture	Purpose	Accuracy
ML	CNN (custom from scratch)	Frame-level classification	~80%
DL	MobileNetV2 + LSTM	Clip-level video classification	~98%

Both models are saved as .keras files ready for loading.

🛠 Tools & Libraries
TensorFlow / Keras

OpenCV

Scikit-learn

NumPy, Pandas

Matplotlib

Jupyter Notebook / Google Colab

Microsoft Azure (for deployment exploration)

📊 Evaluation Metrics
Accuracy: Primary evaluation metric.

Confusion Matrix: For clear visualization of performance.

Precision & Recall: To handle imbalance between normal and violent samples.

Loss & Accuracy Curves: Tracked over training epochs.

🔮 Future Work
Real-time inference on CCTV streams.

Adding a separate "Weaponized" class in the DL model.

Deploying on Azure or local edge devices for faster, secure processing.

Expanding to more surveillance datasets.

🙏 Acknowledgements
Kaggle Dataset Providers.

Open-source community (TensorFlow, Keras, OpenCV).

