Transfer Learning Image Classifier – TF Flowers Dataset

---
📌 Objective
The goal of this project is to build a high-performance image classifier using transfer learning with a pre-trained CNN (ResNet50).
This demonstrates end-to-end mastery of:

1 dataset preparation


2 preprocessing & augmentation

3 feature extraction

4 fine-tuning

5 evaluation & interpretability

6 baseline model comparison

This is a complete deep learning workflow commonly used in modern computer vision systems.
---
📂 Dataset
TF Flowers Dataset
Classes:
1 Daisy
2 Dandelion
3 Roses
4 Sunflowers
5 Tulips
Dataset was automatically downloaded using TensorFlow Datasets and split into:

split data
---
80% train
10% validation
10% test



Folder structure created:
---
data/train/
data/val/
data/test/

---
🧠 Model Architecture
1. Transfer Learning Model (ResNet50)
Base Model
Pretrained: ImageNet

Base layers frozen during Phase 1

Top 50 layers unfrozen during Phase 2

Input size: 224×224×3

Custom Classification Head
GlobalAveragePooling2D

Dense(256, activation='relu')

Dropout(0.3)

Dense(5, activation='softmax')

Training Strategy
---
Phase 1 — Feature Extraction

Freeze all ResNet50 layers

Train only dense classification head

Optimizer: Adam (LR = 1e-3)
---
Phase 2 — Fine-Tuning


Unfreeze top 50 ResNet50 layers

Train with very low LR = 1e-5

Prevent catastrophic forgetting

Improves feature alignment to flowers dataset

Callbacks
ModelCheckpoint → saves best_model.h5

EarlyStopping → prevents overfitting
---

📈 Baseline Model (Trained From Scratch)
A simple CNN with ~11M parameters:

3× Conv2D + MaxPooling

Flatten

Dense(128 → 5)

This provides a performance reference to compare the impact of transfer learning.
---
🧪 Evaluation Results
✔ Test Accuracy
Model	Test Accuracy
Baseline CNN	68%
Transfer Learning (ResNet50)	Higher accuracy (best_model.h5 saved)

Exact numbers depend on training run, but ResNet50 consistently outperforms baseline.
---
📊 Confusion Matrix


(Attached in visualizations/confusion_matrix.png)

Shows class-wise prediction distribution and helps identify misclassification patterns.
---
🔥 Grad-CAM Visualization


(Attached in visualizations/gradcam_example.png)

Highlights which parts of the image the model focuses on when making predictions.

This demonstrates model interpretability and verifies correct feature utilization.

📁 Project Structure
---

TL-image-classifier/
│
├── data/                      
│   ├── train/
│   ├── val/
│   └── test/
│---


├── models/
│   ├── best_model.h5
│   └── final_model.keras
│---


├── notebooks/
│   └── train_and_eval.ipynb
│---


├── scripts/
│   ├── download_dataset.py
│   └── split_dataset.py
│---


├── visualizations/
│   ├── confusion_matrix.png
│   └── gradcam_example.png
│---


├── README.md
├── requirements.txt
└── .gitignore

---

🚀 How to Run the Project
1. Install dependencies
pip install -r requirements.txt
2. Download and prepare dataset
python scripts/download_dataset.py
python scripts/split_dataset.py
3. Run training notebook
Open:
notebooks/train_and_eval.ipynb
Run all cells for:

preprocessing

feature extraction

fine-tuning

baseline model

evaluation

visualization

saving final model

4. Final model files
Located in:
---
models/best_model.h5
models/final_model.keras

---

🎯 Key Takeaways
Transfer learning significantly improves accuracy vs training from scratch.
Fine-tuning with small learning rates is essential for stability.
Grad-CAM reveals correct feature localization.
Proper dataset splits ensure fair evaluation.

This project demonstrates full ML engineering workflow.
---

📜 License
This project is for academic and learning purposes.




