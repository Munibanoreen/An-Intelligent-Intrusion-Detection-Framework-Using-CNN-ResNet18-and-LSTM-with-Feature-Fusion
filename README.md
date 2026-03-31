# An-Intelligent-Intrusion-Detection-Framework-Using-CNN-ResNet18-and-LSTM-with-Feature-Fusion
Hybrid CNN-ResNet18 + BiLSTM intrusion detection framework with feature fusion for CIC-IDS2017 network traffic classification.
# 🚀 An Intelligent Intrusion Detection Framework Using CNN-ResNet18 and LSTM with Feature Fusion

## 📌 Overview
This repository presents a deep learning-based Intrusion Detection System (IDS) that combines **CNN (ResNet18)** and **BiLSTM** using a feature fusion strategy. The model is designed to detect network intrusions from structured traffic data (CIC-IDS2017 dataset).

The framework leverages:
- CNN (ResNet18) → spatial feature extraction
- BiLSTM → temporal dependency learning
- Feature Fusion → combines both representations for improved accuracy

---

## 🧠 Architecture
1. Input Features
2. ResNet18 Branch (Feature Extraction)
3. BiLSTM Branch (Sequential Learning)
4. Feature Fusion Layer
5. Fully Connected Classifier
6. Output (Benign vs Attack)

---
## Proposed Architecture
![Proposed Architecture](Image 1, image 2, image 3)

## 📊 Dataset
We use **CIC-IDS2017 dataset**.

⚠️ Dataset is NOT included in this repo due to size.

Download from:"https://www.unb.ca/cic/datasets/index.html"
https://www.unb.ca/cic/datasets/ids-2017.html

## 🚀 How to Run

### 1. Install Requirements and next train the model and test the model
```bash
pip install -r requirements.txt
python train.py --data_dir data/cicids2017 --output_dir artifacts --epochs 5 --batch_size 512
python test.py --data_dir data/cicids2017 --artifact_dir artifacts --output_dir results
---

cd your_repo
pip install -r requirements.txt
python train.py --data_dir data/cicids2017
python test.py --data_dir data/cicids2017
# 📂 Where files should be placed

Your repo structure should look like this:

```bash
your_repo/
│── train.py
│── test.py
│── dataset.py
│── requirements.txt
│── README.md
