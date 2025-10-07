# 🎵 Music Classification (Inference Only)

This repository provides **two inference pipelines** for music artist classification — using both **traditional Machine Learning (SVM)** and **Deep Learning (CNN)**.  
Both models are fully trained; you can directly run inference to reproduce classification results.

---

## 📦 Environment Setup

Recreate the environment using the included YAML file:
```bash
conda env create -f music_env.yaml
conda activate music
Or install manually:


conda create -n music python=3.10
conda activate music
pip install torch torchaudio librosa scikit-learn matplotlib tqdm numpy
🧩 Project Structure

music-classification/
├── CNN/                      # Deep learning inference (CNN)
│   ├── inference_CNN.py
│   ├── best_model_cnn.pth
│   ├── r14942087.json
│   └── val_CNN.png
│
├── ML_code/                  # Traditional ML inference (SVM)
│   ├── inference.py
│   ├── artist20_svm.pkl
│   ├── artist20_scaler.pkl
│   ├── ML_round2.json
│   └── val_confusion_matrix.png
│
├── music_env.yaml            # Conda environment file
├── .gitignore
├── LICENSE
└── README.md
🚀 Inference Guide
🧠 Deep Learning (CNN)

cd CNN
python inference_CNN.py
📤 Output files

r14942087.json → top-3 predictions per track

val_CNN.png → validation confusion matrix

Model: best_model_cnn.pth

🧩 Machine Learning (SVM)

cd ML_code
python inference.py
📤 Output files

ML_round2.json → predicted top-3 artists

val_confusion_matrix.png → confusion matrix

Models: artist20_svm.pkl, artist20_scaler.pkl

⚙️ Notes
No training or fine-tuning required — both models are inference-ready.

File paths and configs are preset for the included dataset.

The provided music_env.yaml ensures full reproducibility.

📜 License
This project is released under the MIT License.
