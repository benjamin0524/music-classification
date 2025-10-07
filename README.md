# 🎵 Music Classification (Inference Only)

This repository provides **two inference pipelines** for music artist classification:
1. 🎼 **Traditional Machine Learning (SVM)** — feature-based classification  
2. 🎧 **Deep Learning (CNN)** — spectrogram-based classification  

Both models are already trained.  
You only need to run the inference scripts to reproduce the classification results.

---

## 📦 Environment Setup

You can directly recreate the environment using the included YAML file:

```bash
conda env create -f music_env.yaml
conda activate music
If you prefer to install manually:

bash
複製程式碼
conda create -n music python=3.10
conda activate music
pip install torch torchaudio librosa scikit-learn matplotlib tqdm numpy
🧩 Project Structure
bash
複製程式碼
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
🚀 How to Run
🧠 Deep Learning (CNN)
bash
複製程式碼
cd CNN
python inference_CNN.py
Expected output:

r14942087.json → top-3 predictions for each test track

val_CNN.png → validation confusion matrix visualization

Uses pretrained model: best_model_cnn.pth

🧩 Machine Learning (SVM)
bash
複製程式碼
cd ML_code
python inference.py
Expected output:

ML_round2.json → predicted top-3 artists

val_confusion_matrix.png → confusion matrix visualization

Uses models: artist20_svm.pkl, artist20_scaler.pkl

⚙️ Notes
No fine-tuning or training is needed — both models are inference-ready.

All file paths and settings are preconfigured for the included dataset structure.

The conda environment ensures reproducibility across systems.

📜 License
This project is released under the MIT License.

🧑‍💻 Author: benjamin0524
Department of Electrical Engineering, National Taiwan University
Project: Music Artist Classification using ML & CNN

yaml
複製程式碼

---

你可以直接把上面這段內容整份貼進 `README.md`（取代舊的）。  
它會完整描述：
- 兩種模型架構
- YAML 安裝方式
- 專案目錄結構
- 推論方法  

要我幫你生成兩個子資料夾（`CNN/README.md`、`ML_code/README.md`）的版本也一起補上嗎？






你說：
給我一件複製能貼上的


ChatGPT 說：
當然可以 👍
這裡是一份你可以直接複製貼上到 GitHub 的 README.md（完整 markdown，可直接用）：

markdown
複製程式碼
# 🎵 Music Classification (Inference Only)

This repository provides **two inference pipelines** for music artist classification:
1. 🎼 **Traditional Machine Learning (SVM)** — feature-based classification  
2. 🎧 **Deep Learning (CNN)** — spectrogram-based classification  

Both models are already trained.  
You only need to run the inference scripts to reproduce the classification results.

---

## 📦 Environment Setup

You can directly recreate the environment using the included YAML file:

```bash
conda env create -f music_env.yaml
conda activate music
If you prefer to install manually:

bash
複製程式碼
conda create -n music python=3.10
conda activate music
pip install torch torchaudio librosa scikit-learn matplotlib tqdm numpy
🧩 Project Structure
bash
複製程式碼
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
🚀 How to Run
🧠 Deep Learning (CNN)
bash
複製程式碼
cd CNN
python inference_CNN.py
Expected output:

r14942087.json → top-3 predictions for each test track

val_CNN.png → validation confusion matrix visualization

Uses pretrained model: best_model_cnn.pth

🧩 Machine Learning (SVM)
bash
複製程式碼
cd ML_code
python inference.py
Expected output:

ML_round2.json → predicted top-3 artists

val_confusion_matrix.png → confusion matrix visualization

Uses models: artist20_svm.pkl, artist20_scaler.pkl

⚙️ Notes
No fine-tuning or training is needed — both models are inference-ready.

All file paths and settings are preconfigured for the included dataset structure.

The conda environment ensures reproducibility across systems.

📜 License
This project is released under the MIT License.
