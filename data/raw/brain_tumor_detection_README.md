# 🧠 Healthcare AI — Brain Tumor Detection & Segmentation

A complete deep learning pipeline that **detects** and **localizes** brain tumors in MRI scans, using ResNet50 for classification and a custom ResUNet for pixel-level segmentation.

---

## 📋 Table of Contents
- [Project Overview](#-project-overview)
- [Pipeline Architecture](#-pipeline-architecture)
- [Results](#-results)
- [Project Structure](#-project-structure)
- [Local Setup: Step-by-Step Guide](#-local-setup-step-by-step-guide)
- [Dataset Download](#-dataset-download)
- [Running the Notebook](#-running-the-notebook)
- [Key Concepts](#-key-concepts)
- [Technologies Used](#-technologies-used)
- [Acknowledgements](#-acknowledgements)

---

## 🎯 Project Overview

This project simulates a real-world scenario where an AI/ML consultant is hired by a medical diagnosis company to improve the speed and accuracy of detecting and localizing brain tumors from MRI scans.

**Why this matters:**
- The AI healthcare market is projected to grow from **$4.9B → $45.2B USD by 2026**
- Early and accurate tumor detection is **life-saving**
- Deep learning outperforms traditional methods in medical image analysis

---

## 🏗 Pipeline Architecture

```
Brain MRI Scan
      │
      ▼
┌─────────────────────┐
│  ResNet50 Classifier │  ← Transfer Learning (ImageNet weights)
│  (Tumor? Yes / No)  │
└─────────────────────┘
      │
   Tumor Detected?
      │
   YES ──────────────▶ ┌──────────────────────────┐
                        │  ResUNet Segmentation     │
                        │  (Pixel-level localization)│
                        └──────────────────────────┘
                                    │
                                    ▼
                           Tumor Location Map
      │
   NO ──────────────▶  Healthy Patient ✅
```

---

## 📊 Results

| Stage | Model | Metric | Score |
|-------|-------|--------|-------|
| Classification | ResNet50 + Dense Head | Accuracy | **~98%** |
| Classification | ResNet50 + Dense Head | Precision / Recall / F1 | **0.98 / 0.96 / 0.97** |
| Segmentation | ResUNet | Tversky Score | **~0.88** |

---

## 📁 Project Structure

```
healthcare-ai-brain-tumor/
│
├── Healthcare_AI_Brain_Tumor.ipynb   ← Main Jupyter Notebook
│
├── data/
│   └── lgg-mri-segmentation/
│       └── kaggle_3m/
│           ├── TCGA_CS_xxxx/         ← Patient folders (from Kaggle)
│           │   ├── image_001.tif
│           │   └── image_001_mask.tif
│           └── ...
│
├── models/                           ← Auto-created during training
│   ├── classifier-resnet-model.json
│   ├── classifier-resnet-weights.hdf5
│   ├── ResUNet-model.json
│   └── ResUNet-weights.hdf5
│
├── outputs/                          ← Auto-created — saved plots/results
│   ├── sample_mri_visualization.png
│   ├── classifier_training_curves.png
│   ├── confusion_matrix.png
│   ├── resunet_training_curves.png
│   └── segmentation_predictions.png
│
├── requirements.txt                  ← All Python dependencies
└── README.md
```

---

## 💻 Local Setup: Step-by-Step Guide

### Prerequisites
- Python **3.8 – 3.10** (TensorFlow 2.x is not yet compatible with Python 3.11+)
- At least **8 GB RAM** (16 GB recommended)
- **GPU is optional** but speeds up training significantly

---

### Step 1 — Install Python

Check your Python version:
```bash
python --version
```

If you don't have Python 3.8–3.10, download it from [python.org](https://www.python.org/downloads/).

---

### Step 2 — Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/healthcare-ai-brain-tumor.git
cd healthcare-ai-brain-tumor
```

> Replace `YOUR_USERNAME` with your actual GitHub username after uploading.

---

### Step 3 — Create a Virtual Environment

Using a virtual environment keeps dependencies isolated from your system Python.

**On Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**On macOS / Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

You should see `(venv)` appear in your terminal prompt.

---

### Step 4 — Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

This installs all required packages including TensorFlow, OpenCV, Plotly, and Jupyter.

> **Note for GPU users:** The `requirements.txt` installs CPU TensorFlow by default. If you have an NVIDIA GPU with CUDA 11.2+, replace `tensorflow` with `tensorflow-gpu` in `requirements.txt` before running pip install.

---

### Step 5 — Download the Dataset

1. Go to [Kaggle: LGG MRI Segmentation](https://www.kaggle.com/mateuszbuda/lgg-mri-segmentation)
2. Click **Download** (you need a free Kaggle account)
3. Extract the downloaded zip file
4. Place the extracted folder so your structure looks like this:

```
healthcare-ai-brain-tumor/
└── data/
    └── lgg-mri-segmentation/
        └── kaggle_3m/
            ├── TCGA_CS_4941_19960909/
            ├── TCGA_CS_4943_20000902/
            └── ... (110 patient folders total)
```

**Alternative — Kaggle API (faster):**
```bash
pip install kaggle
# Place your kaggle.json API token in ~/.kaggle/kaggle.json
kaggle datasets download -d mateuszbuda/lgg-mri-segmentation -p data/
cd data && unzip lgg-mri-segmentation.zip
```

---

### Step 6 — Launch Jupyter Notebook

```bash
jupyter notebook
```

Your browser will open automatically at `http://localhost:8888`.  
Click on **`Healthcare_AI_Brain_Tumor.ipynb`** to open the notebook.

---

### Step 7 — Run the Notebook

Run cells in order by pressing **`Shift + Enter`** or click **Cell → Run All**.

The notebook will:
- Load and explore the dataset
- Visualize MRI scans with tumor masks
- Train the ResNet50 classifier (~20 minutes on CPU, ~3 min on GPU)
- Evaluate classifier performance (confusion matrix, classification report)
- Build and train the ResUNet segmentation model (~30 minutes on CPU)
- Visualize segmentation predictions vs ground truth

---

## 📥 Dataset Download

| Item | Details |
|------|---------|
| Name | LGG MRI Segmentation |
| Source | [Kaggle](https://www.kaggle.com/mateuszbuda/lgg-mri-segmentation) |
| Size | ~740 MB |
| Patients | 110 |
| Images | 3,929 MRI scans |
| Format | `.tif` image + `_mask.tif` for each scan |

---

## 🔑 Key Concepts

### Image Segmentation
The goal is to assign a class label to **every pixel** in an image. The model output is a binary mask where white pixels = tumor, black pixels = healthy tissue.

### ResNet (Residual Network)
- Solves the **vanishing gradient problem** in deep networks via **skip connections**
- Trained on ImageNet (11M images) — weights are reused via **Transfer Learning**

### Transfer Learning
- Reuse a model trained on a large dataset (ImageNet) as a starting point
- Only train the new classification head; freeze the feature extraction layers
- Dramatically reduces training time and data requirements

### ResUNet
- Combines **UNet** (encoder-decoder with skip connections) + **Residual blocks**
- Encoder: progressively downsamples → learns what features exist
- Decoder: progressively upsamples → learns where features are
- Skip connections: preserve spatial information lost during downsampling

### Focal Tversky Loss
A custom loss function for segmentation that:
- Handles **class imbalance** (tumors are small vs. whole image)
- Penalizes **false negatives** more heavily (missing a tumor is worse than a false alarm)

---

## 🛠 Technologies Used

| Library | Version | Purpose |
|---------|---------|---------|
| TensorFlow / Keras | 2.x | Deep learning framework |
| OpenCV (`cv2`) | 4.x | Image loading and preprocessing |
| NumPy | 1.x | Array operations |
| Pandas | 1.x | Dataset management |
| Matplotlib | 3.x | Static visualizations |
| Seaborn | 0.x | Confusion matrix heatmap |
| Plotly | 5.x | Interactive bar charts |
| scikit-learn | 1.x | Train/test split, metrics |
| Jupyter | latest | Interactive notebook environment |

---

## ⚠️ Troubleshooting

**Problem: `ModuleNotFoundError: No module named 'cv2'`**
```bash
pip install opencv-python-headless
```

**Problem: TensorFlow not finding GPU**
- Ensure CUDA 11.2 and cuDNN 8.1 are installed
- Run `nvidia-smi` to verify GPU is recognized

**Problem: Kernel dies / Out of Memory**
- Reduce `batch_size` from 16 to 8 in the notebook
- Close other applications to free RAM

**Problem: `FileNotFoundError` for dataset**
- Verify the data folder path matches the structure shown in the Project Structure section

**Problem: Jupyter not opening in browser**
```bash
jupyter notebook --no-browser
# Then manually open: http://localhost:8888
```

---

## 🚀 Uploading to GitHub

```bash
# Initialize git (if not already done)
git init

# Add all files
git add .

# Create initial commit
git commit -m "Initial commit: Healthcare AI Brain Tumor Detection"

# Add your GitHub remote (create repo on GitHub first)
git remote add origin https://github.com/YOUR_USERNAME/healthcare-ai-brain-tumor.git

# Push to GitHub
git branch -M main
git push -u origin main
```

> **Important:** The `data/` folder is excluded via `.gitignore` because it is ~740 MB. Users must download the dataset separately from Kaggle.
> **Important:** Notebook contains the comments which will help you understand my project

---

## 👨‍💻 Author

**PRASHANT KUMAR**
- 📧 Email: pj.prashant95@gmail.com
- 💼 LinkedIn: www.linkedin.com/in/prashant-kumar-a5a616298
- 🐙 GitHub: https://github.com/PROGRAMMER-1008

> *This project was independently developed by me as part of my AI/ML portfolio.*

---
## 📚 Acknowledgements

- **Dataset:** [Mateusz Buda — LGG MRI Segmentation on Kaggle](https://www.kaggle.com/mateuszbuda/lgg-mri-segmentation)
- **ResUNet paper:** [ResUNet-a: A deep learning framework for semantic segmentation (arXiv)](https://arxiv.org/abs/1904.00592)
- **UNet paper:** [U-Net: Convolutional Networks for Biomedical Image Segmentation (arXiv)](https://arxiv.org/abs/1505.04597)
- **Focal Tversky Loss:** [Abraham & Khan, 2018 (arXiv:1810.07842)](https://arxiv.org/abs/1810.07842)
- **ResNet paper:** [Deep Residual Learning for Image Recognition (arXiv)](https://arxiv.org/pdf/1512.03385.pdf)

---

## 📄 License & Ownership

© 2025 **PRASHANT KUMAR**. All rights reserved.

This project was created and is maintained by **PRASHANT KUMAR**.
Dataset credit goes to Kaggle (see Acknowledgements).
