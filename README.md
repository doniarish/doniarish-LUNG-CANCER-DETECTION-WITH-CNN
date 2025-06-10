# 🔬 Lung & Colon Cancer Histopathological Image Classification

This project implements a Convolutional Neural Network (CNN) using **PyTorch** to classify histopathological images of lung tissue into three distinct categories:

- **Adenocarcinoma** *(malignant)*
- **Squamous cell carcinoma** *(malignant)*
- **Benign lung tissue** *(non-cancerous)*

It utilizes a publicly available dataset from **Kaggle**, providing a strong baseline for digital pathology classification using deep learning.

---

## 🧠 Objective

The main objectives of this project are to:

- Explore deep learning techniques for medical image classification.
- Build a CNN from scratch *(without pre-trained models)*.
- Understand model behavior through metrics, visualizations, and failure cases.
- Provide a clear and reproducible PyTorch-based pipeline for academic and research use.

---

## 📦 Dataset Overview

- **Source:** Kaggle - [Lung and Colon Cancer Histopathological Images](https://www.kaggle.com/datasets/andrewmvd/lung-and-colon-cancer-histopathological-images)
- **Image Format:** RGB JPEG
- **Image Size:** Varies *(resized to 224×224)*
- **Classes:**
  - `lung_aca`: Adenocarcinoma *(malignant)*
  - `lung_scc`: Squamous cell carcinoma *(malignant)*
  - `lung_n`: Normal lung tissue *(benign)*
- **Split:** 70% train, 15% validation, 15% test *(manual split)*
- **Class Balance:** Relatively balanced, slight variations

---

## 🧬 Project Structure

```
.
├── data/
│   └── lung_colon_image_set/
│       └── lung_image_sets/
│           ├── lung_aca/
│           ├── lung_scc/
│           └── lung_n/
├── src/
│   ├── dataset.py         # PyTorch dataset and augmentations
│   ├── model.py           # CNN architecture
│   ├── train.py           # Training pipeline
│   └── evaluate.py        # Evaluation logic
├── outputs/
│   ├── best_cnn_model.pth
│   ├── confusion_matrix.png
│   └── pr_curve.png
├── notebooks/
├── requirements.txt
├── .gitignore
└── README.md
```

---

## ⚙️ Setup Instructions

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/lung-cancer-cnn
cd lung-cancer-cnn
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

**requirements.txt**
```
torch
torchvision
numpy
matplotlib
scikit-learn
seaborn
pillow
kagglehub
```

### 3. Download the dataset

Using Python:

```python
import kagglehub
kagglehub.dataset_download("andrewmvd/lung-and-colon-cancer-histopathological-images")
```

Or use CLI:

```bash
kaggle datasets download -d andrewmvd/lung-and-colon-cancer-histopathological-images
unzip lung-and-colon-cancer-histopathological-images.zip -d data/
```

---

## 🛠️ Model Architecture

Implemented in `src/model.py`:

```python
Conv2d(3, 32, kernel_size=3) → ReLU → MaxPool
Conv2d(32, 64, kernel_size=3) → ReLU → MaxPool
Conv2d(64, 128, kernel_size=3) → ReLU → MaxPool
Flatten → Linear(128*28*28, 512) → ReLU → Dropout
Linear(512, 128) → ReLU
Linear(128, 3) → LogSoftmax
```

- **Activation:** ReLU  
- **Output:** Log-probabilities for 3 classes

---

## 🎓 Training Strategy

Run the training script:

```bash
python src/train.py --data_dir data/lung_colon_image_set/lung_image_sets
```

**Features:**

- Optimizer: `torch.optim.SGD`
- LR Scheduler: `ReduceLROnPlateau`
- Epochs: 15
- Batch Size: 32
- Best model saved as: `outputs/best_cnn_model.pth`

---

## 🧪 Evaluation

Evaluate model performance:

```bash
python src/evaluate.py --model_path outputs/best_cnn_model.pth
```

**Outputs:**

- Accuracy
- Per-class Precision, Recall, F1
- Confusion Matrix
- PR and ROC curves
- Misclassified examples

---

## 📈 Results

| Class            | Precision | Recall | F1-Score |
|------------------|-----------|--------|----------|
| Adenocarcinoma   | 0.94      | 0.89   | 0.91     |
| Squamous Cell    | 0.90      | 0.94   | 0.92     |
| Benign           | 0.99      | 1.00   | 0.99     |

- **Overall Accuracy:** 94.22%
- **AUC (micro/macro):** Calculated via ROC curve

---

## 📸 Visualizations

- ✅ Confusion Matrix  
- ✅ Precision-Recall Curve  
- ✅ Misclassified Image Samples

---

## 💡 Why This Matters

- Lung cancer is a leading cause of cancer-related deaths globally.
- Accurate early detection via deep learning aids medical diagnosis.
- Automating classification supports pathologists in decision-making.

---

## 🔧 Customization Guide

| Feature        | Modify File     |
|----------------|-----------------|
| Architecture   | `src/model.py`  |
| Augmentations  | `src/dataset.py`|
| Training logic | `src/train.py`  |
| Evaluation     | `src/evaluate.py`|
| Learning Rate  | CLI: `--learning_rate`|

---

## 🔄 Future Work

- ✅ Apply **Transfer Learning** (e.g., ResNet50)  
- ✅ Improve data **augmentation** techniques  
- ⏳ Implement **Grad-CAM** for interpretability  
- ⏳ Add **Web UI** for demo (Flask or Gradio)  
- ⏳ Experiment tracking via **W&B** or **TensorBoard**  
- ⏳ Explore **Vision Transformers (ViT)**

---

## 🧾 License

This project is licensed under the [MIT License](LICENSE).

---

## 👋 Acknowledgements

- **Dataset:** [Andrew MVD on Kaggle](https://www.kaggle.com/datasets/andrewmvd/lung-and-colon-cancer-histopathological-images)
- **Framework:** PyTorch  
- **Support:** Medical experts and contributors

---

## 📫 Contact

- **GitHub:** [@yourusername](https://github.com/yourusername)  
- **Email:** doniarish1@gmail.com

---

## 🌟 Like this project?

Give it a ⭐ on GitHub!
