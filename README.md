# doniarish-LUNG-CANCER-DETECTION-WITH-CNN
🔬 Lung & Colon Cancer Histopathological Image Classification
This project implements a Convolutional Neural Network (CNN) using PyTorch to classify histopathological images of lung tissue into three distinct categories:

Adenocarcinoma (malignant)

Squamous cell carcinoma (malignant)

Benign lung tissue (non-cancerous)

It utilizes a publicly available dataset from Kaggle, providing a baseline for digital pathology classification using deep learning.

🧠 Objective
The main objective of this project is to:

Explore deep learning techniques for medical image classification.

Build a CNN from scratch (without pre-trained models).

Understand model behavior by evaluating metrics, visualizations, and failure cases.

Provide a clear and reproducible PyTorch-based pipeline for academic use.

📦 Dataset Overview
Source: Kaggle - Lung and Colon Cancer Histopathological Images

Image size: Varies (resized to 224×224 during preprocessing)

Classes:

lung_aca: Adenocarcinoma (malignant)

lung_scc: Squamous cell carcinoma (malignant)

lung_n: Normal lung tissue (benign)

Format: RGB JPEG images

Split Strategy: Manual 70% train, 15% validation, 15% test split

Balance: Relatively balanced but varies slightly

🧬 Project Structure
graphql
Copy
Edit
.
├── data/                        # Dataset directory (after extraction)
│   └── lung_colon_image_set/
│       └── lung_image_sets/
│           ├── lung_aca/
│           ├── lung_scc/
│           └── lung_n/
├── src/                         # Source code
│   ├── dataset.py               # PyTorch Dataset and transforms
│   ├── model.py                 # CNN architecture
│   ├── train.py                 # Training pipeline
│   └── evaluate.py              # Model evaluation and metrics
├── notebooks/                   # Optional: Jupyter exploration
├── outputs/
│   ├── best_cnn_model.pth       # Saved best model
│   ├── confusion_matrix.png
│   └── pr_curve.png
├── README.md
├── requirements.txt
└── .gitignore
⚙️ Setup Instructions
1. Clone Repository
bash
Copy
Edit
git clone https://github.com/yourusername/lung-cancer-cnn
cd lung-cancer-cnn
2. Install Dependencies
bash
Copy
Edit
pip install -r requirements.txt
Contents of requirements.txt:

txt
Copy
Edit
torch
torchvision
numpy
matplotlib
scikit-learn
seaborn
pillow
kagglehub
3. Download Dataset
Use Kaggle API or kagglehub:

python
Copy
Edit
import kagglehub
kagglehub.dataset_download("andrewmvd/lung-and-colon-cancer-histopathological-images")
OR use CLI:

bash
Copy
Edit
kaggle datasets download -d andrewmvd/lung-and-colon-cancer-histopathological-images
unzip lung-and-colon-cancer-histopathological-images.zip -d data/
🛠️ Model Architecture
LungCancerCNN (src/model.py)
python
Copy
Edit
Conv2d(3, 32, kernel_size=3) → ReLU → MaxPool  
Conv2d(32, 64, kernel_size=3) → ReLU → MaxPool  
Conv2d(64, 128, kernel_size=3) → ReLU → MaxPool  
Flatten → Linear(128*28*28, 512) → ReLU → Dropout  
Linear(512, 128) → ReLU  
Linear(128, 3) → LogSoftmax  
Activation: ReLU

Output: Log-probabilities (log softmax) with 3 output neurons

🎓 Training Strategy
Run the training script:

bash
Copy
Edit
python src/train.py --data_dir data/lung_colon_image_set/lung_image_sets
Features:

Optimizer: torch.optim.SGD

LR scheduler: ReduceLROnPlateau

Epochs: 15

Batch size: 32

Best model saved as best_cnn_model.pth

🧪 Evaluation
To evaluate the model:

bash
Copy
Edit
python src/evaluate.py --model_path outputs/best_cnn_model.pth
Output:
Accuracy

Precision, Recall, F1 (per class)

Confusion matrix

PR & ROC curves

Misclassified image samples

📈 Results
Class	Precision	Recall	F1-score
Adenocarcinoma	0.94	0.89	0.91
Squamous Cell	0.90	0.94	0.92
Benign	0.99	1.00	0.99

Overall Accuracy: 94.22%

AUC (micro/macro): Reported via ROC curve

Model performs very well on benign cases; some confusion between cancer subtypes.

📸 Visualizations
🔁 Confusion Matrix

📉 Precision-Recall Curve

🤖 Misclassified Examples
Image	True Label	Predicted Label
📷	Squamous	Adeno
📷	Benign	Adeno

💡 Why This Matters
Lung cancer is one of the leading causes of cancer-related deaths globally.

Early and accurate classification of histopathology slides can improve diagnosis.

Deep learning automates tedious tasks for pathologists and aids decision-making.

🔧 Customization
Feature	How to Modify
Architecture	src/model.py
Augmentation	src/dataset.py
Training loop	src/train.py
Visualization	src/evaluate.py
Learning Rate	Pass --learning_rate as CLI arg

🔄 Future Work
✅ Apply Transfer Learning (e.g., ResNet50)

✅ Improve augmentation pipeline

⏳ Use Grad-CAM for model interpretability

⏳ Web UI for demo (Flask/Gradio)

⏳ Add experiment tracking with Weights & Biases or TensorBoard

⏳ Explore transformer-based vision models (ViT)

🧾 License
This project is licensed under the MIT License.

👋 Acknowledgements
Dataset: Andrew MVD on Kaggle

PyTorch Community

Medical experts and contributors

📫 Contact
For questions, suggestions, or collaborations:

GitHub: @yourusername

Email: doniarish1@gmail.com

🌟 LIKE THIS PROJECT? GIVE IT A ⭐ ON GITHUB!
