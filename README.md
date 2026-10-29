# 🥔 Potato Leaf Disease Detection

A deep learning project that classifies potato leaf images into three classes using a CNN built with TensorFlow/Keras, with a Streamlit web app for predictions.

| Class | Description |
|---|---|
| `Potato___Early_blight` | Caused by *Alternaria solani* |
| `Potato___Late_blight` | Caused by *Phytophthora infestans* |
| `Potato___healthy` | No disease detected |

## Project Structure

```
Potato_leaf_disease_Detection/
├── Dataset/          # Train / Valid / Test images (not included in repo)
├── Model/
│   └── train_potato_disease.ipynb   # model training notebook
├── Ui/
│   ├── app.py        # Streamlit web app
│   └── diseases.jpg  # banner image
├── requirements.txt
└── README.md
```

## Dataset

The dataset is not included because of its size. Download the potato leaf images (PlantVillage potato subset, e.g. from Kaggle) and arrange them as:

```
Dataset/
├── Train/  (Potato___Early_blight, Potato___Late_blight, Potato___healthy)
├── Valid/  (same 3 folders)
└── Test/   (same 3 folders)
```

## Model

- Input: 128 × 128 RGB images
- Architecture: Conv2D (32) → Conv2D (32) → MaxPooling → Dropout → Dense (512) → Dropout → Softmax (3 classes)
- Optimizer: Adam (lr = 0.0001), loss: categorical crossentropy, 10 epochs
- Accuracy:  Training accuracy 99.22%, Validation accuracy 90.67%, Test accuracy 93.00%

The trained model (`trained_plant_disease_model.keras`, ~744 MB) is too large for GitHub. The app **downloads it automatically from Google Drive** the first time it runs.

## Run Locally

```bash
git clone(https://github.com/Shailesh-Pawar-18/Potato-Leaf-Disease-Detection)
pip install -r requirements.txt
streamlit run Ui/app.py
```

Open the link shown in the terminal (usually http://localhost:8501), go to **Disease Recognition**, upload a leaf image and click **Predict**.

## Retrain the Model

1. Place the dataset in `Dataset/` as shown above.
2. Open `Model/train_potato_disease.ipynb` in Jupyter and run all cells.
3. The new model is saved as `Model/trained_plant_disease_model.keras`.

## Tech Stack

Python, TensorFlow/Keras, Streamlit, NumPy, Pillow

## Author

Shailesh Pawar — Data Science internship project
