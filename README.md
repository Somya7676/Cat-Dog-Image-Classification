# Cat & Dog Image Classification

A beginner-friendly Deep Learning project using **TensorFlow + MobileNetV2 transfer learning + Streamlit**.

## 1. Requirements
- Python 3.10 or 3.11 recommended
- VS Code
- Internet connection for the first dataset download and model weights

## 2. Open in VS Code
Extract the ZIP, open the `cat_dog_classifier_project` folder in VS Code, then open Terminal.

### Windows PowerShell setup
```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 3. Train the model
The first training run downloads the Cats vs Dogs dataset through TensorFlow Datasets. It is a large download, so keep a stable internet connection and enough disk space.

```powershell
python src/train.py
```

The trained model is saved to `models/cat_dog_model.keras`.

## 4. Run the web app
```powershell
streamlit run app.py
```
Upload a JPG, JPEG or PNG image. The app predicts **Cat** or **Dog** and displays confidence.

## Project structure
```text
cat_dog_classifier_project/
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
├── models/
│   └── .gitkeep
├── src/
│   ├── train.py
│   └── predict.py
└── sample_images/
    └── README.txt
```

## Important
- The model is not included in this ZIP; it is created after training.
- Dataset download and training can take time.
- This is an educational classifier, not a guaranteed animal-identification system.
