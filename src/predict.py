from pathlib import Path
import numpy as np
import tensorflow as tf
from PIL import Image

MODEL_PATH = Path("models/cat_dog_model.keras")
IMG_SIZE = (160, 160)

def predict_image(image_path):
    if not MODEL_PATH.exists():
        raise FileNotFoundError("Model missing. Run `python src/train.py` first.")

    model = tf.keras.models.load_model(MODEL_PATH)
    image = Image.open(image_path).convert("RGB").resize(IMG_SIZE)
    array = np.asarray(image, dtype=np.float32)[None, ...]
    dog_probability = float(model.predict(array, verbose=0)[0][0])

    if dog_probability >= 0.5:
        return "Dog", dog_probability
    return "Cat", 1.0 - dog_probability

if __name__ == "__main__":
    path = input("Enter image path: ").strip().strip('"')
    label, confidence = predict_image(path)
    print(f"Prediction: {label}")
    print(f"Confidence: {confidence * 100:.2f}%")
