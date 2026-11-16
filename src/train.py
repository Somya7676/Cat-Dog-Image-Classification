
from pathlib import Path
import zipfile
import tensorflow as tf
import tensorflow_datasets as tfds

# Configuration
IMG_SIZE = (160, 160)
BATCH_SIZE = 32
EPOCHS = 5
SEED = 123

MODEL_PATH = Path("models/cat_dog_model.keras")


# Windows ZIP path compatibility fix
def load_dataset():
    print("Loading Cats vs Dogs dataset...")

    # Save original ZIP open method
    original_zip_open = zipfile.ZipFile.open

    # Normalize Windows backslashes to ZIP forward slashes
    def safe_zip_open(self, name, mode="r", pwd=None, *, force_zip64=False):
        if isinstance(name, str):
            name = name.replace("\\", "/")

        return original_zip_open(
            self,
            name,
            mode,
            pwd,
            force_zip64=force_zip64
        )

    # Apply temporary fix
    zipfile.ZipFile.open = safe_zip_open

    try:
        (train_ds, validation_ds), info = tfds.load(
            "cats_vs_dogs",
            split=["train[:80%]", "train[80%:]"],
            as_supervised=True,
            with_info=True,
            data_dir="data/tensorflow_datasets",
        )
    finally:
        # Restore original ZIP method
        zipfile.ZipFile.open = original_zip_open

    print("Dataset loaded successfully!")

    return train_ds, validation_ds, info


# Image preprocessing
def preprocess(image, label):

    image = tf.image.resize(image, IMG_SIZE)

    image = tf.cast(image, tf.float32)

    label = tf.cast(label, tf.float32)

    return image, label


# Main training function
def main():

    print("\n========== CAT & DOG CLASSIFICATION ==========")

    # Load dataset
    train_ds, validation_ds, info = load_dataset()

    print("\nPreparing training data...")

    # Training pipeline
    train_ds = (
        train_ds
        .map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
        .shuffle(1000, seed=SEED)
        .batch(BATCH_SIZE)
        .prefetch(tf.data.AUTOTUNE)
    )

    # Validation pipeline
    validation_ds = (
        validation_ds
        .map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
        .batch(BATCH_SIZE)
        .prefetch(tf.data.AUTOTUNE)
    )

    print("\nDataset preparation completed!")

    # Data augmentation
    data_augmentation = tf.keras.Sequential([

        tf.keras.layers.RandomFlip("horizontal"),

        tf.keras.layers.RandomRotation(0.08),

        tf.keras.layers.RandomZoom(0.1),

    ])

    # Load MobileNetV2 pretrained model
    print("\nLoading MobileNetV2 model...")

    base_model = tf.keras.applications.MobileNetV2(

        input_shape=IMG_SIZE + (3,),

        include_top=False,

        weights="imagenet"

    )

    # Freeze pretrained layers
    base_model.trainable = False

    # Build model
    inputs = tf.keras.Input(shape=IMG_SIZE + (3,))

    x = data_augmentation(inputs)

    x = tf.keras.applications.mobilenet_v2.preprocess_input(x)

    x = base_model(x, training=False)

    x = tf.keras.layers.GlobalAveragePooling2D()(x)

    x = tf.keras.layers.Dropout(0.2)(x)

    outputs = tf.keras.layers.Dense(
        1,
        activation="sigmoid"
    )(x)

    model = tf.keras.Model(inputs, outputs)

    # Compile model
    model.compile(

        optimizer=tf.keras.optimizers.Adam(
            learning_rate=0.001
        ),

        loss="binary_crossentropy",

        metrics=["accuracy"]

    )

    print("\nModel Summary:")

    model.summary()

    # Train model
    print("\n========== STARTING TRAINING ==========")

    history = model.fit(

        train_ds,

        validation_data=validation_ds,

        epochs=EPOCHS

    )

    # Save trained model
    MODEL_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    model.save(MODEL_PATH)

    print("\n========== TRAINING COMPLETED ==========")

    print(f"\nModel saved successfully: {MODEL_PATH}")

    # Final accuracy
    final_train_accuracy = history.history["accuracy"][-1]

    final_val_accuracy = history.history["val_accuracy"][-1]

    print(f"\nTraining Accuracy: {final_train_accuracy * 100:.2f}%")

    print(f"Validation Accuracy: {final_val_accuracy * 100:.2f}%")

    print("\nCat & Dog Classification Model is Ready!")


if __name__ == "__main__":

    main()
