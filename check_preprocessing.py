import tensorflow as tf
import matplotlib.pyplot as plt
import os
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# --- CONFIG ---
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
DATASET_PATH = "Dataset" 

# --- CHARGEMENT DATASETS ---
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    os.path.join(DATASET_PATH, "train"),
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    os.path.join(DATASET_PATH, "val"),
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

test_ds = tf.keras.preprocessing.image_dataset_from_directory(
    os.path.join(DATASET_PATH, "test"),
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

class_names = train_ds.class_names
print("Classes :", class_names)

# --- FONCTION DE PRÉTRAITEMENT ---
def preprocess_batch(batch_images):
    return preprocess_input(batch_images)  # Normalisation spécifique MobileNetV2

# --- VÉRIFICATION ---
def check_dataset(dataset, name):
    print(f"\n--- Vérification dataset : {name} ---")
    for batch_images, batch_labels in dataset.take(1):
        print("Batch shape :", batch_images.shape)
        print("Labels :", batch_labels.numpy())
        print("Classes :", [class_names[i] for i in batch_labels.numpy()])
        print("Valeurs min :", tf.reduce_min(batch_images).numpy())
        print("Valeurs max :", tf.reduce_max(batch_images).numpy())

        # Prétraitement
        preprocessed = preprocess_batch(batch_images)
        print("Après preprocess_input → min :", tf.reduce_min(preprocessed).numpy(),
              "max :", tf.reduce_max(preprocessed).numpy())
        
# Affichage des images
        plt.figure(figsize=(10, 10))
        for i in range(min(9, batch_images.shape[0])):
            ax = plt.subplot(3, 3, i + 1)
            plt.imshow(batch_images[i].numpy().astype("uint8"))
            plt.title(class_names[batch_labels[i].numpy()])
            plt.axis("off")
        plt.show()
# --- CHECK TRAIN / VAL / TEST ---
check_dataset(train_ds, "Train")
check_dataset(val_ds, "Validation")
check_dataset(test_ds, "Test")

# --- VÉRIFIER L’ÉQUILIBRE DES CLASSES ---
print("\n--- Vérification équilibre des classes ---")
for split in ["train", "val", "test"]:
    print(f"\nDataset {split} :")
    split_path = os.path.join(DATASET_PATH, split)
    for class_name in os.listdir(split_path):
        class_path = os.path.join(split_path, class_name)
        if os.path.isdir(class_path):
            print(f"  {class_name} : {len(os.listdir(class_path))} images")