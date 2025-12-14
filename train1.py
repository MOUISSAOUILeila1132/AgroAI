import os
import seaborn as sns
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras.preprocessing.image import ImageDataGenerator # type: ignore
from tensorflow.keras.applications import MobileNetV2 # type: ignore
from tensorflow.keras import layers, models, Input # type: ignore

# Chemin vers ton dataset
dataset_path = "Dataset\dataset_for_train"

# Catégories
categories = [
    "Adristyrannus", "Aphids", "Beetle", "Bugs", "Cabbage Looper","Peach___Bacterial_spot","Peach___healthy","Pepper,_bell___Bacterial_spot","Pepper,_bell___healthy","Potato___Early_blight","Potato___healthy","Potato___Late_blight","Raspberry___healthy",
    "Cicadellidae", "Cutworm", "Earwig", "FieldCricket", "Grasshopper","Soybean___healthy","Squash___Powdery_mildew","Strawberry___healthy","Strawberry___Leaf_scorch","Tomato___Bacterial_spot","Tomato___Early_blight","Tomato___healthy","Tomato___Late_blight",
    "Tomato___Leaf_Mold","Tomato___Septoria_leaf_spot","Tomato___Spider_mites Two-spotted_spider_mite","Tomato___Target_Spot","Tomato___Tomato_mosaic_virus","Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Mediterranean fruit fly", "Mites", "RedSpider", "Riptortus", "Slug","Grape___Black_rot","Grape___Esca_(Black_Measles)","Grape___healthy","Grape___Leaf_blight_(Isariopsis_Leaf_Spot)","Orange___Haunglongbing_(Citrus_greening)" ,
    "Snail", "Thrips", "Weevil", "Whitefly","Apple___Apple_scab","Apple___Black_rot","Apple___Cedar_apple_rust","Apple___healthy","Background_without_leaves","Blueberry___healthy","Cherry___healthy","Cherry___Powdery_mildew","Corn___Cercospora_leaf_spot Gray_leaf_spot","Corn___Common_rust","Corn___healthy","Corn___Northern_Leaf_Blight"
]
num_classes = len(categories)

# Générateurs avec augmentation pour l'entraînement et normalisation pour test
datagen_train = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode="nearest"
)

datagen_test = ImageDataGenerator(rescale=1./255)

# Chargement des données depuis les dossiers
train_generator = datagen_train.flow_from_directory(
    os.path.join(dataset_path, "train"),
    target_size=(224, 224),
    batch_size=32,
    class_mode="categorical",
    shuffle=True
)

test_generator = datagen_test.flow_from_directory(
    os.path.join(dataset_path, "test"),
    target_size=(224, 224),
    batch_size=32,
    class_mode="categorical",
    shuffle=False
)

# Création du modèle MobileNetV2
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False

model_input = Input(shape=(224, 224, 3))
x = base_model(model_input, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.5)(x)
x = layers.Dense(512, activation='relu')(x)
output_layer = layers.Dense(num_classes, activation='softmax')(x)

model = models.Model(inputs=model_input, outputs=output_layer)
model.summary()

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# Entraînement
history = model.fit(
    train_generator,
    validation_data=None,  # plus de val
    epochs=20
)

# Sauvegarde directe dans le répertoire courant
model.save("mobilenetv2_agriculture.keras")
print("✅ Modèle sauvegardé sous mobilenetv2_agriculture.keras")

# Évaluation sur le jeu de test
test_loss, test_accuracy = model.evaluate(test_generator, verbose=1)
print(f"Perte sur le jeu de test: {test_loss:.4f}")
print(f"Précision sur le jeu de test: {test_accuracy:.4f}")

# Prédictions et matrice de confusion
y_true = test_generator.classes
y_pred = model.predict(test_generator, verbose=1)
y_pred_classes = y_pred.argmax(axis=1)

conf_matrix = confusion_matrix(y_true, y_pred_classes)
plt.figure(figsize=(12, 10))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=categories, yticklabels=categories)
plt.xlabel("Étiquette Prédite")
plt.ylabel("Étiquette Réelle")
plt.title("Matrice de Confusion")
plt.show()

print("Rapport de Classification:")
print(classification_report(y_true, y_pred_classes, target_names=categories))
