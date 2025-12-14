import tensorflow as tf
from tensorflow.keras.preprocessing import image_dataset_from_directory # type: ignore
from tensorflow.keras.applications import MobileNetV2 # type: ignore
from tensorflow.keras import layers, models, callbacks # type: ignore

#chargement du dataset
img_size =  (224, 224)
batch_size = 32

train_ds = image_dataset_from_directory(
    "Dataset/train",
    image_size=img_size,
    batch_size=batch_size
)

val_ds = image_dataset_from_directory(
    "Dataset/val",
    image_size=img_size,
    batch_size=batch_size
)

test_ds = image_dataset_from_directory(
    "Dataset/test",
    image_size=img_size,
    batch_size=batch_size
)
class_names = train_ds.class_names

#Optimisation du pipeline
AUTOTUNE =tf.data.AUTOTUNE
train_ds = train_ds.shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)
test_ds = test_ds.prefetch(buffer_size=AUTOTUNE)

#Data augmentation 
data_augmentation = models.Sequential([
    layers.RandomFlip("horizontal_and_vertical"),
    layers.RandomRotation(0.2),
    layers.RandomZoom(0.2),
    layers.RandomContrast(0.2),
])

#chargement du modele MobileNetV2
base_model = MobileNetV2(
    input_shape = img_size +(3,),
    include_top=False,
    weights="imagenet"
)

#construction du modele
inputs = layers.Input(shape=img_size+ (3,))
x = data_augmentation(inputs)
x = layers.Rescaling(1./255)(x)
x = base_model(x, training=False)
x = layers.GlobalAveragePooling2D() (x)
x = layers.Dropout(0.4)(x)
outputs = layers.Dense(len(class_names), activation="softmax")(x)

model = models.Model(inputs, outputs)

#compilation du modele 
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]

)

#callbacks
callbacks_list = [
    callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
    callbacks.ModelCheckpoint("best_mobilenet.h5", save_best_only=True),
    callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.2, patience=3)

]

#entrainement 
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=15,
    callbacks=callbacks_list
)

#Fine-tuning
base_model.trainable= True
for layer in base_model.layers[:-30]:
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)
history_fine = model.fit(train_ds, validation_data=val_ds, epochs=10, callbacks=callbacks_list)

#evaluation 
test_loss, test_acc = model.evaluate(test_ds)
print(f"Test accuracy: {test_acc:.2f}")

#sauvgarde du modele
model.save("mobilenet_ravageurs_final.h5")

print("Classes détectées :", train_ds.class_names)

