import os
import shutil
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split

original_data_dir = r"C:\Users\This PC\Downloads\IntrnForte\Project 2\archive"
base_dir = r"C:\Users\This PC\Downloads\IntrnForte\Project 2\breast_cancer_data"
train_dir = os.path.join(base_dir, "train")
test_dir = os.path.join(base_dir, "test")

for category in ["benign", "malignant"]:
    os.makedirs(os.path.join(train_dir, category), exist_ok=True)
    os.makedirs(os.path.join(test_dir, category), exist_ok=True)

benign_images = []
malignant_images = []

for case_folder in os.listdir(original_data_dir):
    case_path = os.path.join(original_data_dir, case_folder)
    
    if os.path.isdir(case_path):
        benign_path = os.path.join(case_path, "0")
        malignant_path = os.path.join(case_path, "1")

        if os.path.exists(benign_path):
            benign_images.extend([os.path.join(benign_path, img) for img in os.listdir(benign_path)])

        if os.path.exists(malignant_path):
            malignant_images.extend([os.path.join(malignant_path, img) for img in os.listdir(malignant_path)])

train_benign, test_benign = train_test_split(benign_images, test_size=0.2, random_state=42)
train_malignant, test_malignant = train_test_split(malignant_images, test_size=0.2, random_state=42)

def move_files(file_list, destination_folder):
    for file_path in file_list:
        filename = os.path.basename(file_path)
        shutil.copy(file_path, os.path.join(destination_folder, filename))

move_files(train_benign, os.path.join(train_dir, "benign"))
move_files(test_benign, os.path.join(test_dir, "benign"))
move_files(train_malignant, os.path.join(train_dir, "malignant"))
move_files(test_malignant, os.path.join(test_dir, "malignant"))

train_data_dir = train_dir
test_data_dir = test_dir

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode="nearest"
)

test_datagen = ImageDataGenerator(rescale=1./255)

train_data = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(50, 50),
    batch_size=32,
    class_mode="binary"
)

test_data = test_datagen.flow_from_directory(
    test_data_dir,
    target_size=(50, 50),
    batch_size=32,
    class_mode="binary"
)

model = Sequential([
    Conv2D(32, (3, 3), activation="relu", input_shape=(50, 50, 3)),
    MaxPooling2D(pool_size=(2, 2)),

    Conv2D(64, (3, 3), activation="relu"),
    MaxPooling2D(pool_size=(2, 2)),

    Flatten(),
    Dense(128, activation="relu"),
    Dropout(0.5),
    Dense(1, activation="sigmoid")
])

model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

history = model.fit(
    train_data,
    epochs=10,
    validation_data=test_data
)

test_loss, test_accuracy = model.evaluate(test_data)

y_pred_prob = model.predict(test_data)
y_pred = (y_pred_prob > 0.5).astype(int)
y_true = test_data.classes

cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Benign", "Malignant"], yticklabels=["Benign", "Malignant"])
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.show()

accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)

plt.figure(figsize=(8, 6))
plt.plot(history.history["accuracy"], label="Train Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.title("Model Accuracy Over Epochs")
plt.show()