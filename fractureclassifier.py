import tensorflow as tf
import pandas as pd
import numpy as np
import os

# Define paths
train_data_dir = r'C:\Users\sburman\Downloads\MURA-v1.1\MURA-v1.1\train_data'
valid_data_dir = r'C:\Users\sburman\Downloads\MURA-v1.1\MURA-v1.1\valid_data'

# Function to get image paths and labels
def get_image_paths_and_labels(data_dir):
    image_paths = []
    labels = []
    for root, _, files in os.walk(data_dir):
        for file in files:
            if file.endswith('.png') or file.endswith('.jpg'):
                image_paths.append(os.path.join(root, file))
                labels.append(1 if 'positive' in file else 0)
    return image_paths, labels

# Get train and validation data
train_image_paths, train_labels = get_image_paths_and_labels(train_data_dir)
valid_image_paths, valid_labels = get_image_paths_and_labels(valid_data_dir)

# Preprocess images
def preprocess_image(image):
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [224, 224])
    image /= 255.0  # Normalize to [0,1] range
    return image

def load_and_preprocess_image(path):
    image = tf.io.read_file(path)
    return preprocess_image(image)

# Create datasets
def load_dataset(image_paths, labels):
    path_ds = tf.data.Dataset.from_tensor_slices(image_paths)
    image_ds = path_ds.map(load_and_preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
    label_ds = tf.data.Dataset.from_tensor_slices(tf.cast(labels, tf.int64))
    return tf.data.Dataset.zip((image_ds, label_ds))

train_dataset = load_dataset(train_image_paths, train_labels).repeat()
valid_dataset = load_dataset(valid_image_paths, valid_labels).repeat()

# Batch and prefetch
BATCH_SIZE = 32
train_dataset = train_dataset.batch(BATCH_SIZE).prefetch(buffer_size=tf.data.AUTOTUNE)
valid_dataset = valid_dataset.batch(BATCH_SIZE).prefetch(buffer_size=tf.data.AUTOTUNE)

# Define the model
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Calculate steps_per_epoch and validation_steps
steps_per_epoch = len(train_image_paths) // BATCH_SIZE
validation_steps = len(valid_image_paths) // BATCH_SIZE

# Define a callback to stop training when accuracy reaches 98%
class StopAtAccuracy(tf.keras.callbacks.Callback):
    def __init__(self, accuracy=0.98):
        super(StopAtAccuracy, self).__init__()
        self.accuracy = accuracy

    def on_epoch_end(self, epoch, logs=None):
        if logs.get('accuracy') >= self.accuracy:
            print(f"\nReached {self.accuracy*100}% accuracy, stopping training!")
            self.model.stop_training = True


# Train the model
history = model.fit(train_dataset,
                    epochs=2,
                    steps_per_epoch=10,#steps_per_epoch,
                    validation_data=valid_dataset,
                    validation_steps=validation_steps,
                    callbacks=[StopAtAccuracy(accuracy=0.98)])

# Save the model
model.save('fracture_classifier.h5')