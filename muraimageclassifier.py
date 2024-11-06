from email import header
import tensorflow as tf
import pandas as pd
import numpy as np

# Load data paths from CSV files with headers and declare column names
train_image_paths_file = r'C:\Users\sburman\Downloads\MURA-v1.1\MURA-v1.1\train_image_paths.csv'
valid_image_paths_file = r'C:\Users\sburman\Downloads\MURA-v1.1\MURA-v1.1\valid_image_paths.csv'
train_labeled_studies_file = r'C:\Users\sburman\Downloads\MURA-v1.1\MURA-v1.1\train_labeled_studies.csv'
valid_labeled_studies_file = r'C:\Users\sburman\Downloads\MURA-v1.1\MURA-v1.1\valid_labeled_studies.csv'

# Load data paths from CSV files with headers and declare column names
train_image_paths = pd.DataFrame(pd.read_csv(train_image_paths_file), columns=['path'])
valid_image_paths = pd.DataFrame(pd.read_csv(valid_image_paths_file), columns=['path'])
train_labeled_studies = pd.DataFrame(pd.read_csv(train_labeled_studies_file, delimiter='\t'), columns=['path', 'label'])
valid_labeled_studies = pd.DataFrame(pd.read_csv(valid_labeled_studies_file, delimiter='\t'), columns=['path', 'label'])


# Ensure the paths in train_image_paths and train_labeled_studies match
train_image_paths['path'] = train_image_paths['path'].astype(str).str.strip()
train_labeled_studies['path'] = train_labeled_studies['path'].astype(str).str.strip()

# Ensure the paths in valid_image_paths and valid_labeled_studies match
valid_image_paths['path'] = valid_image_paths['path'].astype(str).str.strip()
valid_labeled_studies['path'] = valid_labeled_studies['path'].astype(str).str.strip()

# Merge image paths with labels
train_data = pd.merge(train_image_paths, train_labeled_studies, on='path', how='inner')
valid_data = pd.merge(valid_image_paths, valid_labeled_studies, on='path', how='inner')

# Check for NaN values
print("NaN values in train_data:", train_data.isna().sum().sum())
print("NaN values in valid_data:", valid_data.isna().sum().sum())

# Check for infinite values in numeric columns
train_data_numeric = train_data.select_dtypes(include=[np.number])
valid_data_numeric = valid_data.select_dtypes(include=[np.number])
print("Infinite values in train_data:", np.isinf(train_data_numeric).sum().sum())
print("Infinite values in valid_data:", np.isinf(valid_data_numeric).sum().sum())

# Preprocess images
def preprocess_image(image):
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [224, 224])
    image /= 255.0  # Normalize to [0,1] range
    return image

def load_and_preprocess_image(path):
    image = tf.io.read_file(path)
    return preprocess_image(image)

# Data augmentation
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip('horizontal'),
    tf.keras.layers.RandomRotation(0.2),
    tf.keras.layers.RandomZoom(0.2),
])

# Load dataset
def load_dataset(image_paths, labels):
    path_ds = tf.data.Dataset.from_tensor_slices(image_paths)
    image_ds = path_ds.map(load_and_preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
    label_ds = tf.data.Dataset.from_tensor_slices(tf.cast(labels, tf.int64))
    return tf.data.Dataset.zip((image_ds, label_ds))

train_labels = train_data['label'].values
valid_labels = valid_data['label'].values

train_dataset = load_dataset(train_data['path'].values, train_labels).repeat()
valid_dataset = load_dataset(valid_data['path'].values, valid_labels).repeat()

# Batch and prefetch
BATCH_SIZE = 32
train_dataset = train_dataset.batch(BATCH_SIZE).prefetch(buffer_size=tf.data.AUTOTUNE)
valid_dataset = valid_dataset.batch(BATCH_SIZE).prefetch(buffer_size=tf.data.AUTOTUNE)

# Debugging: Print dataset shapes and sample values
print("Train dataset shape:", train_data.shape)
print("Validation dataset shape:", valid_data.shape)
print("Sample train labels:", train_labels[:10])
print("Sample validation labels:", valid_labels[:10])

# Define the model
base_model = tf.keras.applications.ResNet50(input_shape=(224, 224, 3),
                                            include_top=False,
                                            weights='imagenet')
base_model.trainable = False  # Freeze the base model

model = tf.keras.Sequential([
    data_augmentation,
    base_model,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer=tf.keras.optimizers.Adam(),
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Calculate steps_per_epoch and validation_steps based on dataset size and batch size
steps_per_epoch = len(train_data) // BATCH_SIZE
validation_steps = len(valid_data) // BATCH_SIZE

# Check the first few rows of each CSV file
print(pd.read_csv(train_image_paths_file).head())
print(pd.read_csv(valid_image_paths_file).head())
print(pd.read_csv(train_labeled_studies_file, delimiter='\t').head())
print(pd.read_csv(valid_labeled_studies_file, delimiter='\t').head())

# Print a few paths to check formatting
print(train_image_paths['path'].head())
print(valid_image_paths['path'].head())
print(train_labeled_studies['path'].head())
print(valid_labeled_studies['path'].head())


# Debugging: Print dataset shapes and sample values
for image, label in train_dataset.take(1):
    print("Image shape:", image.numpy().shape)
    print("Label:", label.numpy())

# Train the model with a try-except block to catch math domain errors
try:
    history = model.fit(train_dataset,
                        epochs=100,
                        steps_per_epoch=steps_per_epoch,
                        validation_data=valid_dataset,
                        validation_steps=validation_steps)
except Exception as e:
    print(f"An error occurred during training: {e}")

# Save the model if training was successful
if 'history' in locals():
    model.save('mura_classifier.h5')
