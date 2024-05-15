import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Define the model architecture
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Load CIFAR car images
cifar_train_datagen = ImageDataGenerator(rescale=1./255)
cifar_train_generator = cifar_train_datagen.flow_from_directory(
        'cifar/train', #change to apropriate path
        target_size=(32, 32),
        batch_size=32,
        class_mode='binary')

# Load gun images
gun_train_datagen = ImageDataGenerator(rescale=1./255)
gun_train_generator = gun_train_datagen.flow_from_directory(
        'gun/train', #change to apropriate path
        target_size=(32, 32),
        batch_size=32,
        class_mode='binary')

# Combine both generators
combined_generator = tf.keras.preprocessing.image.Iterator(
    cifar_train_generator.directory_iterator,
    gun_train_generator.directory_iterator
)

# Train the model
model.fit(combined_generator,
          steps_per_epoch=(len(cifar_train_generator) + len(gun_train_generator)),
          epochs=10)
