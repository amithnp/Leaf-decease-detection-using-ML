import os
import random
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from PIL import Image
import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

# Directory containing the dataset
dataset_dir = 'C:\\project\\tomato leaf disease\\cucumber70'

# List of classes
classes = ["Anthracnose", "Bacterial Wilt", "Downy Mildew", "Fresh Leaf", "Gummy Stem Blight"]

# Function to display images from each class
def display_images_per_class(dataset_dir, classes, num_images=3):
    for class_name in classes:
        class_dir = os.path.join(dataset_dir, class_name)
        images = os.listdir(class_dir)
        images = random.sample(images, min(num_images, len(images)))
        print(f"Displaying {num_images} images from class: {class_name}")
        plt.figure(figsize=(15, 3))
        for i, image_name in enumerate(images):
            image_path = os.path.join(class_dir, image_name)
            image = Image.open(image_path)
            plt.subplot(1, num_images, i+1)
            plt.imshow(image)
            plt.title(f"Image {i+1}")
            plt.axis('off')
        plt.show()

# Display images from each class
display_images_per_class(dataset_dir, classes, num_images=3)

# Define dataset directory and parameters
dataset_dir = 'C:\\project\\tomato leaf disease\\cucumber70'
batch_size = 32
image_shape = (224, 224, 3)  # Specify the size of your input images for VGG16

# Data augmentation and preprocessing
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2  # Splitting the data into training and validation sets
)

train_generator = train_datagen.flow_from_directory(
    dataset_dir,
    target_size=(image_shape[0], image_shape[1]),
    batch_size=batch_size,
    class_mode='categorical',
    subset='training'  # Training set
)

validation_generator = train_datagen.flow_from_directory(
    dataset_dir,
    target_size=(image_shape[0], image_shape[1]),
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation'  # Validation set
)

# Load the pre-trained VGG16 model
base_model = VGG16(weights='imagenet', include_top=False, input_shape=image_shape)

# Freeze the layers of the pre-trained VGG16 model
for layer in base_model.layers:
    layer.trainable = False

# Add custom layers on top of VGG16
model = tf.keras.Sequential([
    base_model,
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(5, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# Train the model
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    epochs=50,  # Adjust the number of epochs as needed
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // batch_size
)

# Save the trained model
model.save('vgg16modelnew1epochs50.h5')

# Plot training and validation accuracy
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Evaluate the model
validation_loss, validation_accuracy = model.evaluate(validation_generator)
print(f'Validation Accuracy: {validation_accuracy}')
print(f'Validation Loss: {validation_loss}')

# Predict classes for validation data
predicted_probabilities = model.predict(validation_generator)
predicted_labels = np.argmax(predicted_probabilities, axis=1)

# True labels
true_labels = validation_generator.classes

# Compute precision, recall, and F1-score
precision, recall, f1_score, _ = precision_recall_fscore_support(true_labels, predicted_labels, average='weighted')

# Print precision, recall, and F1-score
print(f'Precision: {precision}')
print(f'Recall: {recall}')
print(f'F1 Score: {f1_score}')

# Compute confusion matrix
conf_matrix = confusion_matrix(true_labels, predicted_labels)

# Plot confusion matrix with colors
plt.figure(figsize=(10, 8))
sns.set(font_scale=1.2)  # Adjust font size
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap="Blues", cbar=False)
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.title('Confusion Matrix')
plt.show()
