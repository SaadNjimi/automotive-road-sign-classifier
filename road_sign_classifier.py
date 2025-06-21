import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import os
import matplotlib.pyplot as plt

# --- 1. Define Constants and Configuration ---
# You'll need to replace these with your actual dataset paths and parameters.
# A common dataset for this task is the German Traffic Sign Recognition Benchmark (GTSRB).
# For demonstration, we'll assume a 'data' directory with 'train' and 'test' subdirectories,
# each containing subfolders for each sign class (e.g., data/train/stop_sign/, data/train/yield_sign/).

# Define the path to your dataset (e.g., where 'train' and 'test' folders are located)
DATA_DIR = './road_sign_dataset' # Create this directory and put your train/test data inside
IMAGE_HEIGHT = 32
IMAGE_WIDTH = 32
BATCH_SIZE = 32
EPOCHS = 15 # You might need more epochs for better performance

# IMPORTANT: Now using all 43 classes from the GTSRB dataset.
# Ensure that your 'train' and 'test' directories contain subfolders for ALL 43 classes (00000 to 00042).
NUM_CLASSES = 43 # Set to 43 for the full GTSRB dataset. This will be confirmed by data loading.


# --- 2. Data Preprocessing and Augmentation ---
# ImageDataGenerator is a powerful tool for loading images from directories
# and applying real-time data augmentation.
# It supports .ppm files natively through Pillow (PIL), a dependency of TensorFlow.

# Training Data Generator with Augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,           # Normalize pixel values to [0, 1]
    rotation_range=10,        # Randomly rotate images by 10 degrees
    width_shift_range=0.1,    # Randomly shift images horizontally
    height_shift_range=0.1,   # Randomly shift images vertically
    zoom_range=0.1,           # Randomly zoom in on images
    shear_range=0.1,          # Apply shear transformation
    horizontal_flip=False,    # Road signs usually shouldn't be flipped horizontally
    fill_mode='nearest'       # Fill in new pixels created by transformations
)

# Test Data Generator (only rescaling)
test_datagen = ImageDataGenerator(rescale=1./255)

# Load data from directories
# Make sure your 'train' and 'test' directories are structured as:
# road_sign_dataset/
# ├── train/
# │   ├── 00000/
# │   ├── 00001/
# │   ├── ...
# │   └── 00042/
# └── test/
#     ├── 00000/
#     ├── 00001/
#     ├── ...
#     └── 00042/

print("Loading training data...")
try:
    train_generator = train_datagen.flow_from_directory(
        os.path.join(DATA_DIR, 'train'),
        target_size=(IMAGE_WIDTH, IMAGE_HEIGHT),
        batch_size=BATCH_SIZE,
        class_mode='categorical', # Use 'categorical' for one-hot encoded labels
        color_mode='rgb' # Assuming RGB images. Use 'grayscale' for grayscale images.
    )
    print(f"Found {train_generator.samples} training images belonging to {train_generator.num_classes} classes.")
    # Update NUM_CLASSES based on actual data if not set manually
    if NUM_CLASSES != train_generator.num_classes:
        print(f"Warning: NUM_CLASSES updated from {NUM_CLASSES} to {train_generator.num_classes} based on training data.")
        NUM_CLASSES = train_generator.num_classes
except Exception as e:
    print(f"Error loading training data. Please ensure '{os.path.join(DATA_DIR, 'train')}' exists and contains image subdirectories. Error: {e}")
    print("Creating dummy data generators for demonstration purposes...")
    # Create dummy generators for demonstration if data is not found
    train_generator = iter([]) # Empty iterator
    test_generator = iter([]) # Empty iterator
    print("If you see errors after this, it's likely due to missing data.")
    print("Please set up your dataset as described in the comments.")


print("Loading test data...")
try:
    test_generator = test_datagen.flow_from_directory(
        os.path.join(DATA_DIR, 'test'),
        target_size=(IMAGE_WIDTH, IMAGE_HEIGHT),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        color_mode='rgb',
        shuffle=False # Do not shuffle test data to maintain order for evaluation
    )
    print(f"Found {test_generator.samples} test images belonging to {test_generator.num_classes} classes.")
    if NUM_CLASSES != test_generator.num_classes and test_generator.num_classes > 0:
         print(f"Warning: Test data has {test_generator.num_classes} classes, which differs from training data's {NUM_CLASSES} classes.")
except Exception as e:
    print(f"Error loading test data. Please ensure '{os.path.join(DATA_DIR, 'test')}' exists and contains image subdirectories. Error: {e}")
    print("Cannot proceed with testing without test data. Please set up your dataset.")


# --- 3. Build the CNN Model Architecture ---
# We'll use a common CNN architecture: Conv2D -> MaxPooling -> Conv2D -> MaxPooling -> Flatten -> Dense -> Dense (Output)

def build_cnn_model(input_shape, num_classes):
    model = models.Sequential()

    # First Convolutional Block
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape, padding='same'))
    model.add(layers.BatchNormalization()) # Added Batch Normalization
    model.add(layers.MaxPooling2D((2, 2)))

    # Second Convolutional Block
    model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2)))

    # Third Convolutional Block (optional, for deeper networks)
    model.add(layers.Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.25)) # Added Dropout for regularization

    # Flatten the output of the convolutional layers to feed into Dense layers
    model.add(layers.Flatten())

    # Fully Connected Layers (Dense)
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.5)) # Added Dropout for regularization

    # Output Layer
    # Use 'softmax' for multi-class classification
    model.add(layers.Dense(num_classes, activation='softmax'))

    return model

# Define the input shape based on image dimensions and color channels
INPUT_SHAPE = (IMAGE_HEIGHT, IMAGE_WIDTH, 3) # 3 for RGB images

print("\nBuilding the CNN model...")
model = build_cnn_model(INPUT_SHAPE, NUM_CLASSES)
model.summary() # Print a summary of the model's layers

# --- 4. Compile the Model ---
# Compilation configures the model for training.
# Optimizer: How the model updates based on the loss function (e.g., 'adam')
# Loss function: Measures how well the model is doing (e.g., 'categorical_crossentropy' for one-hot encoded labels)
# Metrics: Used to monitor the training and testing steps (e.g., 'accuracy')

print("\nCompiling the model...")
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# --- 5. Train the Model ---
# Train the model using the training data generator.
# steps_per_epoch: Total number of batches (samples / batch_size)
# validation_data: Data to evaluate the model's performance on during training
# callbacks: Functions to call during training (e.g., EarlyStopping, ModelCheckpoint)

# Define callbacks
callbacks = [
    tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
    # Saving in .h5 format for compatibility
    tf.keras.callbacks.ModelCheckpoint('best_road_sign_model.h5', save_best_only=True, monitor='val_accuracy', mode='max')
]

print("\nStarting model training...")
# Check if generators are empty before fitting
if train_generator and test_generator and train_generator.samples > 0 and test_generator.samples > 0:
    history = model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=test_generator,
        validation_steps=test_generator.samples // BATCH_SIZE,
        callbacks=callbacks
    )
    print("\nModel training complete.")

    # --- 6. Evaluate the Model on Test Data ---
    print("\nEvaluating the model on the full test dataset...")
    test_loss, test_accuracy = model.evaluate(test_generator, steps=test_generator.samples // BATCH_SIZE)
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")

    # --- 7. Visualize Training History ---
    print("\nPlotting training history...")
    plt.figure(figsize=(12, 5))

    # Plot training & validation accuracy values
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')

    # Plot training & validation loss values
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.tight_layout()
    plt.show()

    # --- 8. Make Predictions (Example) ---
    print("\nMaking example predictions...")
    # Get a batch of images and labels from the test generator
    # It's important to reset the generator if you've already iterated through it
    test_generator.reset()
    try:
        images, labels = next(test_generator)
        # Select the first image for prediction
        img_to_predict = images[0]
        true_label_one_hot = labels[0]
        true_label_index = np.argmax(true_label_one_hot)

        # Reshape the image for prediction (add batch dimension)
        img_to_predict_batch = np.expand_dims(img_to_predict, axis=0)

        # Predict
        predictions = model.predict(img_to_predict_batch)
        predicted_class_index = np.argmax(predictions[0])
        confidence = np.max(predictions[0])

        # Get class names if available
        class_indices = train_generator.class_indices
        # Invert the dictionary to get class name from index
        idx_to_class = {v: k for k, v in class_indices.items()}

        true_class_name = idx_to_class.get(true_label_index, f"Class {true_label_index}")
        predicted_class_name = idx_to_class.get(predicted_class_index, f"Class {predicted_class_index}")

        print(f"\n--- Prediction for one image ---")
        print(f"True Label: {true_class_name}")
        print(f"Predicted Label: {predicted_class_name} (Confidence: {confidence:.2f})")

        # Display the image
        plt.imshow(img_to_predict)
        plt.title(f"True: {true_class_name}\nPredicted: {predicted_class_name} ({confidence:.2f})")
        plt.axis('off')
        plt.show()

    except StopIteration:
        print("No more images in the test generator for prediction example.")
    except Exception as e:
        print(f"An error occurred during prediction example: {e}")
else:
    print("\nModel training skipped due to missing or empty data generators. Please set up your dataset.")
    print("You need to create a directory structure like:")
    print("  ./road_sign_dataset/")
    print("  ├── train/")
    print("  │   ├── 00000/")
    print("  │   ├── 00001/")
    print("  │   ├── ...")
    print("  │   └── 00042/")
    print("  └── test/")
    print("      ├── 00000/")
    print("      ├── 00001/")
    print("      ├── ...")
    print("      └── 00042/")
    print("Ensure these directories contain the respective .ppm image files for all 43 classes.")


# --- 9. Save the Model (Optional) ---
# It's good practice to save your trained model so you can reuse it later without retraining.
print("\nSaving the trained model...")
try:
    # Saving in .h5 format for compatibility
    model.save('road_sign_cnn_model.h5')
    print("Model saved as 'road_sign_cnn_model.h5'")
except Exception as e:
    print(f"Error saving model: {e}")

# To load the model later:
# loaded_model = tf.keras.models.load_model('road_sign_cnn_model.h5') # Load from .h5
# print("\nModel loaded successfully for future use.")
