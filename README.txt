🚦 Automotive Road Sign Recognition using Convolutional Neural Networks (CNN)
Project Overview
Welcome to my deep learning project focused on building a robust Convolutional Neural Network (CNN) for automotive road sign recognition. This project is a foundational step towards understanding the perception systems vital for Advanced Driver-Assistance Systems (ADAS) and autonomous vehicles. Accurate and rapid recognition of road signs is paramount for safe navigation, adherence to traffic laws, and proactive decision-making in real-world driving scenarios.

As an aspiring automotive software engineer, this project allowed me to dive deep into practical machine learning model development, from data preparation and model architecture design to training optimization and performance evaluation.

✨ Key Features
Efficient Image Preprocessing: Utilizes ImageDataGenerator for on-the-fly data loading, resizing, and pixel normalization.

Robust Data Augmentation: Implements various augmentation techniques (rotations, shifts, zooms, shears) to enhance model generalization and prevent overfitting, simulating diverse real-world conditions.

Custom CNN Architecture: A sequential CNN model designed with multiple convolutional layers, batch normalization, max-pooling, and dropout for effective feature extraction and classification.

Performance Monitoring: Integrates Early Stopping and Model Checkpointing to optimize training duration and save the best-performing model based on validation accuracy.

Comprehensive Evaluation: Provides detailed metrics (loss, accuracy) on unseen test data and visual plots of training history.

Example Prediction: Demonstrates real-time inference with confidence scores on a sample image.

🛠️ Technologies Used
Python 3.x

TensorFlow / Keras: For building and training the deep learning model.

NumPy: For numerical operations and data manipulation.

Matplotlib: For visualizing training performance.

📊 Dataset
This project utilizes the renowned German Traffic Sign Recognition Benchmark (GTSRB) dataset. The GTSRB is a large, real-world benchmark dataset that contains over 50,000 images of 43 different traffic sign classes.

Data Format: Images are provided in .ppm format.

Classes: The full dataset comprises 43 distinct classes of road signs (from 00000 to 00042).

Structure:

road_sign_dataset/
├── train/
│   ├── 00000/ (images for class 0)
│   ├── 00001/ (images for class 1)
│   └── ...
│   └── 00042/ (images for class 42)
└── test/
    ├── 00000/
    ├── 00001/
    └── ...
    └── 00042/

Note on Data Scale: Initially, I prototyped this model using a subset of only two classes (00001 and 00002) to expedite iteration and development on a mid-range machine. Upon successful validation of the core architecture, the project was scaled to incorporate all 43 classes for a comprehensive and robust solution. This iterative approach allowed for efficient resource management while ensuring the final solution's applicability across the full spectrum of traffic signs.

🧠 Model Architecture
The CNN architecture is a Sequential model comprising:

Convolutional Blocks: Three sets of Conv2D layers (32, 64, 128 filters respectively) with ReLU activation and BatchNormalization for stable learning, followed by MaxPooling2D for dimensionality reduction and feature summarization.

Dropout Layers: Applied after the last convolutional block (0.25 rate) and after the first dense layer (0.5 rate) to mitigate overfitting.

Flatten Layer: Converts the 2D feature maps into a 1D vector.

Dense Layers: A hidden Dense layer with 128 units (ReLU activation) and a final Dense output layer with NUM_CLASSES (43) units and softmax activation for multi-class probability distribution.

Model: "sequential"
_________________________________________________________________
 Layer (type)                  Output Shape          Param #
=================================================================
 conv2d (Conv2D)               (None, 32, 32, 32)    896
 batch_normalization (BatchNo  (None, 32, 32, 32)    128
 rmalization)
 max_pooling2d (MaxPooling2D)  (None, 16, 16, 32)    0
 conv2d_1 (Conv2D)             (None, 16, 16, 64)    18496
 batch_normalization_1 (Batc   (None, 16, 16, 64)    256
 hNormalization)
 max_pooling2d_1 (MaxPooling2  (None, 8, 8, 64)      0
 D)
 conv2d_2 (Conv2D)             (None, 8, 8, 128)     73856
 batch_normalization_2 (Batc   (None, 8, 8, 128)     512
 hNormalization)
 max_pooling2d_2 (MaxPooling2  (None, 4, 4, 128)     0
 D)
 dropout (Dropout)             (None, 4, 4, 128)     0
 flatten (Flatten)             (None, 2048)          0
 dense (Dense)                 (None, 128)           262272
 batch_normalization_3 (Batc   (None, 128)           512
 hNormalization)
 dropout_1 (Dropout)           (None, 128)           0
 dense_1 (Dense)               (None, 2)             258  <-- Note: This '2' becomes '43' for full dataset.
=================================================================
Total params: 357186 (1.36 MB)
Trainable params: 356482 (1.36 MB)
Non-trainable params: 704 (2.75 KB)
_________________________________________________________________

(Note: The dense_1 layer's Param # will be larger when NUM_CLASSES is 43, indicating connections to 43 output neurons instead of 2.)