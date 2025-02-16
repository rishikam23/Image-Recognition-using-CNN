# Image Recognition using Neural Networks

## Overview
This project implements image classification using various neural network architectures on the Fashion MNIST dataset. The dataset consists of 70,000 grayscale images of 28x28 pixels, categorized into 10 different classes representing clothing items such as T-shirts, trousers, and shoes. 

The project explores different optimization techniques and architectures, including:
- Fully Connected Neural Networks (Dense Layers)
- Convolutional Neural Networks (CNNs) with Conv2D and MaxPooling2D
- Techniques to reduce overfitting such as BatchNormalization, Dropout, and L2 Regularization
- Comparison of different optimizers including SGD, RMSProp, Adam, and Nadam

## Features
- Loads and preprocesses the Fashion MNIST dataset
- Builds and trains neural networks using TensorFlow and Keras
- Evaluates performance using accuracy and loss metrics
- Visualizes training progress, model performance, and misclassified images
- Implements techniques to improve generalization and performance

## Technologies Used
- Python 3
- TensorFlow/Keras
- NumPy
- Matplotlib

## Installation
To run this project, you need to have Python installed along with the required libraries. You can install dependencies using the following command:

```bash
pip install tensorflow numpy matplotlib
```

## Usage
Run the script to train and evaluate the models:
```bash
python image_recognition_using_nn.py
```

## Dataset
This project uses the **Fashion MNIST dataset**, which is automatically downloaded when running the script.

## Model Architectures
### 1. Fully Connected Neural Network (Dense Layers)
- 2 hidden layers with 256 neurons each (ReLU activation)
- Output layer with 10 neurons (Softmax activation)
- Trained using different optimizers: SGD, RMSProp, Adam, Nadam

### 2. Convolutional Neural Network (CNN)
- 3 convolutional layers with ReLU activation
- MaxPooling2D layers for downsampling
- Fully connected layers at the end
- Optimized with Adam optimizer

### 3. CNN with Regularization
- Uses **BatchNormalization** to stabilize learning
- Uses **Dropout** to prevent overfitting
- Uses **L2 Regularization** to penalize large weights
- Trained with **Adam** optimizer and a lower learning rate

## Results
The project evaluates models based on test accuracy. The final model using CNN with BatchNormalization, Dropout, and L2 Regularization achieves the best accuracy.

## Visualization
The script includes visualization of:
- Training and validation loss
- Training and validation accuracy
- Correctly and incorrectly classified images

## License
This project is licensed under the **MIT License**.

## Author
Rishika Mathur  

## Contributing  
Feedback and contributions are welcome! Feel free to **raise an issue** on GitHub if you encounter any problems or have suggestions for improvements.
