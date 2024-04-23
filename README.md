# Digit Recognizer using Convolutional Neural Networks
This project aims to recognize handwritten digits (0-9) from the famous MNIST dataset using Convolutional Neural Networks (CNNs) implemented in TensorFlow and Keras.

## Dataset
The dataset used in this project is the MNIST dataset, which consists of 28x28 grayscale images of handwritten digits along with their corresponding labels.

Training Data: The training data consists of 42,000 labeled images of handwritten digits.

Test Data: The test data contains 28,000 unlabeled images.

## Project Structure
digit-recognizer.ipynb: Jupyter Notebook containing the Python code for training the CNN model, evaluating its performance, and generating predictions on the test set.

submission.csv: CSV file containing the predicted labels for the test data, formatted for submission to Kaggle.

## Dependencies
- Python
- TensorFlow
- NumPy
- Pandas
- OpenCV
- Matplotlib
- scikit-learn
- Seaborn

## Model Architecture
The CNN model architecture used in this project consists of:
- Three convolutional layers with ReLU activation functions.
- Two max-pooling layers for downsampling.
- Two fully connected (dense) layers with ReLU activation functions.
- Output layer with softmax activation function for multi-class classification.

## Performance
The model achieved a validation accuracy of approximately [99.1785%] after [20] epochs. 

The confusion matrix visualizes the model's performance on the validation set.
![image](https://github.com/Nourhan613/TEKNOSOFT/assets/98773024/c8eefbd4-c238-4936-81eb-7d2d505dde3d)
