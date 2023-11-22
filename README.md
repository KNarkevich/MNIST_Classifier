# MNIST Classifier

This repository contains a Python solution for the MNIST (handwritten digits database) classification problem using three different models:

1. Convolutional Neural Network (CNN)
2. Random Forest Classifier
3. Random Model (Provides random value as a result of classification)

## Solution Structure

**DigitClassifier:**
   - The main `DigitClassifier` class, which takes the name of the algorithm as an input parameter and provides predictions with exactly the same structure (inputs and outputs) irrespective of the selected algorithm. The possible values for the algorithm are: `cnn`, `rf`, `rand` for the three models described above.

## Usage

To use the `DigitClassifier`, simply create an instance of the class with the desired algorithm and call the `predict` function:

```python
from digit_classifier import DigitClassifier

# Create an instance of DigitClassifier with CNN algorithm
digit_classifier = DigitClassifier('cnn')

# Example usage
image = load_your_image()
prediction = digit_classifier.predict(image)
print(f"The predicted digit is: {prediction}")

Check out the test.py file for additional examples on how to use the DigitClassifier with different algorithms. The file contains usage scenarios and demonstrates the predict function for each model type.

