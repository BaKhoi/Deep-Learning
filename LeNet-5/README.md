# Machine Learning Practice Problem

## Description
This is the implementation of the LeNet-5 CNN Architecture to classify clothes using PyTorch's FashionMNIST dataset

## Table of Contents
- [Machine Learning Practice Problem](#machine-learning-practice-problem)
  - [Description](#description)
  - [Table of Contents](#table-of-contents)
  - [Dataset](#dataset)
  - [Model](#model)
  - [Results](#results)
    - [Loss and Accuracy Curve](#loss-and-accuracy-curve)
    - [Precision, Recall and F1 for earch class](#precision-recall-and-f1-for-earch-class)
  - [Why do we use Batch Normalization](#why-do-we-use-batch-normalization)
    
## Dataset
FashionMNIST Dataset

Check documentation for more: 
https://pytorch.org/vision/0.18/generated/torchvision.datasets.FashionMNIST.html

## Model
- **Type:** LeNet-5 Convolutional Neural Network
  
 ![Model Architecture](images\lenet-5.png)
- **Architecture:** 
  - Input: 28x28 grayscale images
  - C1: 6x 5x5 convolution, ReLU activation
  - S2: 2x2 max pooling
  - C3: 16x 5x5 convolution, ReLU activation
  - S4: 2x2 max pooling
  - C5: Fully connected layer with 120 units with Batch Normalization, ReLU activation
  - F6: Fully connected layer with 84 units with Batch Normalization, ReLU activation
  - Output: Fully connected layer with 10 units (for classification), Softmax activation

## Results
Since FashionMNIST is a balanced dataset with 10 classes, accuracy is used to get an overall performance of the model. Precision, Recall, and F1 score are used to get insights into class-specific performance.

### Loss and Accuracy Curve
![Curves](images/Loss%20and%20Accuracy.png)

### Precision, Recall and F1 for earch class
![Othermetrics](images/Other%20metrics.png)


## Why do we use Batch Normalization
Using Batch Normalization speed up the training process and increase test accuracy

- Reduces internal covariate shift meaning that the inputs to a layer will have a mean of 0 and variance of 1
- Improve gradient flow as batch normalization helps to avoid internal covariate shift 
- Adds regularization to the model
- Helps to prevent the dead ReLU problem as it reduces the number of dead neurons (neurons that are not active) during the training process

*Note*: Batch Normalization should not be used when the batch size is small. The reason is because the mean and variance become noisy and less representative (as it tends to be more sensitive to outliers and anomalies), making the normalization inaccurate. Consequently, the model is prone to be unstable and converges more slowly.
