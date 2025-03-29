

# Waste Management using CNN & Bayesian Optimization

This project demonstrates a deep learning solution for waste management using a custom Convolutional Neural Network (CNN) optimized with Bayesian optimization via [Optuna](https://optuna.org/). The system is designed to classify waste images into their respective categories, leveraging data augmentation techniques, class weighting, and hyperparameter tuning for improved performance.

## Overview

- **Objective:** Develop a robust CNN model to accurately classify waste images, thereby aiding in the automation of waste management.
- **Techniques Used:**  
  - **Data Augmentation:** Random rotations, flips, affine transformations, and color jittering to increase dataset diversity.
  - **CNN Architecture:** Custom convolutional layers combined with fully connected layers.
  - **Bayesian Optimization:** Optuna is used to fine-tune hyperparameters (learning rate, batch size, dropout rate, number of filters, and optimizer type) to maximize accuracy.
  - **Class Weighting:** Computed to handle class imbalance effectively.

## Dataset

The project utilizes the **Waste Classification Data** available on Kaggle. This dataset comprises waste images categorized into different classes for classification tasks. You can download the dataset directly from Kaggle:

[Waste Classification Data on Kaggle](https://www.kaggle.com/datasets/techsash/waste-classification-data)  

## Installation

Ensure you have Python 3.7+ installed and install the required libraries:

```bash
pip install torch torchvision scikit-learn numpy matplotlib seaborn tqdm optuna pandas
```

Clone the repository and navigate to the project directory:

```bash
git clone https://github.com/abhay-2108/Waste-Management-using-CNN-Bayesian-Optimization-
cd Waste-Management-using-CNN-Bayesian-Optimization-
```

## Usage

1. **Dataset Preparation:**  
   Download the dataset from Kaggle and organize it into the structure mentioned above.
2. **Run Training:**  
   Execute the training script (e.g., `train.py`). The script applies data transformations, trains the model with the best-found hyperparameters, and saves the best model based on validation loss.
3. **Evaluation:**  
   After training, the model is evaluated using metrics like accuracy, precision, recall, and F1 score. A confusion matrix and ROC curve are plotted to visualize the model's performance.

## Key Results

- **Best Hyperparameters Found via Bayesian Optimization:**
  - `lr`: 0.0002457012561502338
  - `batch_size`: 64
  - `dropout_rate`: 0.2887915419451806
  - `num_filters`: 128
  - `optimizer`: SGD

- **Training & Validation Performance (Epoch 10):**
  - **Train Loss:** 0.3196  
  - **Train Accuracy:** 86.87%
  - **Validation Loss:** 0.3099


## Conclusion

This project presents an end-to-end solution for waste image classification using a custom CNN. By incorporating Bayesian optimization, the model achieves competitive performance, demonstrating the benefits of hyperparameter tuning in deep learning applications. Contributions and improvements are welcome!

---
