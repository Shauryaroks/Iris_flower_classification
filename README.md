# Support Vector Machines Project

This project demonstrates the use of Support Vector Machines (SVM) for classification using the classic Iris dataset. The workflow includes data exploration, visualization, model training, evaluation, and hyperparameter tuning with grid search.

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [Usage](#usage)
- [Project Steps](#project-steps)
- [Results](#results)
- [References](#references)

---

## Overview

The notebook walks through the process of:

1. Loading and visualizing the Iris dataset.
2. Performing exploratory data analysis (EDA).
3. Splitting the data into training and testing sets.
4. Training an SVM classifier.
5. Evaluating the model's performance.
6. Using GridSearchCV to optimize hyperparameters.

## Dataset

- **Source:** Built-in Iris dataset from Seaborn/Scikit-learn.
- **Description:** The dataset contains 150 samples of iris flowers, each described by four features: sepal length, sepal width, petal length, and petal width. There are three classes (species): setosa, versicolor, and virginica.

## Project Structure

- [`SVC.ipynb`](d:/SteamLibrary/epicgames/Refactored_Py_DS_ML_Bootcamp-master/16-Support-Vector-Machines/SVC.ipynb): Main Jupyter notebook containing all code, visualizations, and analysis.

## Requirements

- Python 3.x
- Jupyter Notebook
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn

Install dependencies with:

```sh
pip install pandas numpy matplotlib seaborn scikit-learn
```

## Usage

1. Open [`SVC.ipynb`](d:/SteamLibrary/epicgames/Refactored_Py_DS_ML_Bootcamp-master/16-Support-Vector-Machines/SVC.ipynb) in Jupyter Notebook or VS Code.
2. Run each cell sequentially to follow the workflow.
3. Review the outputs, plots, and model evaluation metrics.

## Project Steps

### 1. Data Loading and Visualization

- Loads the Iris dataset using Seaborn.
- Displays sample images of the three Iris species.
- Visualizes feature relationships using pairplots and KDE plots.

### 2. Exploratory Data Analysis

- Examines the structure and summary statistics of the dataset.
- Visualizes feature distributions and relationships.

### 3. Train-Test Split

- Splits the data into training and testing sets (67% train, 33% test).

### 4. Model Training

- Trains a Support Vector Classifier (`SVC`) on the training data.

### 5. Model Evaluation

- Predicts on the test set.
- Evaluates performance using confusion matrix and classification report.

### 6. Hyperparameter Tuning

- Uses `GridSearchCV` to search for the best `C` and `gamma` parameters.
- Retrains the SVM with optimal parameters and evaluates performance.

## Results

- The SVM classifier achieves high accuracy on the Iris dataset.
- Grid search helps fine-tune the model, but due to the simplicity of the dataset, improvements may be marginal.
- The notebook demonstrates the importance of model selection and parameter tuning in machine learning workflows.

## References

- [Scikit-learn SVM Documentation](https://scikit-learn.org/stable/modules/svm.html)
- [Iris Dataset - Wikipedia](https://en.wikipedia.org/wiki/Iris_flower_data_set)
- [Seaborn Documentation](https://seaborn.pydata.org/)

---

**Author:** [Your Name]  
**Course:** Python Data Science and Machine Learning Bootcamp
