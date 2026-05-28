# Heart Disease Prediction Using Machine Learning

This project implements an end-to-end machine learning classification workflow to predict the presence of heart disease using a clinical dataset. The project compares three different classification models and evaluates their performance using accuracy, precision, recall, confusion matrices, and ROC-AUC curves.

## Project Overview

The goal of this project is to analyze a heart disease dataset and build machine learning models that can classify whether a patient is likely to have heart disease or not.

The project includes:

* Loading and analyzing a CSV dataset
* Checking missing values
* Splitting data into training and testing sets
* Applying feature scaling
* Training multiple classification models
* Evaluating model performance
* Comparing models using classification metrics
* Visualizing confusion matrices and ROC curves
* Selecting the best model based on recall

## Dataset

The dataset used in this project is the Heart Disease Dataset from Kaggle.

Dataset link: https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset

Download the file named `heart.csv` and place it inside the `data` folder.

Expected structure:

```text
heart-disease-prediction-ml/
│
├── data/
│   └── heart.csv
│
├── main.py
├── requirements.txt
├── README.md
└── outputs/
```

## Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn

## Machine Learning Models Used

The following models were trained and compared:

1. Logistic Regression
2. Decision Tree Classifier
3. Random Forest Classifier

## Evaluation Metrics

The models were evaluated using:

* Accuracy
* Precision
* Recall
* Classification Report
* Confusion Matrix
* ROC Curve
* AUC Score

## Why Recall Is Important

In heart disease prediction, recall is very important because false negatives can be dangerous. A false negative means the model predicts that a patient does not have heart disease when they actually do.

For medical diagnosis support systems, reducing false negatives is often more important than reducing false positives because missing a real disease case can delay treatment.

## How to Run the Project

First, install the required libraries:

```bash
pip install -r requirements.txt
```

Then run the project:

```bash
python main.py
```

On Mac, you may need to use:

```bash
python3 main.py
```

## Output Files

After running the project, the following files will be generated inside the `outputs` folder:

```text
outputs/
├── logistic_regression_confusion_matrix.png
├── decision_tree_confusion_matrix.png
├── random_forest_confusion_matrix.png
├── roc_curve_comparison.png
└── model_comparison_results.csv
```

## Project Results

The final model comparison table includes accuracy, precision, recall, and AUC score for all three models. The best model is selected based on recall because recall is highly important in clinical prediction tasks.

## Disclaimer

This project is created for educational and portfolio purposes only. It is not intended for real medical diagnosis or clinical decision-making.
