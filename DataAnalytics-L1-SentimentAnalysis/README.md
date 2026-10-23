
# Task 4: Sentiment Analysis

## Overview
This project uses Natural Language Processing (NLP) and machine learning to classify IMDb movie reviews as positive or negative.

## Objectives
- Clean and preprocess text reviews.
- Convert text into numerical features using TF-IDF.
- Split the dataset into 80% training and 20% testing data.
- Train Naive Bayes and Logistic Regression classifiers.
- Compare model performance using accuracy, precision, recall and F1-score.
- Visualize confusion matrices, sentiment distribution and word clouds.
- Review misclassified examples.

## Technologies
- Python
- Pandas and NumPy
- Scikit-learn
- Matplotlib and Seaborn
- WordCloud

## Dataset
IMDb movie reviews dataset with positive and negative sentiment labels.

## How to Run
Install the dependencies:

```bash
python -m pip install pandas numpy scikit-learn matplotlib seaborn wordcloud
```

Run the project:

```bash
python main.py
```

## Outputs
The outputs folder contains cleaned reviews, model comparison metrics, classification reports, confusion matrices, word clouds, misclassified examples and a project summary.

## Conclusion
The two classifiers are evaluated on the same held-out test set. Their measured results are saved for comparison. The model with the stronger evaluation metrics can be selected for this binary sentiment classification task.
