# Sentiment Analysis of Amazon Product Reviews

This project performs sentiment analysis on Amazon product reviews using multiple approaches — from simple baselines to machine learning and neural network models.

## Dataset
- Source: Kaggle
- Format: fastText
- Size: 100,000 reviews
- Labels: Positive (1) / Negative (0)

## Models

| Model | Description |
|---|---|
| Majority Class | Always predicts the most frequent label |
| Rule-Based | Keyword matching (positive/negative word lists) |
| Logistic Regression (TF-IDF) | Linear classifier on TF-IDF features (10K, bigrams) |
| SVM - LinearSVC (TF-IDF) | Support Vector Machine on TF-IDF features |
| PyTorch FFNN (TF-IDF) | 2-layer fully connected neural network on TF-IDF features |

## Results

| Model | Accuracy | Precision | Recall | F1-score |
|---|---|---|---|---|
| Baseline 1 (Majority) | 0.5199 | 0.5199 | 1.0000 | 0.6841 |
| Baseline 2 (Rule-Based) | 0.6492 | 0.5998 | 0.9774 | 0.7434 |
| Logistic Regression (TF-IDF) | 0.8851 | 0.8859 | 0.8940 | 0.8900 |
| SVM - LinearSVC (TF-IDF) | 0.8803 | 0.8824 | 0.8882 | 0.8853 |
| PyTorch FFNN (TF-IDF) | 0.8800 | 0.8778 | 0.8937 | 0.8857 |

Logistic Regression achieved the best overall F1-score (0.8900), followed closely by SVM (0.8853) and PyTorch FFNN (0.8857). All three ML models significantly outperform the baselines.

## Project Structure

```
Code/
  main.py         - Full pipeline: data loading, baselines, TF-IDF, SVM, FFNN
Data/
  train.ft.txt    - Amazon reviews in fastText format
app.py            - Streamlit web app (live demo)
requirements.txt  - Python dependencies
```

## Run Locally

```bash
pip install -r requirements.txt

# Train all models (CLI):
python Code/main.py

# Launch the web app:
streamlit run app.py
```

## Requirements

- Python 3.x
- pandas, scikit-learn, torch, prettytable, streamlit, plotly

## Author
Roy Meoded
