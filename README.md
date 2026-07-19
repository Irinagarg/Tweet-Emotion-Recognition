# Tweet Emotion Recognition with TensorFlow

A Recurrent Neural Network that classifies tweets into one of six emotions: sadness, joy, love, anger, fear, and surprise.

## Overview

This project builds a full text classification pipeline on the `dair-ai/emotion` dataset:

1. Load and inspect labeled tweets (train, validation, and test splits).
2. Tokenize text into integer sequences using a Keras tokenizer.
3. Pad and truncate sequences to a fixed length.
4. Train a bidirectional LSTM model on top of a trainable embedding layer.
5. Evaluate performance on unseen test data with a classification report and confusion matrix.
6. Run predictions on custom sentences.
7. Train a TF-IDF plus logistic regression baseline on the same data and compare it against the LSTM to justify the choice of a recurrent model over a simpler bag of words approach.

## Model

Embedding layer, two stacked bidirectional LSTM layers, a dense hidden layer with dropout, and a softmax output over 6 classes. Trained with early stopping on validation loss to avoid overfitting.

## Results

The LSTM reaches 88% test accuracy overall. A TF-IDF plus logistic regression baseline was trained on the same data for comparison and reaches 86% overall, a small gap on paper.

The real difference shows up on the minority classes. On love and surprise, the two rarest emotions in the dataset, the baseline recall drops to 0.58 and 0.48, meaning it misses over 40% of the actual examples in those categories. The LSTM recall on the same two classes is 0.79 and 0.71. The baseline only has access to which words appear in a tweet, with no sense of order or context, so it plays it safe on ambiguous emotions and under predicts them. The LSTM can use word order and surrounding context, which matters most exactly where the baseline struggles.

| Model | Accuracy | Macro F1 | Love recall | Surprise recall |
|---|---|---|---|---|
| Bidirectional LSTM | 0.88 | 0.82 | 0.79 | 0.71 |
| TF-IDF + Logistic Regression | 0.86 | 0.80 | 0.58 | 0.48 |

## How to run
Run all cells in order. Training takes a few minutes on CPU.

## Files

* `tweet_emotion.ipynb`: full pipeline, including the LSTM model and the baseline comparison.
* `tweet_emotion_model.keras`: saved trained LSTM model.
* `tokenizer.pkl`: saved tokenizer used to preprocess new text for the LSTM.
