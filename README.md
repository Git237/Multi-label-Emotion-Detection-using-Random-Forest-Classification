# Multi-label-Emotion-Detection-using-Random-Forest-Classification
Project Goal
This project implements a multi-label text classification system capable of detecting multiple emotional labels (anger, fear, joy, sadness, surprise) within a single sentence.

The solution utilizes Sentence-BERT embeddings for robust semantic feature extraction, followed by a Random Forest Classifier ensemble to model the complex multi-label relationships.

✨ Key Features & Technical Highlights
Multi-Label Classification: Solved a complex multi-label problem by training five independent Binary Relevance Random Forest classifiers, one for each emotion.

Advanced Feature Engineering: Used Sentence-BERT (all-MiniLM-L6-v2) to transform raw text into dense, high-quality, 384-dimensional semantic embeddings, capturing deep contextual meaning.

Robust Model Training: Employed K-Fold Cross-Validation (5-Fold) to ensure the models' robustness and generalize performance across different data subsets.

Text Preprocessing Pipeline: Implemented a full NLP pipeline for text cleaning, stopword removal, and WordNet Lemmatization.

Performance Metrics: Evaluated models using key metrics essential for imbalanced data, including Precision, Recall, F1-Score, and ROC AUC.
