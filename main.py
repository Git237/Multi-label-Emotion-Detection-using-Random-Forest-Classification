import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import joblib

# Download required nltk data (run once)
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

emotion_cols = ['anger', 'fear', 'joy', 'sadness', 'surprise']

def preprocess_text(text):
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return ' '.join(tokens)

def load_models_and_vectorizer():
    vectorizer = joblib.load('tfidf_vectorizer.joblib')
    models = []
    for emotion in emotion_cols:
        model = joblib.load(f'rf_model_{emotion}.joblib')
        models.append(model)
    return vectorizer, models

def predict(csv_file_path):
    df = pd.read_csv(csv_file_path)
    df['clean_text'] = df['text'].apply(preprocess_text)

    vectorizer, models = load_models_and_vectorizer()
    X = vectorizer.transform(df['clean_text'])

    predictions = []
    for i in range(X.shape[0]):
        sample_pred = []
        for model in models:
            pred = model.predict(X[i])
            sample_pred.append(int(pred[0]))
        predictions.append(sample_pred)

    return np.array(predictions)

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python main.py <input_csv_path>")
    else:
        preds = predict(sys.argv[1])
        print(preds)
