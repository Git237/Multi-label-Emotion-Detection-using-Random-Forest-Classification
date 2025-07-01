import pandas as pd
import numpy as np
import re
import nltk
import joblib
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sentence_transformers import SentenceTransformer

nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

emotion_cols = ['anger', 'fear', 'joy', 'sadness', 'surprise']

# Preprocessing
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return ' '.join(tokens)

# Load models
models = []
for emotion in emotion_cols:
    model = joblib.load(f'C:/Pranit/MS/Phillips Marburg/SEM 3/NLP/rf_model_{emotion}.joblib')
    models.append(model)

# Load embedding model
bert_model = SentenceTransformer('all-MiniLM-L6-v2')

# Emotion-specific thresholds (tuned)
thresholds = [0.05, 0.3, 0.2, 0.2, 0.25]  # anger, fear, joy, sadness, surprise

def predict_from_text(texts):
    df = pd.DataFrame({'text': texts})
    df['clean_text'] = df['text'].apply(preprocess_text)
    X = bert_model.encode(df['clean_text'].tolist())

    predictions = []
    for i in range(len(X)):
        sample_pred = []
        for j, model in enumerate(models):
            prob = model.predict_proba([X[i]])[0][1]
            sample_pred.append(int(prob > thresholds[j]))
        predictions.append(sample_pred)
    return np.array(predictions)

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python predict_bert_rf.py <input_csv_path>")
    else:
        df = pd.read_csv(sys.argv[1])
        results = predict_from_text(df['text'].tolist())
        print(results)
