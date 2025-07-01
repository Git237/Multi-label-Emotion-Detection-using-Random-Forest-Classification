import joblib
import re
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sentence_transformers import SentenceTransformer

# Download required NLTK resources
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

# Emotion labels
emotion_cols = ['anger', 'fear', 'joy', 'sadness', 'surprise']

# Preprocessing function
def preprocess_text(text):
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return ' '.join(tokens)

# Load models and embedder
def load_models_and_embedder():
    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    models = []
    base_path = "C:/Pranit/MS/Phillips Marburg/SEM 3/NLP"
    for emotion in emotion_cols:
        model = joblib.load(f'{base_path}/rf_model_{emotion}.joblib')
        models.append(model)
    return embedder, models

# Predict emotions for a single text
def predict_emotions(text):
    clean_text = preprocess_text(text)
    embedder, models = load_models_and_embedder()
    X = embedder.encode([clean_text])  # shape: (1, 384)

    predicted = []
    thresholds = [0.1, 0.3, 0.2, 0.2, 0.25]  # [anger, fear, joy, sadness, surprise]
    for j, model in enumerate(models):
        prob = model.predict_proba([X[0]])[0][1]
        print(f"{emotion_cols[j]}: {prob:.3f}")
        predicted.append(int(prob > thresholds[j]))

    detected_emotions = [emotion for emotion, val in zip(emotion_cols, predicted) if val == 1]
    return detected_emotions if detected_emotions else ['No emotion detected']

# Main script
if __name__ == "__main__":
    user_input = input("Enter a sentence: ")
    emotions = predict_emotions(user_input)
    print("Detected emotions:", emotions)
