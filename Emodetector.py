import pandas as pd
import numpy as np
import re
import nltk
import joblib

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sentence_transformers import SentenceTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score

# Download NLTK resources
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

# Load data
df = pd.read_csv('C:/Pranit/MS/Phillips Marburg/SEM 3/NLP/track-a.csv')
print("Dataset shape:", df.shape)

# Emotion labels
emotion_cols = ['anger', 'fear', 'joy', 'sadness', 'surprise']

# Preprocessing function
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return ' '.join(tokens)

# Apply preprocessing
df['clean_text'] = df['text'].apply(preprocess_text)

# Sentence-BERT embeddings
print("Generating sentence embeddings...")
bert_model = SentenceTransformer('all-MiniLM-L6-v2')
X = bert_model.encode(df['clean_text'].tolist())
print("Embedding shape:", X.shape)

# Target labels
y = df[emotion_cols].values
print("Target shape:", y.shape)

# 5-fold Cross-Validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
metrics = {emotion: {'precision': [], 'recall': [], 'f1': [], 'roc_auc': []} for emotion in emotion_cols}

for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
    print(f"\nTraining Fold {fold+1}/5...")
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]

    for i, emotion in enumerate(emotion_cols):
        clf = RandomForestClassifier(random_state=42, class_weight='balanced')
        clf.fit(X_train, y_train[:, i])

        y_pred = clf.predict(X_val)
        y_prob = clf.predict_proba(X_val)[:, 1]

        precision = precision_score(y_val[:, i], y_pred, zero_division=0)
        recall = recall_score(y_val[:, i], y_pred, zero_division=0)
        f1 = f1_score(y_val[:, i], y_pred, zero_division=0)
        try:
            roc_auc = roc_auc_score(y_val[:, i], y_prob)
        except:
            roc_auc = np.nan

        metrics[emotion]['precision'].append(precision)
        metrics[emotion]['recall'].append(recall)
        metrics[emotion]['f1'].append(f1)
        metrics[emotion]['roc_auc'].append(roc_auc)

        print(f"{emotion}: Precision={precision:.3f}, Recall={recall:.3f}, F1={f1:.3f}, ROC AUC={roc_auc:.3f}")

# Average metrics
print("\nAverage validation metrics:")
for emotion in emotion_cols:
    print(f"\n{emotion}:")
    for metric in ['precision', 'recall', 'f1', 'roc_auc']:
        scores = [v for v in metrics[emotion][metric] if not np.isnan(v)]
        avg = np.mean(scores) if scores else 0
        print(f"  {metric}: {avg:.3f}")

# Train final models on full data
print("\nTraining final models on full dataset...")
final_models = []
for i, emotion in enumerate(emotion_cols):
    clf = RandomForestClassifier(random_state=42, class_weight='balanced')
    clf.fit(X, y[:, i])
    final_models.append(clf)

# Save models and embedding pipeline
for i, emotion in enumerate(emotion_cols):
    joblib.dump(final_models[i], f'C:/Pranit/MS/Phillips Marburg/SEM 3/NLP/rf_model_{emotion}.joblib')

bert_model.save('bert_embedding_model')  # Optional, or you can re-load from HuggingFace later
