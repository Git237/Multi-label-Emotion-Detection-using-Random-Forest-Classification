import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score


# Download nltk resources once
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

# 1. Load dataset
df = pd.read_csv('C:\\Pranit\\MS\\Phillips Marburg\\SEM 3\\NLP\\track-a.csv')  

print("Dataset shape:", df.shape)
print("Sample data:")
print(df.head())

# 2. Basic EDA
print("\nLabel distribution:")
emotion_cols = ['anger', 'fear', 'joy', 'sadness', 'surprise']
print(df[emotion_cols].sum())

# Visualize label distribution
#sns.barplot(x=emotion_cols, y=df[emotion_cols].sum().values)
#plt.title("Emotion Label Distribution")
#plt.show()

# 3. Text Preprocessing
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    # Lowercase
    text = text.lower()
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    # Remove special chars and digits
    text = re.sub(r'[^a-z\s]', '', text)
    # Tokenize & remove stopwords + lemmatize
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return ' '.join(tokens)

# Apply preprocessing to the text column
df['clean_text'] = df['text'].apply(preprocess_text)

print("\nSample cleaned text:")
print(df[['text', 'clean_text']].head())

# 4. TF-IDF Vectorization (unigrams + bigrams)
vectorizer = TfidfVectorizer(ngram_range=(1,2), max_features=10000)
X = vectorizer.fit_transform(df['clean_text'])

print(f"\nTF-IDF matrix shape: {X.shape}")

# 5. Multi-label target matrix
y = df[emotion_cols].values

print("\nMulti-label target shape:", y.shape)




# Parameters
n_splits = 5
random_state = 42

kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

# Emotion labels
emotion_cols = ['anger', 'fear', 'joy', 'sadness', 'surprise']

# Initialize metric accumulators
metrics = {emotion: {'precision': [], 'recall': [], 'f1': [], 'roc_auc': []} for emotion in emotion_cols}

# Convert sparse matrix to CSR for efficiency
X_csr = X.tocsr()
y = df[emotion_cols].values

for fold, (train_index, val_index) in enumerate(kf.split(X_csr)):
    print(f"\nTraining Fold {fold+1}/{n_splits}...")
    
    X_train, X_val = X_csr[train_index], X_csr[val_index]
    y_train, y_val = y[train_index], y[val_index]
    
    for i, emotion in enumerate(emotion_cols):
        clf = RandomForestClassifier(random_state=random_state)
        clf.fit(X_train, y_train[:, i])
        
        # Predict on validation fold
        y_pred = clf.predict(X_val)
        y_prob = clf.predict_proba(X_val)[:, 1]
        
        # Calculate validation metrics
        precision = precision_score(y_val[:, i], y_pred, zero_division=0)
        recall = recall_score(y_val[:, i], y_pred, zero_division=0)
        f1 = f1_score(y_val[:, i], y_pred, zero_division=0)
        
        try:
            roc_auc = roc_auc_score(y_val[:, i], y_prob)
        except ValueError:
            roc_auc = np.nan
        
        # Store metrics
        metrics[emotion]['precision'].append(precision)
        metrics[emotion]['recall'].append(recall)
        metrics[emotion]['f1'].append(f1)
        metrics[emotion]['roc_auc'].append(roc_auc)
        
        print(f"{emotion}: Precision={precision:.3f}, Recall={recall:.3f}, F1={f1:.3f}, ROC AUC={roc_auc:.3f}")

# After all folds, print average validation metrics per emotion
print("\nAverage validation metrics over folds:")
for emotion in emotion_cols:
    print(f"\n{emotion}:")
    for metric_name in ['precision', 'recall', 'f1', 'roc_auc']:
        vals = [v for v in metrics[emotion][metric_name] if not np.isnan(v)]
        avg_val = np.mean(vals) if vals else float('nan')
        print(f"  {metric_name}: {avg_val:.3f}")
