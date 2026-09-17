import os
import pickle
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

os.makedirs('model', exist_ok=True)

df = pd.read_csv('reviews.csv', encoding='utf-8')
df['text'] = df['text'].astype(str).str.lower().str.strip()
df = df.dropna(subset=['text', 'label'])

print(f"Total examples: {len(df)}")
print(df['label'].value_counts())

X_train, X_test, y_train, y_test = train_test_split(
    df['text'], df['label'], test_size=0.2, random_state=42, stratify=df['label']
)

vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1, max_df=0.95)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

model = LogisticRegression(max_iter=1000, class_weight='balanced')
model.fit(X_train_vec, y_train)

y_pred = model.predict(X_test_vec)
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred, labels=['negative', 'neutral', 'positive']))

with open('model/model.pkl', 'wb') as f:
    pickle.dump({'model': model, 'vectorizer': vectorizer}, f)

print("\nModel saved to model/model.pkl")