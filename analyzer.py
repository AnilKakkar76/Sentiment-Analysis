import pickle

class SentimentAnalyzer:
    def __init__(self, model_path='model/model.pkl', threshold=0.4):
        with open(model_path, 'rb') as f:
            data = pickle.load(f)
        self.model = data['model']
        self.vectorizer = data['vectorizer']
        self.threshold = threshold

    def predict(self, text):
        text = (text or '').lower().strip()
        if not text:
            return {'label': 'unknown', 'confidence': 0.0, 'probabilities': {}}

        X = self.vectorizer.transform([text])
        probs = self.model.predict_proba(X)[0]
        labels = self.model.classes_
        prob_dict = {label: round(float(p), 3) for label, p in zip(labels, probs)}

        max_idx = probs.argmax()
        confidence = float(probs[max_idx])
        label = labels[max_idx] if confidence >= self.threshold else 'uncertain'

        return {
            'label': label,
            'confidence': round(confidence, 3),
            'probabilities': prob_dict
        }