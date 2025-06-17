import re
import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint, uniform

# Data loading function
def load_data(file_path):
    try:
        if file_path.endswith('.csv'):
            return pd.read_csv(file_path)
        else:
            print(f"Unsupported file format: {file_path}")
            return None
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

# Text preprocessing functions
def clean_text(text):
    """Clean text for misinformation detection"""
    if isinstance(text, str):
        text = text.lower()
        text = re.sub(r'http\S+|www\S+|https\S+', '', text)
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    return ""

def preprocess_text(text):
    """Preprocess text for sarcasm detection"""
    if isinstance(text, str):
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    return ""

# Feature extraction
def extract_features(X, max_features=5000):
    vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=(1, 2))
    X_features = vectorizer.fit_transform(X)
    return X_features, vectorizer

# Model training function
def train_model_with_search(X_train, y_train):
    """Train XGBClassifier using RandomizedSearchCV with reduced fittings"""
    xgb = XGBClassifier(eval_metric='logloss',n_jobs=-1, tree_method='auto')
    param_dist = {
        "n_estimators": randint(100, 500),
        "max_depth": randint(3, 15),
        "learning_rate": uniform(0.01, 0.2),
        "subsample": uniform(0.6, 0.4),
        "colsample_bytree": uniform(0.6, 0.4)
    }
    random_search = RandomizedSearchCV(
        estimator=xgb,
        param_distributions=param_dist,
        n_iter=5, # *** Reduced from 20 to 5 ***
        scoring="accuracy",
        cv=3, # *** Reduced from 5 to 3 ***
        verbose=1,
        random_state=42,
        n_jobs=-1
    )
    random_search.fit(X_train, y_train)
    return random_search.best_estimator_, random_search.best_params_

# Model evaluation function
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    print(f"Accuracy: {accuracy:.4f}")
    print("Classification Report:")
    print(report)
    return y_pred, accuracy, report

# Misinformation detection training
def main_misinformation():
    try:
        print("Starting misinformation detection model training...")
        fake_data = load_data("C:\\Users\\alokk\\OneDrive\\Documents\\Fake News Detector and Sarcasm Detector\\Fake_small.csv")
        true_data = load_data("C:\\Users\\alokk\\OneDrive\\Documents\\Fake News Detector and Sarcasm Detector\\True.csv")

        fake_data['label'] = 1
        true_data['label'] = 0
        data = pd.concat([fake_data, true_data], ignore_index=True)

        print("Label distribution before balancing:")
        print(data['label'].value_counts())

        data = data.dropna(subset=['text'])
        data = data[data['text'].str.strip().astype(bool)]
        data['clean_text'] = data['text'].apply(clean_text)

        vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
        X = vectorizer.fit_transform(data['clean_text'])
        y = data['label']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        smote = SMOTE(random_state=42)
        X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)

        model, best_params = train_model_with_search(X_train_bal, y_train_bal)
        y_pred, accuracy, report = evaluate_model(model, X_test, y_test)

        print(f"Shape of predictions (y_pred): {y_pred.shape}")

        print(f"\n✅ Accuracy: {accuracy:.4f}")
        print("📊 Classification Report:")
        print(report)
        print("🔍 Best Hyperparameters:")
        print(best_params)

        joblib.dump(model, "misinformation_model.pkl")
        joblib.dump(vectorizer, "misinformation_vectorizer.pkl")
        print("💾 Model and vectorizer saved successfully")

    except Exception as e:
        print(f"Error in misinformation model: {e}")

# Sarcasm detection training
def main_sarcasm():
    try:
        print("Starting sarcasm detection model training...")
        data = load_data("C:\\Users\\alokk\\OneDrive\\Documents\\Fake News Detector and Sarcasm Detector\\Sarcasm_Headlines_Dataset.csv")

        print("Label distribution before balancing:")
        print(data['is_sarcastic'].value_counts())

        data['processed_text'] = data['headline'].apply(preprocess_text)
        X = data['processed_text']
        y = data['is_sarcastic']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1, 2))
        X_train_tfidf = vectorizer.fit_transform(X_train)
        X_test_tfidf = vectorizer.transform(X_test)

        smote = SMOTE(random_state=42)
        X_train_bal, y_train_bal = smote.fit_resample(X_train_tfidf, y_train)
        
        print("\nLabel distribution after balancing (sarcasm):")
        print(pd.Series(y_train_bal).value_counts())

        model, best_params = train_model_with_search(X_train_bal, y_train_bal)
        y_pred, accuracy, report = evaluate_model(model, X_test_tfidf, y_test)

        print(f"Shape of predictions (y_pred): {y_pred.shape}")

        print(f"\n😏 Sarcasm Detection Accuracy: {accuracy:.4f}")
        print("Sarcasm Detection Classification Report:")
        print(report)
        print("🔍 Best Hyperparameters:")
        print(best_params)

        joblib.dump(model, 'sarcasm_model.pkl')
        joblib.dump(vectorizer, 'sarcasm_vectorizer.pkl')
        print("💾 Model and vectorizer saved successfully")

    except Exception as e:
        print(f"Error in sarcasm model: {e}")

# Main execution
if __name__ == "__main__":
    print("=== Starting Model Training ===")
    print("\n1. Training Misinformation Detection Model")
    main_misinformation()
    print("\n2. Training Sarcasm Detection Model")
    main_sarcasm()
    print("\n=== Training Complete ===")
