import pandas as pd
import pickle
import re
import string
import os
import zipfile
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

# ─── Kaggle Dataset Info ───
KAGGLE_DATASET = "clmentbisaillon/fake-and-real-news-dataset"
DATA_DIR = "dataset"

def download_from_kaggle():
    """Download Fake and Real News Dataset from Kaggle using kaggle API."""
    try:
        import kaggle
        print(f"Downloading dataset from Kaggle: {KAGGLE_DATASET}")
        os.makedirs(DATA_DIR, exist_ok=True)
        kaggle.api.authenticate()
        kaggle.api.dataset_download_files(KAGGLE_DATASET, path=DATA_DIR, unzip=True)
        print("✅ Kaggle dataset downloaded successfully!")
        return True
    except ImportError:
        print("⚠️  'kaggle' package not installed. Install it with: pip install kaggle")
        return False
    except Exception as e:
        print(f"⚠️  Kaggle download failed: {e}")
        print("   Make sure ~/.kaggle/kaggle.json exists with your API credentials.")
        print("   Get it from: https://www.kaggle.com/settings → API → Create New Token")
        return False

def load_kaggle_csv():
    """Load the Fake.csv and True.csv files from the dataset directory."""
    fake_path = os.path.join(DATA_DIR, "Fake.csv")
    true_path = os.path.join(DATA_DIR, "True.csv")

    if not os.path.exists(fake_path) or not os.path.exists(true_path):
        return None

    print("Loading Kaggle dataset (Fake.csv + True.csv)...")
    df_fake = pd.read_csv(fake_path)
    df_true = pd.read_csv(true_path)

    df_fake['label'] = 'FAKE'
    df_true['label'] = 'REAL'

    # Combine title + text for better features
    df_fake['text'] = df_fake['title'].astype(str) + " " + df_fake['text'].astype(str)
    df_true['text'] = df_true['title'].astype(str) + " " + df_true['text'].astype(str)

    df = pd.concat([df_fake[['text', 'label']], df_true[['text', 'label']]], ignore_index=True)
    print(f"✅ Loaded {len(df_fake)} fake + {len(df_true)} real = {len(df)} total articles")
    return df

def generate_synthetic_data():
    """Fallback: generate synthetic data if no dataset is available."""
    print("⚠️  Generating synthetic data (for testing only)...")
    fake_data = [
        "BREAKING: Scientists confirm the moon is made of cheese, government cover-up exposed",
        "Shocking discovery: Drinking bleach cures all diseases says anonymous doctor",
        "Aliens have landed in New York City, media blackout in effect worldwide",
        "Government secretly implanting chips through vaccines new report reveals",
        "Celebrity found living double life as secret agent for foreign government",
    ] * 500
    real_data = [
        "The Federal Reserve announced a quarter-point interest rate increase today",
        "New study published in Nature shows promising results for cancer treatment",
        "The Senate passed the infrastructure bill with bipartisan support",
        "Global temperatures rose 1.1 degrees Celsius above pre-industrial levels",
        "Tech companies report strong quarterly earnings amid market volatility",
    ] * 500

    df = pd.DataFrame({
        'text': fake_data + real_data,
        'label': ['FAKE'] * len(fake_data) + ['REAL'] * len(real_data)
    })
    return df

def clean(text):
    """Clean and preprocess text for NLP pipeline."""
    text = str(text).lower()
    text = re.sub(r'https?://\S+|www\.\S+', '', text)       # Remove URLs
    text = re.sub(r'<.*?>', '', text)                         # Remove HTML tags
    text = re.sub(r'[%s]' % re.escape(string.punctuation), ' ', text)  # Remove punctuation
    text = re.sub(r'\w*\d\w*', '', text)                      # Remove words with digits
    text = re.sub(r'\s+', ' ', text).strip()                  # Collapse whitespace
    return text

def main():
    # ── Step 1: Load Dataset ──
    df = None

    # Try loading existing Kaggle CSV files first
    df = load_kaggle_csv()

    # If not found, try downloading from Kaggle API
    if df is None:
        if download_from_kaggle():
            df = load_kaggle_csv()

    # If still no data, try manual CSV in project root
    if df is None:
        for csv_name in ["Fake.csv", "True.csv"]:
            if os.path.exists(csv_name):
                print(f"Found {csv_name} in project root, moving to {DATA_DIR}/")
                os.makedirs(DATA_DIR, exist_ok=True)
                os.rename(csv_name, os.path.join(DATA_DIR, csv_name))
        df = load_kaggle_csv()

    # Last resort: synthetic data
    if df is None:
        print("\n" + "="*60)
        print("  DATASET NOT FOUND!")
        print("  Download manually from Kaggle:")
        print("  https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset")
        print(f"  Place Fake.csv and True.csv in the '{DATA_DIR}/' folder")
        print("="*60 + "\n")
        df = generate_synthetic_data()

    # ── Step 2: Clean Text ──
    print("Cleaning text...")
    df['text'] = df['text'].apply(clean)

    # ── Step 3: Encode Labels ──
    df['label_num'] = df['label'].map({'REAL': 1, 'FAKE': 0})
    df['label_num'] = df['label_num'].fillna(0)

    X = df['text']
    y = df['label_num']

    # ── Step 4: Split Data ──
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Train: {len(X_train)} | Test: {len(X_test)}")

    # ── Step 5: TF-IDF Vectorization ──
    print("Vectorizing with TF-IDF...")
    vec = TfidfVectorizer(stop_words="english", max_df=0.7, max_features=50000)
    X_train_vec = vec.fit_transform(X_train)
    X_test_vec = vec.transform(X_test)

    # ── Step 6: Train Model ──
    print("Training Logistic Regression model...")
    model = LogisticRegression(class_weight="balanced", max_iter=1000, C=1.0)
    model.fit(X_train_vec, y_train)

    # ── Step 7: Evaluate ──
    preds = model.predict(X_test_vec)
    acc = accuracy_score(y_test, preds)
    print(f"\n{'='*40}")
    print(f"  Model Accuracy: {acc:.4f} ({acc*100:.2f}%)")
    print(f"{'='*40}\n")
    print(classification_report(y_test, preds, target_names=['FAKE', 'REAL']))

    # ── Step 8: Save Model ──
    print("Saving model and vectorizer...")
    with open("model.pkl", "wb") as f:
        pickle.dump(model, f)
    with open("vectorizer.pkl", "wb") as f:
        pickle.dump(vec, f)

    print("✅ Done! Files saved:")
    print("   → model.pkl")
    print("   → vectorizer.pkl")
    print("\nRun the app with: python3 app.py")

if __name__ == '__main__':
    main()
