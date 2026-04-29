import pandas as pd
import pickle
import re
import string
import os
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

    # ── Create TWO versions of training data ──
    # 1. Title-only (for headline detection)
    df_fake['title_text'] = df_fake['title'].astype(str)
    df_true['title_text'] = df_true['title'].astype(str)
    
    # 2. Full article (title + body for detailed article analysis)
    df_fake['full_text'] = df_fake['title'].astype(str) + " " + df_fake['text'].astype(str)
    df_true['full_text'] = df_true['title'].astype(str) + " " + df_true['text'].astype(str)

    df = pd.concat([
        df_fake[['title_text', 'full_text', 'label']], 
        df_true[['title_text', 'full_text', 'label']]
    ], ignore_index=True)
    print(f"✅ Loaded {len(df_fake)} fake + {len(df_true)} real = {len(df)} total articles")
    return df

def generate_synthetic_data():
    """Fallback: generate synthetic data if no dataset is available."""
    print("⚠️  Generating synthetic data (for testing only)...")
    fake_data_titles = [
        "BREAKING: Scientists confirm the moon is made of cheese, government cover-up exposed",
        "Shocking discovery: Drinking bleach cures all diseases says anonymous doctor",
        "Aliens have landed in New York City, media blackout in effect worldwide",
        "Government secretly implanting chips through vaccines new report reveals",
        "Celebrity found living double life as secret agent for foreign government",
    ] * 500
    real_data_titles = [
        "The Federal Reserve announced a quarter-point interest rate increase today",
        "New study published in Nature shows promising results for cancer treatment",
        "The Senate passed the infrastructure bill with bipartisan support",
        "Global temperatures rose 1.1 degrees Celsius above pre-industrial levels",
        "Tech companies report strong quarterly earnings amid market volatility",
    ] * 500

    df = pd.DataFrame({
        'title_text': fake_data_titles + real_data_titles,
        'full_text': fake_data_titles + real_data_titles,
        'label': ['FAKE'] * len(fake_data_titles) + ['REAL'] * len(real_data_titles)
    })
    return df

def clean(text):
    """Clean and preprocess text for NLP pipeline."""
    text = str(text).lower()
    text = re.sub(r'https?://\S+|www\.\S+', '', text)       # Remove URLs
    text = re.sub(r'<.*?>', '', text)                         # Remove HTML tags
    text = re.sub(r'[%s]' % re.escape(string.punctuation), ' ', text)  # Remove punctuation
    # NOTE: Removed the digit-word filter — it was stripping dates/numbers 
    # that are important signals for real news (e.g., "2024", "$1.5 billion")
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
    df['title_text'] = df['title_text'].apply(clean)
    df['full_text'] = df['full_text'].apply(clean)

    # ── Step 3: Encode Labels ──
    df['label_num'] = df['label'].map({'REAL': 1, 'FAKE': 0})
    df['label_num'] = df['label_num'].fillna(0)

    y = df['label_num']

    # ══════════════════════════════════════════════════════════
    # MODEL 1: HEADLINE MODEL (trained on TITLES only)
    # This handles short text / headline inputs accurately
    # ══════════════════════════════════════════════════════════
    print("\n" + "="*50)
    print("  TRAINING HEADLINE MODEL (titles only)")
    print("="*50)
    
    X_title = df['title_text']
    X_train_t, X_test_t, y_train_t, y_test_t = train_test_split(
        X_title, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Train: {len(X_train_t)} | Test: {len(X_test_t)}")

    # TF-IDF tuned for short text (headlines)
    vec_title = TfidfVectorizer(
        stop_words="english",
        max_df=0.85,          # Higher threshold — keep more common words
        min_df=2,             # Remove very rare words
        max_features=30000,
        ngram_range=(1, 2),   # Bigrams capture headline patterns better
        sublinear_tf=True     # Apply log normalization to TF
    )
    X_train_t_vec = vec_title.fit_transform(X_train_t)
    X_test_t_vec = vec_title.transform(X_test_t)

    model_title = LogisticRegression(
        class_weight="balanced", max_iter=1000, C=1.0, solver='lbfgs'
    )
    model_title.fit(X_train_t_vec, y_train_t)

    preds_t = model_title.predict(X_test_t_vec)
    acc_t = accuracy_score(y_test_t, preds_t)
    print(f"\n  Headline Model Accuracy: {acc_t:.4f} ({acc_t*100:.2f}%)\n")
    print(classification_report(y_test_t, preds_t, target_names=['FAKE', 'REAL']))

    # ══════════════════════════════════════════════════════════
    # MODEL 2: FULL ARTICLE MODEL (trained on title + body)
    # This handles longer, detailed article text
    # ══════════════════════════════════════════════════════════
    print("\n" + "="*50)
    print("  TRAINING FULL ARTICLE MODEL (title + body)")
    print("="*50)

    X_full = df['full_text']
    X_train_f, X_test_f, y_train_f, y_test_f = train_test_split(
        X_full, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Train: {len(X_train_f)} | Test: {len(X_test_f)}")

    vec_full = TfidfVectorizer(
        stop_words="english",
        max_df=0.7,
        min_df=3,
        max_features=50000,
        ngram_range=(1, 2),
        sublinear_tf=True
    )
    X_train_f_vec = vec_full.fit_transform(X_train_f)
    X_test_f_vec = vec_full.transform(X_test_f)

    model_full = LogisticRegression(
        class_weight="balanced", max_iter=1000, C=1.0, solver='lbfgs'
    )
    model_full.fit(X_train_f_vec, y_train_f)

    preds_f = model_full.predict(X_test_f_vec)
    acc_f = accuracy_score(y_test_f, preds_f)
    print(f"\n  Full Article Model Accuracy: {acc_f:.4f} ({acc_f*100:.2f}%)\n")
    print(classification_report(y_test_f, preds_f, target_names=['FAKE', 'REAL']))

    # ── Step 8: Save Both Models ──
    print("\nSaving models and vectorizers...")
    
    # Headline model
    with open("model_headline.pkl", "wb") as f:
        pickle.dump(model_title, f)
    with open("vectorizer_headline.pkl", "wb") as f:
        pickle.dump(vec_title, f)
    
    # Full article model
    with open("model_full.pkl", "wb") as f:
        pickle.dump(model_full, f)
    with open("vectorizer_full.pkl", "wb") as f:
        pickle.dump(vec_full, f)
    
    # Also save as the default model.pkl / vectorizer.pkl (headline model as default)
    # since most users paste headlines
    with open("model.pkl", "wb") as f:
        pickle.dump(model_title, f)
    with open("vectorizer.pkl", "wb") as f:
        pickle.dump(vec_title, f)

    print("\n✅ Done! Files saved:")
    print("   → model_headline.pkl + vectorizer_headline.pkl (for short headlines)")
    print("   → model_full.pkl + vectorizer_full.pkl (for full articles)")
    print("   → model.pkl + vectorizer.pkl (default = headline model)")
    print(f"\n   Headline Model Accuracy: {acc_t*100:.2f}%")
    print(f"   Full Article Accuracy:   {acc_f*100:.2f}%")
    print("\nRun the app with: python3 app.py")

if __name__ == '__main__':
    main()
