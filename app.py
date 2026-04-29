from flask import Flask, render_template, request, jsonify
import pickle, re, string, os

# Use absolute paths for Vercel serverless compatibility
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
app = Flask(__name__, template_folder=os.path.join(BASE_DIR, "templates"))

# ─── Load models ───
headline_model_path = os.path.join(BASE_DIR, "model_headline.pkl")
headline_vec_path = os.path.join(BASE_DIR, "vectorizer_headline.pkl")
full_model_path = os.path.join(BASE_DIR, "model_full.pkl")
full_vec_path = os.path.join(BASE_DIR, "vectorizer_full.pkl")

DUAL_MODEL = False

if os.path.exists(headline_model_path) and os.path.exists(full_model_path):
    model_headline = pickle.load(open(headline_model_path, "rb"))
    vec_headline = pickle.load(open(headline_vec_path, "rb"))
    model_full = pickle.load(open(full_model_path, "rb"))
    vec_full = pickle.load(open(full_vec_path, "rb"))
    DUAL_MODEL = True
    print("✅ Dual model loaded (headline + full article)")
else:
    model_single = pickle.load(open(os.path.join(BASE_DIR, "model.pkl"), "rb"))
    vec_single = pickle.load(open(os.path.join(BASE_DIR, "vectorizer.pkl"), "rb"))
    print("⚠️  Single model loaded — using smart prediction for short text")

# ─── Strong FAKE indicators (sensationalism / misinformation patterns) ───
STRONG_FAKE = [
    'you wont believe', 'doctors hate', 'one weird trick', 'miracle cure',
    'government hiding', 'media blackout', 'aliens landed', 'cover up',
    'implanting chips', 'made of cheese', 'drinking bleach', 'flat earth',
    'chemtrails', 'illuminati', 'deep state', 'sheeple', 'wake up people',
    'crisis actor', 'false flag', 'new world order', 'they dont want you',
    'anonymous doctor', 'secret cure', 'banned video', 'mind control',
    'big pharma hiding', 'mainstream media lies'
]

# ─── Strong REAL indicators (legitimate journalism patterns) ───
STRONG_REAL = [
    'reuters', 'associated press', 'according to officials',
    'spokesperson said', 'press conference', 'published in nature',
    'published in lancet', 'peer reviewed', 'official statement',
    'quarterly earnings report', 'fiscal quarter',
    'bipartisan support', 'passed the bill', 'signed into law'
]

def clean(text):
    text = text.lower()
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[%s]' % re.escape(string.punctuation), ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def smart_predict(news_text, cleaned_text):
    """
    Balanced prediction using the single model (trained on full articles).
    For short text, adjusts the decision threshold to compensate for
    sparse TF-IDF vectors, without blindly favoring REAL or FAKE.
    """
    word_count = len(cleaned_text.split())
    text_lower = cleaned_text.lower()

    # Get model prediction probabilities
    vector = vec_single.transform([cleaned_text])
    prob = model_single.predict_proba(vector)[0]
    prob_fake = prob[0]  # class 0 = FAKE
    prob_real = prob[1]  # class 1 = REAL

    # Count keyword matches
    fake_hits = sum(1 for p in STRONG_FAKE if p in text_lower)
    real_hits = sum(1 for p in STRONG_REAL if p in text_lower)

    # ─── Decision Logic ───
    if word_count >= 50:
        # LONG TEXT: model is reliable, trust it directly
        # Small keyword nudge only for extreme cases
        adjusted_prob_real = prob_real
        if fake_hits >= 3 and real_hits == 0:
            adjusted_prob_real -= 0.08
        elif real_hits >= 2 and fake_hits == 0:
            adjusted_prob_real += 0.05
        
        label = "REAL" if adjusted_prob_real >= 0.5 else "FAKE"

    else:
        # SHORT TEXT: model has sparse-vector bias toward FAKE
        # Use a SLIGHTLY lower threshold (0.42 instead of 0.5) to compensate
        # but NOT so low that fake news passes through
        threshold = 0.42

        adjusted_prob_real = prob_real

        # Keyword overrides — these are strong signals for short text
        if fake_hits >= 2:
            # Multiple strong fake indicators → definitely FAKE
            label = "FAKE"
        elif real_hits >= 2:
            # Multiple strong real indicators → likely REAL
            label = "REAL"
        elif fake_hits == 1 and real_hits == 0:
            # One fake indicator, push threshold up (harder to be REAL)
            label = "REAL" if adjusted_prob_real >= 0.55 else "FAKE"
        elif real_hits == 1 and fake_hits == 0:
            # One real indicator, keep adjusted threshold
            label = "REAL" if adjusted_prob_real >= 0.38 else "FAKE"
        else:
            # No strong keyword signals — use adjusted threshold
            label = "REAL" if adjusted_prob_real >= threshold else "FAKE"

    return label

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    news = request.json.get('news', '')
    cleaned = clean(news)
    word_count = len(cleaned.split())

    if DUAL_MODEL:
        if word_count <= 80:
            model, vec = model_headline, vec_headline
        else:
            model, vec = model_full, vec_full

        vector = vec.transform([cleaned])
        pred = model.predict(vector)[0]
        label = "REAL" if pred == 1 else "FAKE"
    else:
        label = smart_predict(news, cleaned)

    return jsonify({
        'result': label
    })

if __name__ == '__main__':
    app.run(debug=True)
