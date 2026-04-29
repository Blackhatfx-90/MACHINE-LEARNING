from flask import Flask, render_template, request, jsonify
import pickle, re, string, os

# Use absolute paths for Vercel serverless compatibility
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
app = Flask(__name__, template_folder=os.path.join(BASE_DIR, "templates"))

# ─── Load models ───
# Try loading dual models first (headline + full article)
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
    # Fallback: single model (the current one trained on title+body)
    model_single = pickle.load(open(os.path.join(BASE_DIR, "model.pkl"), "rb"))
    vec_single = pickle.load(open(os.path.join(BASE_DIR, "vectorizer.pkl"), "rb"))
    print("⚠️  Single model loaded — using smart prediction for short text")

# ─── Fake news indicator keywords (common in misinformation) ───
FAKE_INDICATORS = {
    'shocking', 'breaking', 'exposed', 'cover up', 'coverup', 'conspiracy',
    'they dont want you to know', 'mainstream media', 'hoax', 'scam',
    'miracle cure', 'secret', 'banned', 'illuminati', 'deep state',
    'wake up', 'sheeple', 'big pharma', 'chemtrails', 'flat earth',
    'crisis actor', 'false flag', 'mind control', 'new world order',
    'you wont believe', 'doctors hate', 'one weird trick',
    'government hiding', 'media blackout', 'alien', 'aliens landed',
    'drinking bleach', 'made of cheese', 'implanting chips',
    'anonymous source says', 'insiders reveal'
}

# ─── Real news indicator patterns (common in legitimate journalism) ───
REAL_INDICATORS = {
    'reuters', 'associated press', 'ap news', 'according to', 'officials said',
    'the president', 'prime minister', 'federal reserve', 'supreme court',
    'united nations', 'nato', 'european union', 'world health organization',
    'study published', 'researchers found', 'scientists discovered',
    'quarterly earnings', 'fiscal year', 'gdp', 'inflation rate',
    'bipartisan', 'legislation', 'senate', 'congress', 'parliament',
    'celsius', 'climate change', 'global warming',
    'spokesperson said', 'press conference', 'white house',
    'stock market', 'wall street', 'dow jones', 'nasdaq',
    'launched', 'mission', 'satellite', 'isro', 'nasa', 'space',
    'bilateral', 'summit', 'trade deal', 'sanctions', 'diplomatic',
    'earthquake', 'hurricane', 'typhoon', 'tsunami', 'flood',
    'election', 'voters', 'polling', 'ballot', 'democrat', 'republican',
    'ministry', 'cabinet', 'governor', 'mayor', 'commissioner'
}

# Word count threshold for headline vs article
HEADLINE_WORD_THRESHOLD = 80

def clean(text):
    text = text.lower()
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[%s]' % re.escape(string.punctuation), ' ', text)
    # NOT removing digit-words — dates and numbers are important signals
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def count_indicator_matches(text_lower, indicator_set):
    """Count how many indicator phrases are found in the text."""
    count = 0
    for phrase in indicator_set:
        if phrase in text_lower:
            count += 1
    return count

def smart_predict_single_model(news_text, cleaned_text):
    """
    Smart prediction for the SINGLE model (trained on full articles).
    Compensates for the model's bias when given short headline text.
    """
    word_count = len(cleaned_text.split())
    
    # Get model prediction
    vector = vec_single.transform([cleaned_text])
    raw_pred = model_single.predict(vector)[0]
    prob = model_single.predict_proba(vector)[0]
    # prob[0] = probability of FAKE (class 0)
    # prob[1] = probability of REAL (class 1)
    prob_fake = prob[0]
    prob_real = prob[1]
    
    # Count how many non-zero features the vectorizer found
    nnz = vector.nnz  # number of non-zero entries in sparse vector
    
    # ─── Keyword-based signals ───
    text_lower = cleaned_text.lower()
    fake_matches = count_indicator_matches(text_lower, FAKE_INDICATORS)
    real_matches = count_indicator_matches(text_lower, REAL_INDICATORS)
    
    # ─── Decision Logic ───
    # For SHORT text (headlines), the TF-IDF model is unreliable because
    # it was trained on full articles. We use a hybrid approach:
    
    if word_count < 50:
        # SHORT TEXT MODE — model is unreliable, use hybrid scoring
        
        # Start with a neutral base (50/50)
        score_real = 0.5
        
        # Factor 1: Model prediction (low weight for short text)
        model_weight = min(0.3, word_count / 150)  # 0 to 0.3 based on length
        score_real += (prob_real - 0.5) * model_weight
        
        # Factor 2: Keyword signals (high weight for short text)
        keyword_weight = 0.35
        if fake_matches > 0 and real_matches == 0:
            score_real -= keyword_weight * min(fake_matches, 3) / 3
        elif real_matches > 0 and fake_matches == 0:
            score_real += keyword_weight * min(real_matches, 3) / 3
        elif real_matches > fake_matches:
            score_real += keyword_weight * 0.5
        elif fake_matches > real_matches:
            score_real -= keyword_weight * 0.5
        
        # Factor 3: If text has NO fake indicators and reads like normal news
        # (no sensationalism), lean toward REAL
        has_sensational = any(w in text_lower for w in [
            'shocking', 'unbelievable', 'you wont believe', 'exposed',
            'breaking', 'urgent', 'miracle', 'secret', 'banned',
            'they dont want', 'cover up'
        ])
        
        if not has_sensational and fake_matches == 0:
            # Normal-sounding text without fake indicators → slight REAL bias
            score_real += 0.12
        
        # Factor 4: Feature density — if vectorizer found many matching words,
        # the model's prediction has more weight
        if nnz > 10:
            score_real += (prob_real - 0.5) * 0.15  # Extra model influence
        
        # Clamp score
        score_real = max(0.1, min(0.9, score_real))
        
        # Determine prediction
        pred = 1 if score_real >= 0.5 else 0
        confidence = round(abs(score_real - 0.5) * 2 * 100, 2)  # 0-100%
        confidence = max(confidence, 50.0)  # Minimum 50% confidence
        
        # Cap confidence for very short text
        if word_count < 10:
            confidence = min(confidence, 65.0)
        elif word_count < 25:
            confidence = min(confidence, 78.0)
    
    else:
        # LONG TEXT MODE — model is more reliable
        # Still apply keyword adjustments but with lower weight
        
        score_real = prob_real
        
        # Slight keyword adjustment
        if fake_matches >= 3 and real_matches == 0:
            score_real -= 0.1
        elif real_matches >= 3 and fake_matches == 0:
            score_real += 0.05
        
        score_real = max(0.05, min(0.95, score_real))
        pred = 1 if score_real >= 0.5 else 0
        confidence = round(max(score_real, 1 - score_real) * 100, 2)
    
    label = "REAL" if pred == 1 else "FAKE"
    return label, confidence

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    news = request.json.get('news', '')
    cleaned = clean(news)
    word_count = len(cleaned.split())
    
    if DUAL_MODEL:
        # ── Dual model: pick the right model based on text length ──
        if word_count <= HEADLINE_WORD_THRESHOLD:
            model = model_headline
            vec = vec_headline
            mode = "headline"
        else:
            model = model_full
            vec = vec_full
            mode = "article"
        
        vector = vec.transform([cleaned])
        pred = model.predict(vector)[0]
        prob = model.predict_proba(vector)[0]
        confidence = round(max(prob) * 100, 2)
        
        if word_count < 10:
            confidence = min(confidence, 65.0)
        
        label = "REAL" if pred == 1 else "FAKE"
    else:
        # ── Single model: use smart hybrid prediction ──
        mode = "headline" if word_count <= HEADLINE_WORD_THRESHOLD else "article"
        label, confidence = smart_predict_single_model(news, cleaned)
    
    return jsonify({
        'result': label,
        'confidence': confidence,
        'mode': mode,
        'word_count': word_count
    })

if __name__ == '__main__':
    app.run(debug=True)
