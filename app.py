from flask import Flask, render_template, request, jsonify
import json, re, string, os, math

# Use absolute paths for Vercel serverless compatibility
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
app = Flask(__name__, template_folder=os.path.join(BASE_DIR, "templates"))

# ═══════════════════════════════════════════════════════════════
# WORD-SCORE CLASSIFIER (learned from 44K training articles)
# Each word has a score: positive = fake-leaning, negative = real-leaning
# This is essentially a Naive Bayes classifier built from training data
# ═══════════════════════════════════════════════════════════════

# Load word scores (precomputed from Fake.csv + True.csv titles)
word_stats_path = os.path.join(BASE_DIR, "word_stats.json")
with open(word_stats_path, 'r') as f:
    stats = json.load(f)

WORD_SCORES = stats['word_scores']  # word -> log2(fake_freq / real_freq)
print(f"✅ Loaded {len(WORD_SCORES)} word scores from training data")

# ─── Also try loading sklearn model for long articles ───
ML_MODEL = None
try:
    import pickle
    model_path = os.path.join(BASE_DIR, "model.pkl")
    vec_path = os.path.join(BASE_DIR, "vectorizer.pkl")
    ML_MODEL = pickle.load(open(model_path, "rb"))
    ML_VEC = pickle.load(open(vec_path, "rb"))
    print("✅ ML model loaded for long article analysis")
except Exception as e:
    print(f"⚠️  ML model not available: {e}")

# Stop words
STOP_WORDS = set('a an the is are was were be been being have has had do does did '
                 'will would shall should may might can could of in to for on with '
                 'at by from as into about above after before between through during '
                 'and or but not no nor so yet both either neither each every all '
                 'any few more most other some such than too very also how what '
                 'which who whom this that these those i me my we our you your '
                 'he him his she her it its they them their'.split())

def clean_for_scoring(text):
    """Clean text for word-score classification."""
    text = str(text).lower()
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[%s]' % re.escape(string.punctuation), ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def clean_for_ml(text):
    """Clean text for ML model (must match training pipeline exactly)."""
    text = str(text).lower()
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[%s]' % re.escape(string.punctuation), ' ', text)
    text = re.sub(r'\w*\d\w*', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# ─── Absurd claims / known misinformation patterns ───
# These catch things the training data (2017) can't detect
ABSURD_CLAIMS = [
    # Celebrities/non-politicians as political leaders
    r'\b(mia khalifa|kim kardashian|kanye west|elon musk|pewdiepie|'
    r'mr beast|justin bieber|taylor swift|drake|rihanna|beyonce|'
    r'cristiano ronaldo|messi|neymar|salman khan|shahrukh khan|'
    r'amitabh bachchan|virat kohli|sachin)\b.{0,30}'
    r'\b(president|prime minister|pm|king|queen|ruler|dictator|'
    r'chief minister|governor|senator|minister)\b',
    
    # Reverse: political title then celebrity
    r'\b(president|prime minister|pm|king|queen|ruler|chief minister)\b'
    r'.{0,30}\b(mia khalifa|kim kardashian|kanye west|pewdiepie|'
    r'mr beast|justin bieber|taylor swift|drake|rihanna|beyonce|'
    r'salman khan|shahrukh khan|amitabh bachchan)\b',
]

# Post-2017 misinformation topics (not in training data)
MODERN_FAKE_PATTERNS = [
    'microchip', 'microchips', '5g causes', '5g spread', '5g tower',
    'plandemic', 'scamdemic', 'covid hoax', 'corona hoax',
    'covid is fake', 'vaccine kill', 'vaccine death', 'depopulation',
    'great reset', 'qanon', 'adrenochrome', 'pizzagate',
    'bill gates chip', 'bill gates vaccine', 'soros funded',
    'antifa', 'stolen election', 'rigged election',
]

# Real journalism patterns (boost real score)
JOURNALISM_PATTERNS = [
    'according to', 'officials said', 'spokesperson', 'press conference',
    'reuters', 'associated press', 'study published', 'researchers found',
    'degrees celsius', 'percent increase', 'percent decrease',
    'quarterly', 'fiscal', 'bipartisan', 'legislation',
    'united nations', 'world health', 'supreme court',
    'nasa', 'isro', 'european space agency', 'space station',
    'rover', 'satellite launch', 'chandrayaan', 'mangalyaan',
    'pentagon', 'world bank', 'imf', 'federal bureau',
    'earthquake', 'magnitude', 'richter', 'tsunami warning',
    'hurricane category', 'cyclone', 'typhoon',
]

def word_score_predict(text):
    """
    Classify using word scores learned from 44K training articles,
    plus absurd claims detection and modern fake patterns.
    """
    cleaned = clean_for_scoring(text)
    text_lower = text.lower()
    words = [w for w in cleaned.split() if w not in STOP_WORDS and len(w) > 2]
    
    if not words:
        return "FAKE", 0.0
    
    # ─── Check absurd claims first (instant FAKE) ───
    for pattern in ABSURD_CLAIMS:
        if re.search(pattern, cleaned):
            return "FAKE", 10.0  # Very high fake score
    
    # ─── Word score from training data ───
    total_score = 0.0
    matched_words = 0
    
    for word in words:
        if word in WORD_SCORES:
            total_score += WORD_SCORES[word]
            matched_words += 1
    
    # ─── Modern fake pattern bonus ───
    modern_fake_hits = sum(1 for p in MODERN_FAKE_PATTERNS if p in cleaned)
    if modern_fake_hits > 0:
        total_score += modern_fake_hits * 3.0  # Strong fake boost
        matched_words += modern_fake_hits
    
    # ─── Real journalism pattern bonus ───
    journalism_hits = sum(1 for p in JOURNALISM_PATTERNS if p in cleaned)
    if journalism_hits > 0:
        total_score -= journalism_hits * 2.0  # Real boost
        matched_words += journalism_hits
    
    # Normalize
    if matched_words > 0:
        avg_score = total_score / matched_words
    else:
        # No words matched at all — unknown vocabulary
        # Could be anything, slight lean toward FAKE for safety
        avg_score = 0.3
    
    # Positive = FAKE, Negative = REAL
    if avg_score > 0:
        return "FAKE", avg_score
    else:
        return "REAL", avg_score

def ml_predict(text):
    """Use sklearn model for long article classification."""
    cleaned = clean_for_ml(text)
    vector = ML_VEC.transform([cleaned])
    pred = ML_MODEL.predict(vector)[0]
    prob = ML_MODEL.predict_proba(vector)[0]
    label = "REAL" if pred == 1 else "FAKE"
    return label, prob

def predict_news(news_text):
    """
    Main prediction function.
    - Short text (< 80 words): Use word-score classifier (trained on titles)
    - Long text (80+ words): Combine word-score + ML model
    """
    word_count = len(news_text.split())
    
    # Always get word-score prediction
    ws_label, ws_score = word_score_predict(news_text)
    
    if ML_MODEL is not None and word_count >= 80:
        # Long text: combine with ML model
        ml_label, ml_prob = ml_predict(news_text)
        
        # If both agree, use that
        if ws_label == ml_label:
            return ws_label
        
        # If they disagree, ML model is more reliable for long text
        # But if word-score has strong signal, trust it
        if abs(ws_score) > 3.0:
            return ws_label  # Strong word-score signal
        else:
            return ml_label  # Trust ML for long text
    
    # Short text: word-score classifier only
    return ws_label

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    news = request.json.get('news', '')
    label = predict_news(news)
    return jsonify({'result': label})

if __name__ == '__main__':
    app.run(debug=True)
