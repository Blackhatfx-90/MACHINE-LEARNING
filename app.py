from flask import Flask, render_template, request, jsonify
import pickle, re, string, os, math

# Use absolute paths for Vercel serverless compatibility
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
app = Flask(__name__, template_folder=os.path.join(BASE_DIR, "templates"))

# ─── Load ML model ───
model = pickle.load(open(os.path.join(BASE_DIR, "model.pkl"), "rb"))
vec = pickle.load(open(os.path.join(BASE_DIR, "vectorizer.pkl"), "rb"))

# ═══════════════════════════════════════════════════════════════
# FAKE NEWS DETECTION — HYBRID APPROACH
# Combines ML model prediction with linguistic analysis
# ═══════════════════════════════════════════════════════════════

# ─── Sensational / clickbait / conspiracy phrases (strong FAKE signals) ───
FAKE_PHRASES = [
    'you wont believe', 'you won t believe', 'doctors hate',
    'one weird trick', 'this one trick', 'secret they',
    'what they dont want', 'what they don t want',
    'shocking truth', 'jaw dropping', 'mind blowing',
    'illuminati', 'new world order', 'deep state',
    'chemtrails', 'flat earth', 'crisis actor',
    'false flag', 'mind control', 'sheeple',
    'wake up people', 'government cover up', 'government hiding',
    'big pharma hiding', 'media blackout',
    'mainstream media lies', 'miracle cure', 'cures all',
    'secret cure', 'drinking bleach', 'bleach cures',
    'vaccines cause autism', 'implanting chips', 'microchip vaccine',
    '5g causes', '5g spread', 'aliens landed', 'alien invasion',
    'moon is made of cheese', 'earth is flat', 'zombie apocalypse',
    'share before deleted', 'share before they', 'they will delete',
    'banned from sharing', 'anonymous doctor says',
    'scientists baffled by this', 'exposed exposed',
]

# ─── Sensational single words ───
SENSATIONAL_WORDS = [
    'shocking', 'unbelievable', 'incredible', 'horrifying', 'terrifying',
    'disgusting', 'outrageous', 'insane', 'crazy', 'mindblowing',
    'explosive', 'bombshell', 'devastating', 'nightmare', 'catastrophic',
    'evil', 'sinister', 'corrupt', 'betrayal', 'treason', 'traitor',
    'destroy', 'destroyed', 'annihilate', 'demolish',
]

# ─── Journalistic / institutional phrases (strong REAL signals) ───
REAL_PHRASES = [
    'reuters', 'associated press', 'according to',
    'officials said', 'officials stated', 'spokesperson said',
    'spokesperson confirmed', 'press conference', 'official statement',
    'confirmed by', 'in a statement', 'told reporters',
    'prime minister', 'federal reserve', 'supreme court',
    'united nations', 'world health organization',
    'study published', 'published in nature', 'published in lancet',
    'peer reviewed', 'researchers found', 'researchers at',
    'clinical trial', 'quarterly earnings', 'fiscal year',
    'gdp growth', 'inflation rate', 'interest rate',
    'signed into law', 'bipartisan support', 'passed the bill',
    'election commission', 'press trust of india',
]

def clean(text):
    """Clean text — same as training pipeline."""
    text = str(text).lower()
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[%s]' % re.escape(string.punctuation), ' ', text)
    text = re.sub(r'\w*\d\w*', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def compute_linguistic_score(original_text, cleaned_text):
    """
    Compute a linguistic fake-score from 0 (very real) to 1 (very fake).
    Analyzes writing style, emotional language, and structural patterns.
    """
    text_lower = cleaned_text.lower()
    words = cleaned_text.split()
    word_count = len(words)
    if word_count == 0:
        return 0.5
    
    score = 0.0
    signals = 0
    
    # ─── 1. Fake phrase matching (strong signal) ───
    fake_phrase_hits = sum(1 for p in FAKE_PHRASES if p in text_lower)
    if fake_phrase_hits >= 2:
        score += 0.9
        signals += 3
    elif fake_phrase_hits == 1:
        score += 0.7
        signals += 2
    
    # ─── 2. Real phrase matching (strong signal) ───
    real_phrase_hits = sum(1 for p in REAL_PHRASES if p in text_lower)
    if real_phrase_hits >= 2:
        score += 0.05
        signals += 3
    elif real_phrase_hits == 1:
        score += 0.15
        signals += 2
    
    # ─── 3. Sensational word density ───
    sensational_count = sum(1 for w in SENSATIONAL_WORDS if w in text_lower)
    sensational_ratio = sensational_count / max(word_count, 1)
    if sensational_count >= 3:
        score += 0.8
        signals += 2
    elif sensational_count >= 2:
        score += 0.65
        signals += 1
    elif sensational_count == 1:
        score += 0.5
        signals += 1
    else:
        score += 0.4  # No sensational words = slightly leans real
        signals += 1
    
    # ─── 4. ALL CAPS usage (sensationalism) ───
    orig_words = original_text.split()
    caps_words = sum(1 for w in orig_words if w.isupper() and len(w) > 2)
    caps_ratio = caps_words / max(len(orig_words), 1)
    if caps_ratio > 0.5 and len(orig_words) > 3:
        score += 0.8
        signals += 1
    elif caps_ratio > 0.25:
        score += 0.6
        signals += 1
    else:
        score += 0.35
        signals += 1
    
    # ─── 5. Exclamation / question mark excess ───
    excl_count = original_text.count('!') + original_text.count('?')
    if excl_count >= 3:
        score += 0.75
        signals += 1
    elif excl_count >= 2:
        score += 0.55
        signals += 1
    else:
        score += 0.35
        signals += 1
    
    # ─── 6. Vague attribution ("they say", "sources say") ───
    vague_phrases = ['they say', 'sources say', 'people are saying',
                     'everyone knows', 'it is known', 'some say',
                     'many believe', 'rumor has it', 'word is that']
    vague_count = sum(1 for p in vague_phrases if p in text_lower)
    if vague_count > 0:
        score += 0.65
        signals += 1
    
    # ─── 7. Absolute language ("always", "never", "all", "every") ───
    absolutes = ['always', 'never', 'all of them', 'every single',
                 'nobody', 'everyone', 'completely', 'totally',
                 'absolutely', 'definitely', 'proven fact', '100 percent']
    absolute_count = sum(1 for a in absolutes if a in text_lower)
    if absolute_count >= 2:
        score += 0.65
        signals += 1
    elif absolute_count == 1:
        score += 0.5
        signals += 1
    
    # ─── 8. Extreme / unrealistic claims detection ───
    extreme_claim = False
    
    # Extreme price drops: "falls to $1", "drops to $5", "crashes to $0"
    price_drop = re.search(
        r'(falls?|drops?|crash|crashes|crashed|plummets?|sinks?|tumbles?)'
        r'\s+(to|below)\s+\$?\s*\d{1,2}(\.\d+)?\b',
        text_lower
    )
    if price_drop:
        extreme_claim = True
    
    # Extreme percentage claims: "drops 90%", "falls 99%", "increases 1000%"
    pct_match = re.search(
        r'(drops?|falls?|crash|loses?|gains?|increases?|rises?|surges?)'
        r'\s+\d+\s*(%|percent)',
        text_lower
    )
    if pct_match:
        num_match = re.search(r'(\d+)\s*(%|percent)', text_lower)
        if num_match:
            pct_val = int(num_match.group(1))
            if pct_val >= 80 or pct_val >= 500:  # 80%+ drop or 500%+ gain
                extreme_claim = True
    
    # "Free" / too good to be true
    too_good = re.search(
        r'(free money|free iphone|free bitcoin|won a|you have won|'
        r'congratulations you|claim your prize|lottery winner|'
        r'send this to|forward this to)',
        text_lower
    )
    if too_good:
        extreme_claim = True
    
    # Extreme world event claims (without proper source attribution)
    extreme_events = [
        'world war 3', 'world war iii', 'nuclear bomb dropped',
        'president assassinated', 'president killed', 'country destroyed',
        'city destroyed', 'millions dead', 'billions dead',
        'end of the world', 'apocalypse', 'martial law declared',
        'internet shutdown', 'dollar collapsed', 'economy collapsed',
    ]
    extreme_event_hits = sum(1 for e in extreme_events if e in text_lower)
    if extreme_event_hits > 0 and real_phrase_hits == 0:
        extreme_claim = True
    
    if extreme_claim:
        score += 0.85
        signals += 2  # Double weight for extreme claims
    
    # Compute average (weighted by signals)
    if signals > 0:
        avg_score = score / signals
    else:
        avg_score = 0.5
    
    return avg_score

def predict_news(news_text):
    """
    Hybrid prediction combining ML model + linguistic analysis.
    Returns 'REAL' or 'FAKE'.
    """
    cleaned = clean(news_text)
    word_count = len(cleaned.split())
    
    # ─── ML Model Prediction ───
    vector = vec.transform([cleaned])
    prob = model.predict_proba(vector)[0]
    ml_prob_real = prob[1]  # probability of REAL (class 1)
    
    # ─── Linguistic Analysis ───
    ling_score = compute_linguistic_score(news_text, news_text.lower())
    # ling_score: 0 = very real, 1 = very fake
    ling_prob_real = 1.0 - ling_score
    
    # ─── Combine Predictions ───
    if word_count >= 80:
        # LONG TEXT: ML model is very reliable → heavy ML weight
        ml_weight = 0.85
        ling_weight = 0.15
    elif word_count >= 30:
        # MEDIUM TEXT: ML model is somewhat reliable
        ml_weight = 0.55
        ling_weight = 0.45
    else:
        # SHORT TEXT (headlines): ML model is unreliable → linguistic dominates
        ml_weight = 0.30
        ling_weight = 0.70
    
    combined_prob_real = (ml_prob_real * ml_weight) + (ling_prob_real * ling_weight)
    
    # ─── Final Decision ───
    if combined_prob_real >= 0.50:
        return "REAL"
    else:
        return "FAKE"

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
