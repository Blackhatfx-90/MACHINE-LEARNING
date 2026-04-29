from flask import Flask, render_template, request, jsonify
import re, string, os

# Use absolute paths for Vercel serverless compatibility
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
app = Flask(__name__, template_folder=os.path.join(BASE_DIR, "templates"))

# ═══════════════════════════════════════════════════════════════
# RULE-BASED FAKE NEWS CLASSIFIER
# Uses linguistic pattern analysis instead of ML model
# Works accurately for both headlines AND full articles
# ═══════════════════════════════════════════════════════════════

# ─── FAKE patterns: sensationalism, clickbait, conspiracy ───
FAKE_PATTERNS = {
    # Clickbait / sensationalism
    'you wont believe': 3, 'you won t believe': 3,
    'doctors hate': 3, 'one weird trick': 3,
    'this one trick': 3, 'secret they': 2,
    'what they dont want': 3, 'what they don t want': 3,
    'shocking truth': 3, 'jaw dropping': 2,
    'mind blowing secret': 3, 'insiders reveal': 2,
    
    # Conspiracy theories
    'illuminati': 3, 'new world order': 3, 'deep state': 2,
    'chemtrails': 3, 'flat earth': 3, 'crisis actor': 3,
    'false flag': 2, 'mind control': 2, 'sheeple': 3,
    'wake up people': 2, 'open your eyes': 1,
    'government cover up': 3, 'government coverup': 3,
    'government hiding': 2, 'they are hiding': 2,
    'big pharma hiding': 3, 'big pharma doesn': 2,
    'media blackout': 2, 'mainstream media lies': 3,
    'mainstream media won t tell': 3,
    
    # Health misinformation
    'miracle cure': 3, 'cures all diseases': 3,
    'cure for cancer that': 2, 'secret cure': 3,
    'drinking bleach': 3, 'bleach cures': 3,
    'vaccines cause autism': 3, 'implanting chips': 3,
    'microchip vaccine': 3, 'vaccine kills': 2,
    '5g causes': 3, '5g spread': 3,
    
    # Outlandish claims
    'aliens landed': 3, 'alien invasion': 2,
    'moon is made of cheese': 3, 'earth is flat': 3,
    'zombie apocalypse': 2, 'time travel proven': 3,
    'mermaids are real': 3, 'bigfoot captured': 3,
    
    # Fake authority / anonymous sources with wild claims
    'anonymous doctor says': 2, 'anonymous source reveals': 1,
    'leaked documents show': 1, 'insider reveals': 1,
    'scientists baffled': 1, 'exposed exposed': 2,
    
    # Urgency / emotional manipulation
    'share before deleted': 3, 'share before they': 3,
    'banned from sharing': 2, 'they will delete this': 3,
    'breaking breaking': 2, 'urgent urgent': 2,
}

# ─── Words/phrases that ADD to fake score ───
FAKE_WORDS = [
    'shocking', 'unbelievable', 'hoax', 'scam', 'conspiracy',
    'coverup', 'banned', 'exposed', 'secret', 'miracle',
    'horrifying', 'terrifying', 'disgusting'
]

# ─── REAL patterns: journalistic language, institutional references ───
REAL_PATTERNS = {
    # News agency attribution
    'reuters': 2, 'associated press': 2, 'ap news': 2,
    'agence france presse': 2, 'afp': 1,
    'press trust of india': 2, 'pti': 1, 'ani news': 1,
    
    # Proper sourcing / attribution
    'according to': 1, 'officials said': 1, 'officials stated': 1,
    'spokesperson said': 2, 'spokesperson confirmed': 2,
    'press conference': 1, 'official statement': 1,
    'confirmed by': 1, 'announced that': 1,
    'in a statement': 1, 'told reporters': 1,
    
    # Government / institutional
    'the president': 1, 'prime minister': 1, 'chief minister': 1,
    'federal reserve': 2, 'supreme court': 1,
    'united nations': 1, 'european union': 1,
    'world health organization': 2, 'nato': 1,
    'ministry of': 1, 'department of': 1,
    'white house': 1, 'parliament': 1, 'congress': 1,
    'senate passed': 2, 'house passed': 2, 'signed into law': 2,
    'bipartisan': 1, 'legislation': 1,
    
    # Science / research
    'study published': 2, 'published in nature': 2,
    'published in lancet': 2, 'published in science': 2,
    'peer reviewed': 2, 'researchers found': 1,
    'researchers at': 1, 'scientists discovered': 1,
    'university of': 1, 'clinical trial': 2,
    
    # Finance / economy  
    'quarterly earnings': 2, 'fiscal year': 1, 'fiscal quarter': 1,
    'gdp growth': 1, 'inflation rate': 1, 'interest rate': 1,
    'stock market': 1, 'wall street': 1, 'dow jones': 1,
    'nasdaq': 1, 'sensex': 1, 'nifty': 1,
    
    # Space / science orgs
    'isro': 1, 'nasa': 1, 'esa': 1,
    'space mission': 1, 'satellite launch': 1,
    'chandrayaan': 1, 'mangalyaan': 1,
    
    # Elections / politics (neutral)
    'election commission': 1, 'voting': 0, 'ballot': 0,
    'polling station': 1, 'exit poll': 1,
    
    # Disasters (factual reporting)
    'earthquake': 0, 'magnitude': 1, 'hurricane': 0,
    'typhoon': 0, 'tsunami': 0, 'richter scale': 1,
    
    # International relations
    'bilateral talks': 1, 'summit': 0, 'trade deal': 1,
    'sanctions': 1, 'diplomatic': 1, 'ambassador': 1,
    'treaty': 1, 'ceasefire': 1,
}

def clean(text):
    text = text.lower()
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[%s]' % re.escape(string.punctuation), ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def classify_news(text):
    """
    Rule-based fake news classifier using linguistic pattern analysis.
    Returns 'REAL' or 'FAKE' based on pattern scoring.
    """
    cleaned = clean(text)
    
    # Calculate FAKE score
    fake_score = 0
    for pattern, weight in FAKE_PATTERNS.items():
        if pattern in cleaned:
            fake_score += weight
    
    # Count sensational words (each adds 0.5)
    for word in FAKE_WORDS:
        if word in cleaned:
            fake_score += 0.5
    
    # ALL CAPS words count (sensationalism signal)
    words = text.split()
    caps_words = sum(1 for w in words if w.isupper() and len(w) > 2)
    caps_ratio = caps_words / max(len(words), 1)
    if caps_ratio > 0.4 and len(words) > 3:
        fake_score += 2  # Lots of ALL CAPS = sensational
    
    # Excessive exclamation/question marks in original text
    excl_count = text.count('!') + text.count('?')
    if excl_count >= 3:
        fake_score += 1.5
    elif excl_count >= 2:
        fake_score += 0.5
    
    # Calculate REAL score
    real_score = 0
    for pattern, weight in REAL_PATTERNS.items():
        if pattern in cleaned:
            real_score += weight
    
    # Quotation marks suggest attributed quotes (journalistic)
    quote_count = text.count('"') + text.count("'") + text.count('\u201c') + text.count('\u201d')
    if quote_count >= 2:
        real_score += 0.5
    
    # Numbers/dates suggest factual reporting
    numbers = re.findall(r'\b\d+\.?\d*\b', cleaned)
    if len(numbers) >= 2:
        real_score += 0.5
    
    # ─── Decision ───
    # If both scores are 0 (no signals found), default to REAL
    # because normal news text without sensationalism is usually real
    if fake_score == 0 and real_score == 0:
        return "REAL"
    
    # If fake score is significantly higher → FAKE
    if fake_score >= 2 and fake_score > real_score:
        return "FAKE"
    
    # If fake has some signal but real has more → REAL
    if real_score > fake_score:
        return "REAL"
    
    # If fake_score is positive but low (< 2), and real_score is 0
    # This handles borderline cases — slight sensationalism doesn't make it fake
    if fake_score < 2:
        return "REAL"
    
    return "FAKE"

# ─── Try loading ML model as backup for long articles ───
ML_MODEL_AVAILABLE = False
try:
    import pickle
    model_ml = pickle.load(open(os.path.join(BASE_DIR, "model.pkl"), "rb"))
    vec_ml = pickle.load(open(os.path.join(BASE_DIR, "vectorizer.pkl"), "rb"))
    ML_MODEL_AVAILABLE = True
    print("✅ ML model loaded as backup for long articles")
except Exception as e:
    print(f"⚠️  ML model not available: {e}")
    print("   Using rule-based classifier only")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    news = request.json.get('news', '')
    word_count = len(news.split())
    
    if ML_MODEL_AVAILABLE and word_count >= 100:
        # For long articles (100+ words), ML model is reliable
        cleaned = clean(news)
        vector = vec_ml.transform([cleaned])
        pred = model_ml.predict(vector)[0]
        label = "REAL" if pred == 1 else "FAKE"
    else:
        # For headlines & short text, use rule-based classifier
        label = classify_news(news)
    
    return jsonify({'result': label})

if __name__ == '__main__':
    app.run(debug=True)
