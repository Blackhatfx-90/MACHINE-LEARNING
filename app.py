from flask import Flask, render_template, request, jsonify
import pickle, re, string, os

# Use absolute paths for Vercel serverless compatibility
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
app = Flask(__name__, template_folder=os.path.join(BASE_DIR, "templates"))

model = pickle.load(open(os.path.join(BASE_DIR, "model.pkl"), "rb"))
vec   = pickle.load(open(os.path.join(BASE_DIR, "vectorizer.pkl"), "rb"))

def clean(text):
    text = text.lower()
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[%s]' % re.escape(string.punctuation), ' ', text)
    text = re.sub(r'\w*\d\w*', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    news = request.json.get('news','')
    cleaned = clean(news)
    vector = vec.transform([cleaned])
    pred = model.predict(vector)[0]
    prob = model.predict_proba(vector)[0]
    confidence = round(max(prob) * 100, 2)
    label = "REAL" if pred == 1 else "FAKE"
    return jsonify({'result': label, 'confidence': confidence})

if __name__ == '__main__':
    app.run(debug=True)
