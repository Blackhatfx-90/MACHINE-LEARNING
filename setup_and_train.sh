#!/bin/bash
# ═══════════════════════════════════════════════════════
# Fake News Detection System — One-Click Setup & Train
# Run: bash setup_and_train.sh
# ═══════════════════════════════════════════════════════
set -e

echo "══════════════════════════════════════════════"
echo "  Fake News Detection System — Setup Script"
echo "══════════════════════════════════════════════"
echo ""

# Step 1: Install dependencies
echo "[1/4] Installing Python dependencies..."
pip3 install flask pandas scikit-learn nltk requests gunicorn --user
echo "✅ Dependencies installed"
echo ""

# Step 2: Train the model (dual model: headline + full article)
echo "[2/4] Training model (downloads dataset + trains)..."
python3 train.py
echo "✅ Model trained"
echo ""

# Step 3: Verify files
echo "[3/4] Verifying files..."
FILES_OK=true
for f in model.pkl vectorizer.pkl model_headline.pkl vectorizer_headline.pkl model_full.pkl vectorizer_full.pkl; do
    if [ -f "$f" ]; then
        echo "✅ $f exists"
    else
        echo "⚠️  $f not found (will use fallback)"
    fi
done
echo ""

# Step 4: Test run
echo "[4/4] Starting Flask app..."
echo "✅ Open http://localhost:5000 in your browser"
echo ""
python3 app.py
