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

# Step 2: Train the model
echo "[2/4] Training model (downloads dataset + trains)..."
python3 train.py
echo "✅ Model trained"
echo ""

# Step 3: Verify files
echo "[3/4] Verifying files..."
if [ -f "model.pkl" ] && [ -f "vectorizer.pkl" ]; then
    echo "✅ model.pkl exists"
    echo "✅ vectorizer.pkl exists"
else
    echo "❌ ERROR: Model files not found!"
    exit 1
fi
echo ""

# Step 4: Test run
echo "[4/4] Starting Flask app..."
echo "✅ Open http://localhost:5000 in your browser"
echo ""
python3 app.py
