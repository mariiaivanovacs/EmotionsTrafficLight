#!/bin/bash

# Start Backend Server for Emotion Traffic Light

cd "$(dirname "$0")"

echo "🚀 Starting Emotion Traffic Light Backend..."
echo "============================================="
echo ""

# Activate virtual environment
source venv/bin/activate

# Start Flask backend
cd backend
python app.py
