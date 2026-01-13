#!/bin/bash

# Start Frontend for Emotion Traffic Light

cd "$(dirname "$0")"

echo "🚀 Starting Emotion Traffic Light Frontend..."
echo "=============================================="
echo ""

# Go to frontend directory
cd frontend

# Start Vite dev server
npm run dev
