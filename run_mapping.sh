#!/bin/bash

# Script to generate MediaPipe to FLAME mapping
# Usage: ./run_mapping.sh

echo "=========================================="
echo "MediaPipe to FLAME Mapping Generator"
echo "=========================================="
echo ""

# Check if virtual environment is activated
if [[ -z "$VIRTUAL_ENV" ]]; then
    echo "⚠️  Virtual environment not activated!"
    echo "Activating venv..."
    source venv/bin/activate
fi

# Check if mp_neutral_frame.npy exists
MP_FILE="backend/mp_neutral_frame.npy"
if [ ! -f "$MP_FILE" ]; then
    echo "❌ Error: $MP_FILE not found!"
    echo ""
    echo "Please capture a neutral face first:"
    echo "  1. Start the FLAME reconstruction panel"
    echo "  2. Look at the camera with a neutral expression"
    echo "  3. Click 'Save Face' button"
    echo ""
    exit 1
fi

echo "✓ Found MediaPipe landmarks file: $MP_FILE"
echo ""

# Run the mapping script
echo "Running mapping script..."
echo ""

python map_mediapipe_to_flame.py \
    --mp "$MP_FILE" \
    --out "backend/mapping.json" \
    --img-w 640 \
    --img-h 480

echo ""
echo "=========================================="
echo "Done!"
echo "=========================================="

