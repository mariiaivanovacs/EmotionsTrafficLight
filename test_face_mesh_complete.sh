#!/bin/bash

echo "=========================================="
echo "🔍 Face Mesh Complete Diagnostic Test"
echo "=========================================="
echo ""

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Test 1: Check if backend is running
echo "1. Testing Backend Connection..."
echo "------------------------------------------"
if curl -s http://localhost:5001/api/health > /dev/null 2>&1; then
    echo -e "${GREEN}✓ Backend is running on port 5001${NC}"
    
    # Get health data
    HEALTH=$(curl -s http://localhost:5001/api/health)
    echo "  Health: $HEALTH"
else
    echo -e "${RED}✗ Backend is NOT running${NC}"
    echo "  Start it with: ./start_backend.sh"
    exit 1
fi

echo ""

# Test 2: Check face mesh availability
echo "2. Testing Face Mesh Availability..."
echo "------------------------------------------"
FACE_MESH=$(curl -s http://localhost:5001/api/face_mesh/available)
echo "  Response: $FACE_MESH"

if echo "$FACE_MESH" | grep -q '"available": true'; then
    echo -e "${GREEN}✓ Face Mesh is available${NC}"
else
    echo -e "${RED}✗ Face Mesh is NOT available${NC}"
    echo "  Run: cd backend && python download_face_landmarker_model.py"
    exit 1
fi

echo ""

# Test 3: Check cameras
echo "3. Testing Camera Detection..."
echo "------------------------------------------"
echo "  (This may take 5-10 seconds...)"
CAMERAS=$(curl -s http://localhost:5001/api/cameras)
CAMERA_COUNT=$(echo "$CAMERAS" | grep -o '"id"' | wc -l | tr -d ' ')

if [ "$CAMERA_COUNT" -gt 0 ]; then
    echo -e "${GREEN}✓ Found $CAMERA_COUNT camera(s)${NC}"
    echo "  Cameras: $CAMERAS"
else
    echo -e "${RED}✗ No cameras found${NC}"
    echo "  Check camera permissions in System Preferences"
    exit 1
fi

echo ""

# Test 4: Check frontend
echo "4. Testing Frontend..."
echo "------------------------------------------"
if curl -s http://localhost:5173 > /dev/null 2>&1; then
    echo -e "${GREEN}✓ Frontend is running on port 5173${NC}"
else
    echo -e "${YELLOW}⚠ Frontend is NOT running${NC}"
    echo "  Start it with: ./start_frontend.sh"
fi

echo ""

# Test 5: Check model file
echo "5. Testing MediaPipe Model..."
echo "------------------------------------------"
if [ -f "backend/face_landmarker.task" ]; then
    SIZE=$(ls -lh backend/face_landmarker.task | awk '{print $5}')
    echo -e "${GREEN}✓ Model file exists (${SIZE})${NC}"
else
    echo -e "${RED}✗ Model file missing${NC}"
    echo "  Run: cd backend && python download_face_landmarker_model.py"
    exit 1
fi

echo ""
echo "=========================================="
echo "📋 Summary"
echo "=========================================="
echo -e "${GREEN}✓ Backend: Running${NC}"
echo -e "${GREEN}✓ Face Mesh: Available${NC}"
echo -e "${GREEN}✓ Cameras: $CAMERA_COUNT found${NC}"
echo -e "${GREEN}✓ Model: Loaded${NC}"

if curl -s http://localhost:5173 > /dev/null 2>&1; then
    echo -e "${GREEN}✓ Frontend: Running${NC}"
else
    echo -e "${YELLOW}⚠ Frontend: Not running${NC}"
fi

echo ""
echo "=========================================="
echo "🎯 Next Steps"
echo "=========================================="
echo "1. Open browser: http://localhost:5173"
echo "2. Click 'Face Mesh Analysis (3D)' tab"
echo "3. Select camera from dropdown"
echo "4. Click 'Start Face Mesh Analysis'"
echo "5. Look at your camera - you should see:"
echo "   - 2D video feed with face detection"
echo "   - 3D face model (rotating green points)"
echo "   - Emotion label (e.g., 'Calm 😌')"
echo "   - Valence-Arousal plot"
echo "   - Geometry features (mouth, eyes, etc.)"
echo "   - Temporal analysis graphs"
echo ""
echo "=========================================="
echo "🐛 Debugging"
echo "=========================================="
echo "If analysis panel is empty:"
echo "1. Open browser console (F12)"
echo "2. Look for '📊 Face mesh update received'"
echo "3. Check if geometry_features is empty"
echo "4. Ensure your face is visible and well-lit"
echo ""
echo "For detailed debugging:"
echo "  - Open: test_api.html in browser"
echo "  - Read: DEBUGGING_GUIDE.md"
echo ""
echo "=========================================="

