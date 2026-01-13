# EmotionsTrafficLight

## Project Structure

```
EmotionsTrafficLight/
├── frontend/                  # React app (basic camera view)
├── backend/                   # Backend (to be implemented)
├── emotion_traffic_light.py   # Main Python app with emotion detection
├── test_camera.py            # Camera smoke test
└── venv/                     # Python virtual environment
```

## Setup

### 1. Install Python Dependencies
```bash
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Grant Camera Permissions (macOS)
**Important:** Go to System Preferences → Security & Privacy → Camera and enable camera access for:
- Terminal (if running from Terminal)
- Your IDE (VSCode, PyCharm, etc. if running from there)

### 3. Test Camera
```bash
python test_camera.py
```

## Running the Emotion Traffic Light

```bash
python emotion_traffic_light.py
```

**Features:**
- ✅ **Real-time emotion detection** with optimized performance
- ✅ **Face detection** using OpenCV Haar Cascade
- ✅ **Emotion scoring** - displays confidence scores for each emotion
- ✅ **Lower resolution inference** (320px width) for faster processing
- ✅ **Multi-face support** - detects and tracks multiple faces simultaneously
- ✅ **Traffic light color coding:**
  - 🟢 Green: Happy, Surprise
  - 🟡 Yellow: Neutral
  - 🔴 Red: Angry, Sad, Fear, Disgust
- ✅ **Top 3 emotions** displayed per face with scores
- ✅ **Rolling average** for smooth color transitions
- ✅ **Non-blocking TTS** - emotion announcements in background thread
- ✅ **FPS counter** - monitor real-time performance

**Controls:**
- Press `q` to quit
- Press `s` to toggle text-to-speech on/off

**Display Information:**
- Each face shows dominant emotion + confidence score
- Side panel shows top 3 emotions with individual scores
- Traffic light circle in corner of each face box
- Real-time FPS and face count in top-left corner

## Running the Frontend (Alternative)

```bash
cd frontend
npm run dev
```

Then open your browser to http://localhost:5173
