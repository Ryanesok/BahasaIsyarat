# 🤟 Sign Language Reader

Real-time sign language alphabet recognition system with custom training.

---

## 🚀 Quick Start

### 1. Collect Training Data (IMPORTANT!)
```powershell
python collect_data.py
```
**Collects 150 images per letter** for optimal accuracy.  
**🔒 Privacy Protected**: Faces automatically blurred in saved images.

**Tips for BEST accuracy:**
- ✅ Use **BRIGHT, EVEN LIGHTING**
- ✅ Show gesture from **MULTIPLE ANGLES** (rotate hand slightly)
- ✅ Vary **DISTANCE** (move closer/farther)
- ✅ Keep **CONSISTENT** hand orientation
- ✅ Use **PLAIN BACKGROUND**

### 2. Train Model
```powershell
python train_model.py
```
Trains optimized RandomForest (200 trees) on your data. Expect **95%+ accuracy**.

### 3. Run Application
```powershell
python desktop_app.py
```
Real-time recognition with:
- **15-frame stability** (prevents flickering)
- **2.5s cooldown** (prevents duplicate letters)
- **Enhanced 3D features** (X, Y, Z coordinates)

---

## 📦 Installation

```powershell
pip install -r requirements.txt
```

**Requirements:**
- Python 3.12+
- Webcam
- Good lighting

---

## 🎮 Application Features

- ✅ **Privacy Protected** - Automatic face blurring in training data
- ✅ **Enhanced accuracy** - 3D feature extraction (X, Y, Z)
- ✅ **Stable recognition** - 15-frame stability check
- ✅ **Reduced flickering** - High confidence tracking (0.7)
- ✅ **Optimized ML** - RandomForest with 200 trees
- ✅ **Fixed video frame** - 800x600 resolution (no expanding)
- ✅ **Word & sentence builder**
- ✅ **MQTT remote monitoring**
- ✅ **Developer console** with real-time metrics
- ✅ **30-45 FPS** performance

### Controls
- **SPC** - Add word to sentence
- **⌫** - Backspace
- **CLR** - Clear all

---

## 📁 Project Structure

```
BahasaIsyarat/
├── desktop_app.py              # Main application
├── collect_data.py             # Data collection tool  
├── train_model.py              # Model training
├── run_desktop_app.bat         # Launcher with no warnings
├── run_collect_data.bat        # Data collection launcher
├── run_train_model.bat         # Training launcher
│
├── built-in/                   # Library modules
│   ├── __init__.py
│   ├── alphabet_classifier.py  # ML classifier
│   ├── hand_tracker.py         # MediaPipe tracking
│   └── dev_console.py          # Developer metrics
│
├── model.p                     # Trained model (after training)
├── data.pickle                 # Processed training data
│
├── sign-language-detector-python/
│   └── data/                   # Training images
│       ├── A/
│       ├── B/
│       └── ...
│
└── docs/                       # Documentation
    ├── WARNINGS_EXPLAINED.md
    ├── FIXES_LOG.md
    └── DATA_COLLECTION_GUIDE.md
```

---

## 🔇 Console Output

**Current Status**: Some C++ warnings appear (harmless, informational only)

```powershell
python desktop_app.py      # Application with some C++ warnings
python collect_data.py     # Data collection with some C++ warnings  
python train_model.py      # Training with some C++ warnings
```

**For completely clean output** (no warnings):
```powershell
python desktop_app.py 2>$null     # PowerShell: redirect stderr
python desktop_app.py 2>nul       # CMD: redirect stderr
```

**Note**: The C++ warnings from MediaPipe/TensorFlow are printed during library initialization and cannot be suppressed at the Python level. They are informational only and do not affect functionality. See `docs/WARNING_SUPPRESSION_STATUS.md` for details.

---

## ⚙️ Configuration

**For Better Accuracy (built-in/alphabet_classifier.py):**
```python
self.stability_frames = 15         # Current: 15 frames (~0.5s)
self.cooldown_period = 2.5         # Current: 2.5 seconds between letters
```

**For More Training Data (collect_data.py):**
```python
dataset_size = 150                 # Current: 150 images per letter
                                  # Increase to 200 for even better accuracy
```

**For Smoother Tracking (hand_tracker.py):**
```python
min_detection_confidence=0.7       # Higher = more stable (current: 0.7)
min_tracking_confidence=0.7        # Higher = less flickering (current: 0.7)
```

---

## 🐛 Troubleshooting

**No model found?**
```powershell
python collect_data.py  # Collect data first
python train_model.py   # Then train
```

**Camera not opening?**
- Close other apps using camera
- Change camera index in `collect_data.py` (try 0, 1, 2)

**Low accuracy?**
- Collect more images (increase dataset_size)
- Use consistent gestures
- Improve lighting

---

## 📚 Full Documentation

See `docs/` folder for detailed guides:
- **QUICKSTART.md** - Step-by-step tutorial
- **DATASET_MIGRATION.md** - Technical details
- **SETUP_GUIDE.md** - Installation guide

---

## 🙏 Credits

- Training approach: [computervisioneng/sign-language-detector-python](https://github.com/computervisioneng/sign-language-detector-python)
- Hand tracking: Google MediaPipe
- ML: scikit-learn RandomForest

---

**MIT License** | Made with ❤️ for sign language accessibility


