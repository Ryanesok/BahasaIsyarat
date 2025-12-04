# 🤟 Sign Language Reader - Sistem Penerjemah Bahasa Isyarat Real-time

Sistem pengenalan bahasa isyarat real-time yang menggunakan Computer Vision (MediaPipe) dan Machine Learning untuk menerjemahkan gerakan tangan menjadi teks. Mendukung integrasi dengan sensor ESP32 untuk kontrol gestur tambahan.

![Python Version](https://img.shields.io/badge/python-3.12%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10%2B-orange)

---

## 📋 Daftar Isi

- [Fitur Utama](#-fitur-utama)
- [Instalasi](#-instalasi)
- [Penggunaan Cepat](#-penggunaan-cepat)
- [Struktur Project](#-struktur-project)
- [Cara Kerja Sistem](#-cara-kerja-sistem)
- [Konfigurasi](#%EF%B8%8F-konfigurasi)
- [ESP32 Integration](#-esp32-integration)
- [Troubleshooting](#-troubleshooting)
- [Dokumentasi](#-dokumentasi)
- [Kontribusi](#-kontribusi)
- [Lisensi](#-lisensi)

---

## ✨ Fitur Utama

### 🎯 Deteksi & Pengenalan
- ✅ **Real-time Hand Tracking** - MediaPipe 21-point hand landmark detection
- ✅ **Alphabet Recognition** - Pengenalan huruf A-Z dengan akurasi 95%+
- ✅ **Number Recognition** - Pengenalan angka 0-9
- ✅ **Dynamic Gesture** - Deteksi gerakan dinamis untuk kata-kata
- ✅ **Stability Check** - 15-frame stability untuk mencegah deteksi ganda
- ✅ **Cooldown System** - Jeda 2.5 detik antar deteksi untuk presisi

### 🛡️ Privacy & Security
- ✅ **Automatic Face Blurring** - Wajah otomatis diblur saat pengumpulan data
- ✅ **Local Processing** - Semua processing dilakukan secara lokal
- ✅ **No Cloud Storage** - Data training tersimpan di device

### 🎨 User Interface
- ✅ **Modern GUI** - Interface Tkinter dengan dark mode
- ✅ **Real-time Preview** - Video feed 800x600 @ 30-45 FPS
- ✅ **Word Builder** - Sistem pembentukan kata otomatis
- ✅ **Sentence Builder** - Akumulasi kalimat lengkap
- ✅ **Developer Console** - Console monitoring untuk debugging

### 🔗 Konektivitas
- ✅ **MQTT Integration** - Komunikasi dengan ESP32 via MQTT
- ✅ **Remote Control** - Kontrol via sensor accelerometer ESP32
- ✅ **Auto-reconnect** - Automatic reconnection untuk WiFi & MQTT
- ✅ **Session Logging** - Log deteksi lengkap dengan statistik

### 🚀 Performance
- ✅ **High FPS** - 30-45 FPS dengan processing minimal lag
- ✅ **Low Latency** - <50ms processing time per frame
- ✅ **Optimized ML** - RandomForest dengan 200 trees
- ✅ **3D Feature Extraction** - X, Y, Z coordinates untuk akurasi tinggi

---

## 📦 Instalasi

### Prasyarat

- **Python 3.12+** (direkomendasikan 3.12)
- **Webcam** (built-in atau eksternal)
- **Lighting yang baik** untuk akurasi optimal
- **ESP32** (opsional, untuk sensor fusion)

### Langkah Instalasi

1. **Clone Repository**
   ```bash
   git clone https://github.com/Ryanesok/BahasaIsyarat.git
   cd BahasaIsyarat
   ```

2. **Install Dependencies**
   ```powershell
   pip install -r built-in/requirements.txt
   ```

   Atau install manual:
   ```powershell
   pip install opencv-python>=4.8.0
   pip install mediapipe>=0.10.0
   pip install pillow>=10.0.0
   pip install scikit-learn>=1.3.0
   pip install numpy>=1.24.0
   pip install paho-mqtt>=1.6.0
   ```

3. **Verifikasi Instalasi**
   ```powershell
   python -c "import cv2, mediapipe, sklearn; print('✓ All dependencies installed')"
   ```

---

## 🚀 Penggunaan Cepat

### 1️⃣ Pengumpulan Data Training (WAJIB!)

```powershell
python collect_data.py
```

**Proses:**
- Pilih kategori: `static` (huruf/angka) atau `dynamic` (kata)
- Untuk static → pilih subcategory: `alphabet` atau `numbers`
- Masukkan label (contoh: A, B, C, atau 0, 1, 2)
- Tekan `Q` untuk mulai capture (150 gambar per label)
- Wajah otomatis diblur untuk privacy

**💡 Tips untuk Akurasi Terbaik:**
- ✅ Gunakan **pencahayaan terang dan merata**
- ✅ Tampilkan gesture dari **berbagai sudut** (rotasi tangan sedikit)
- ✅ Variasikan **jarak** (dekat/jauh dari kamera)
- ✅ Jaga **orientasi tangan konsisten**
- ✅ Gunakan **background polos**
- ✅ Hindari **bayangan pada tangan**

### 2️⃣ Training Model

```powershell
python train_model.py
```

**Proses:**
1. Extract features dari gambar training
2. Train RandomForestClassifier (200 trees)
3. Simpan model ke `built-in/dataset/model.p`

**Output:**
- Akurasi training: 95%+
- Model size: ~5-10 MB
- Training time: 1-3 menit (tergantung data)

### 3️⃣ Jalankan Aplikasi

```powershell
python desktop_app.py
```

**Kontrol:**
- **SPC** - Tambah kata ke kalimat (add space)
- **⌫** - Backspace (hapus karakter terakhir)
- **CLR** - Clear all (reset semua)

**Shortcut Keyboard:**
- `ESC` - Keluar aplikasi
- `D` - Toggle developer console

---

## 📁 Struktur Project

```
BahasaIsyarat/
│
├── desktop_app.py              # ⭐ Aplikasi utama (GUI + Logic)
├── collect_data.py             # 📸 Tool pengumpulan data training
├── train_model.py              # 🎓 Script training model ML
├── config.json                 # ⚙️ Konfigurasi aplikasi
│
├── built-in/                   # 📚 Library modules
│   ├── __init__.py
│   ├── alphabet_classifier.py  # 🧠 ML classifier + stability logic
│   ├── hand_tracker.py         # ✋ MediaPipe hand tracking
│   ├── dev_console.py          # 🔍 Developer console window
│   ├── session_logger.py       # 📊 Logging sistem
│   ├── autocorrect.py          # ✏️ Auto-correction untuk kata
│   ├── sensor_fusion.py        # 🔗 Sensor fusion ESP32 + camera
│   ├── visual_helpers.py       # 🎨 Helper untuk visualisasi
│   ├── dictionary.txt          # 📖 Dictionary untuk autocorrect
│   ├── requirements.txt        # 📦 Python dependencies
│   ├── README.md               # 📄 Built-in module docs
│   │
│   └── dataset/                # 💾 Trained model storage
│       ├── model.p             # 🎯 Trained RandomForest model
│       └── data.pickle         # 📊 Processed training data
│
├── sign-language-detector-python/  # 📂 Training data structure
│   └── data/
│       ├── static/             # 🔤 Static gestures (images)
│       │   ├── alphabet/       # A-Z folders
│       │   │   ├── A/
│       │   │   ├── B/
│       │   │   └── ...
│       │   └── numbers/        # 0-9 folders
│       │       ├── 0/
│       │       ├── 1/
│       │       └── ...
│       │
│       └── dynamic/            # 🎬 Dynamic gestures (videos)
│           ├── words/          # Single words
│           ├── phrases/        # Short phrases
│           └── sentences/      # Full sentences
│
├── esp32_upload/               # 🤖 ESP32 MicroPython code
│   ├── main.py                 # ESP32 gesture control
│   └── mpu6050.py              # MPU6050 sensor driver
│
├── logs/                       # 📜 Session logs & statistics
│   ├── session_YYYYMMDD/
│   │   ├── detections.csv      # Detection log
│   │   └── summary.json        # Session summary
│   └── weekly_report_*.json    # Weekly analytics
│
└── docs/                       # 📚 Documentation
    ├── QUICKSTART.md
    ├── SETUP_GUIDE.md
    └── API_REFERENCE.md
```

---

## 🔬 Cara Kerja Sistem

### 1. Hand Tracking Pipeline

```
Camera Input (30-45 FPS)
    ↓
MediaPipe Hand Detection
    ↓ (21 landmarks × 3D coordinates)
Feature Extraction
    ↓ (63 features: x, y, z for each landmark)
Normalization & Scaling
    ↓
RandomForest Classifier
    ↓
Letter Recognition
    ↓
Stability Check (15 frames)
    ↓
Cooldown (2.5s)
    ↓
Add to Word Buffer
```

### 2. State Machine

```
┌─────────┐
│  IDLE   │ ◄──────────────┐
└────┬────┘                │
     │ Hand Detected       │
     ↓                     │
┌──────────┐               │
│LISTENING │               │
└────┬─────┘               │
     │ Stable Detection    │
     ↓                     │
┌──────────┐               │
│COOLDOWN  │───────────────┘
└──────────┘ 2.5s timeout
```

### 3. Sensor Fusion (ESP32 + Camera)

```
┌─────────────┐        ┌──────────────┐
│   Camera    │        │    ESP32     │
│  (Letters)  │        │(Acc + Gyro)  │
└──────┬──────┘        └──────┬───────┘
       │                      │
       │   MQTT Broker        │
       └──────────┬───────────┘
                  ↓
          Decision Engine
          (Weighted Fusion)
                  ↓
            Final Gesture
```

---

## ⚙️ Konfigurasi

### config.json - Pengaturan Utama

```json
{
  "mqtt": {
    "broker": "broker.hivemq.com",
    "port": 1883,
    "topic_gesture": "pameran/gerakan",
    "topic_visual": "pameran/visual",
    "client_id": "kelompok-4-desktop",
    "reconnect_interval": 3
  },
  "camera": {
    "device_id": 0,           // 0 = default, 1 = eksternal
    "width": 1280,
    "height": 720,
    "fps_limit": 45
  },
  "detection": {
    "min_detection_confidence": 0.5,
    "min_tracking_confidence": 0.5,
    "max_num_hands": 1,
    "cooldown_seconds": 2.5,   // Jeda antar deteksi
    "min_stable_frames": 3     // Frame stabil sebelum accept
  },
  "model": {
    "path": "built-in/dataset/model.p",
    "data_path": "sign-language-detector-python/data"
  },
  "logging": {
    "enabled": true,
    "log_dir": "logs",
    "save_sessions": true,
    "csv_output": true
  },
  "ui": {
    "show_roi_guide": true,
    "show_landmarks": true,
    "dark_mode": true
  },
  "autocorrect": {
    "enabled": true,
    "dictionary_path": "built-in/dictionary.txt",
    "max_suggestions": 3
  }
}
```

### Fine-tuning Parameter

**Untuk Akurasi Lebih Tinggi:**
```python
# built-in/alphabet_classifier.py
self.stability_frames = 15      # Naikkan ke 20 (lebih stabil)
self.cooldown_period = 2.5      # Naikkan ke 3.0 (lebih lambat)
```

**Untuk Responsiveness Lebih Cepat:**
```python
self.stability_frames = 10      # Turunkan ke 10 (lebih cepat)
self.cooldown_period = 1.5      # Turunkan ke 1.5 (lebih responsif)
```

**Untuk Tracking Lebih Smooth:**
```python
# built-in/hand_tracker.py
min_detection_confidence=0.7    # Default: 0.7 (naikkan = lebih stabil)
min_tracking_confidence=0.7     # Default: 0.7 (naikkan = less flicker)
```

---

## 🤖 ESP32 Integration

### Hardware Setup

**Komponen:**
- ESP32 DevKit V1
- MPU6050 Accelerometer + Gyroscope
- Kabel jumper

**Koneksi:**
```
ESP32          MPU6050
─────────────  ────────
GPIO 21 (SDA)  →  SDA
GPIO 22 (SCL)  →  SCL
3.3V           →  VCC
GND            →  GND
```

### Upload Kode ke ESP32

1. **Install Thonny IDE** atau **esptool**

2. **Upload MicroPython Firmware** (jika belum)
   ```bash
   esptool.py --port COM3 erase_flash
   esptool.py --port COM3 write_flash -z 0x1000 esp32-micropython.bin
   ```

3. **Upload Files**
   - Upload `esp32_upload/mpu6050.py`
   - Upload `esp32_upload/main.py`

4. **Konfigurasi WiFi**
   Edit `main.py`:
   ```python
   SSID = "NamaWiFiAnda"
   PASSWORD = "PasswordWiFi"
   ```

5. **Reset ESP32** - kode akan auto-run

### Gesture Control via ESP32

| Gerakan ESP32      | Aksi Desktop    | Threshold |
|--------------------|-----------------|-----------|
| Miring **KANAN**   | SPACE (spasi)   | AcX > 7000|
| Miring **KIRI**    | BACKSPACE       | AcX < -7000|
| Miring **DEPAN**   | CLEAR           | AcY > 7000|

**Kalibrasi Threshold:**
```python
# esp32_upload/main.py
BATAS_MIRING = 7000  # Naikkan jika terlalu sensitif
                     # Turunkan jika kurang responsif
```

---

## 🐛 Troubleshooting

### ❌ Kamera Tidak Terbuka

**Solusi:**
1. Cek kamera tidak digunakan aplikasi lain
2. Ganti `device_id` di `config.json`:
   ```json
   "camera": { "device_id": 1 }
   ```
3. Test kamera:
   ```python
   import cv2
   cap = cv2.VideoCapture(0)
   print(cap.isOpened())
   ```

### ❌ Model Not Found

**Penyebab:** Belum training model

**Solusi:**
```powershell
python collect_data.py   # Kumpulkan data dulu
python train_model.py    # Kemudian train
```

### ❌ Akurasi Rendah (<80%)

**Solusi:**
1. Kumpulkan lebih banyak data:
   ```python
   # collect_data.py
   dataset_size = 200  # Naikkan dari 150
   ```
2. Improve lighting (gunakan lampu ring)
3. Pastikan gesture konsisten
4. Check MediaPipe confidence:
   ```json
   "min_detection_confidence": 0.7
   ```

### ❌ ESP32 Tidak Konek MQTT

**Solusi:**
1. Cek WiFi credentials di `main.py`
2. Test broker:
   ```bash
   ping broker.hivemq.com
   ```
3. Cek firewall tidak block port 1883
4. Monitor serial output:
   ```
   Thonny → View → Plotter
   ```

### ❌ Detection Flickering

**Penyebab:** Stability frames terlalu rendah

**Solusi:**
```python
# built-in/alphabet_classifier.py
self.stability_frames = 20  # Naikkan dari 15
```

### ❌ C++ Warnings Muncul

**Info:** Ini normal dari MediaPipe/TensorFlow, tidak mempengaruhi fungsi.

**Suppress (optional):**
```powershell
python desktop_app.py 2>$null    # PowerShell
python desktop_app.py 2>nul      # CMD
```

---

## 📚 Dokumentasi

### File Dokumentasi

- **[built-in/README.md](built-in/README.md)** - Detail module built-in
- **[docs/QUICKSTART.md](docs/)** - Tutorial step-by-step
- **[docs/API_REFERENCE.md](docs/)** - API reference lengkap

### Developer Console

Tekan `D` saat aplikasi running untuk membuka console yang menampilkan:
- Frame count & detection rate
- FPS real-time
- Hand landmarks (21 points)
- Gesture confidence
- MQTT status
- Error logs

### Session Logging

Semua deteksi disimpan di `logs/session_YYYYMMDD/`:
- `detections.csv` - Log setiap deteksi
- `summary.json` - Statistik session

**Contoh CSV:**
```csv
timestamp,frame,letter,confidence,word,sentence,x,y,processing_time
2025-12-04 10:30:45,150,A,0.95,HELLO,,320,240,45
```

---

## 🤝 Kontribusi

Kontribusi sangat diterima! Cara berkontribusi:

1. **Fork** repository ini
2. **Buat branch** feature (`git checkout -b feature/AmazingFeature`)
3. **Commit** changes (`git commit -m 'Add some AmazingFeature'`)
4. **Push** ke branch (`git push origin feature/AmazingFeature`)
5. **Open Pull Request**

### Roadmap

- [ ] Support untuk lebih banyak gesture dinamis
- [ ] Model deep learning (CNN/LSTM)
- [ ] Mobile app (Flutter/React Native)
- [ ] Cloud sync untuk dataset
- [ ] Multi-language support (ASL, BSL, dll)
- [ ] Voice output (text-to-speech)

---

## 📄 Lisensi

Project ini dilisensikan di bawah **MIT License** - lihat file [LICENSE](LICENSE) untuk detail.

```
MIT License

Copyright (c) 2025 Ryanesok

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction...
```

---

## 🙏 Credits & Acknowledgments

### Libraries & Frameworks
- **[MediaPipe](https://mediapipe.dev/)** - Google's hand tracking solution
- **[OpenCV](https://opencv.org/)** - Computer vision library
- **[scikit-learn](https://scikit-learn.org/)** - Machine learning framework
- **[Paho MQTT](https://www.eclipse.org/paho/)** - MQTT client library

### Inspiration
- Training approach inspired by [computervisioneng/sign-language-detector-python](https://github.com/computervisioneng/sign-language-detector-python)
- Dataset structure based on ASL (American Sign Language) standards

### Team
- **Kelompok 4** - Development Team
- **Ryanesok** - Project Lead & Main Developer

---

## 📞 Kontak & Support

- **GitHub Issues:** [Report Bug](https://github.com/Ryanesok/BahasaIsyarat/issues)
- **Email:** [Contact via GitHub]
- **Repository:** [github.com/Ryanesok/BahasaIsyarat](https://github.com/Ryanesok/BahasaIsyarat)

---

## 🌟 Star History

Jika project ini bermanfaat, jangan lupa beri ⭐ di GitHub!

---

## 📊 Statistics

![GitHub stars](https://img.shields.io/github/stars/Ryanesok/BahasaIsyarat?style=social)
![GitHub forks](https://img.shields.io/github/forks/Ryanesok/BahasaIsyarat?style=social)
![GitHub issues](https://img.shields.io/github/issues/Ryanesok/BahasaIsyarat)

---

**Made with ❤️ for sign language accessibility**

*Membantu komunikasi tanpa batas melalui teknologi* 🤟
