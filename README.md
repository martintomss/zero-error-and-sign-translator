<div align="center">

# ⚡ Zero-Error Real-Time Hand Sign Translator
### Penerjemah Isyarat Tangan Real-Time dengan Stabilisasi Spasial Anti-Gravity & Protokol Zero-Error

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.8+-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)](https://opencv.org/)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-Hands_21_Landmarks-00A98F?style=for-the-badge&logo=google&logoColor=white)](https://developers.google.com/mediapipe)
[![Flask](https://img.shields.io/badge/Flask-Web_Server-000000?style=for-the-badge&logo=flask&logoColor=white)](https://flask.palletsprojects.com/)
[![Netlify](https://img.shields.io/badge/Netlify-Ready-00C7B7?style=for-the-badge&logo=netlify&logoColor=white)](https://www.netlify.com/)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)

<br/>

**Sistem penerjemah isyarat tangan real-time otonom** yang menggabungkan visi komputer mutakhir (*21-Joint MediaPipe 3D Tracking*), **Stabilisasi Spasial Anti-Gravity**, **Kalman Filter 6-State** untuk meredam jitter koordinat, serta **Strict Zero-Error Protocol (>88% Gate)** yang menjamin sistem tidak akan pernah menebak jika tingkat kepercayaan rendah.

</div>

---

## 🌟 Fitur Utama & Keunggulan

### 1. 🛸 Stabilisasi Spasial Anti-Gravity (`stabilizer.py`)
- **Origin Anchoring (Wrist $L_0$)**: Menjadikan titik pergelangan tangan sebagai titik pusat $(0, 0, 0)$, sehingga koordinat 20 sendi lainnya sepenuhnya bebas dari variasi posisi tangan di depan kamera.
- **Normalisasi Skala Kanonikal**: Mengukur jarak Euclidean telapak ($L_0$ ke $L_9$) dan memetakan sendi ke skala unit 3D, menghilangkan variasi akibat jarak tangan dekat/jauh dari lensa.
- **Kompensasi Gravitasi (Rotational Invariance)**: Menyelaraskan orientasi vertikal tangan, mengatasi kemiringan tangan akibat gravitasi atau sudut webcam.
- **Kalman Filter (6-State Tracking)**: Melacak status $[x, y, z, v_x, v_y, v_z]$ untuk seluruh 21 sendi secara real-time guna meredam noise kamera dan flicker pencahayaan.

### 2. 🛡️ Strict Zero-Error Protocol (>88% Confidence Gate) (`model.py`)
- Model AI konvensional seringkali memaksakan tebakan pada gestur ambigu.
- Sistem ini menerapkan **Strict Gating**:
  $$\max P(c \mid X) > 0.88 \implies \text{Hasil Terverifikasi (Verified)}$$
  $$\max P(c \mid X) \le 0.88 \implies \text{"Signal Unclear / Analyzing"} \text{ (Tanpa Tebakan Palsu)}$$
- Dilengkapi feedback suara dinamis (*Web Audio Synthesizer Chime & Speech Synthesis* bahasa Indonesia).

### 3. 🎯 Mode Latihan Interaktif & AI Coach
- **Checklist 5 Jari Real-Time**: Memantau apakah Jempol, Telunjuk, Tengah, Manis, dan Kelingking dalam posisi *Lurus* atau *Ditekuk*.
- **AI Coach Tips**: Memberikan saran instan untuk menyempurnakan bentuk tangan.
- **Tantangan Tahan Posisi (Hold-to-Succeed)**: Pengguna harus mempertahankan kesesuaian gestur $\ge 92\%$ selama 1.4 detik untuk membuka gestur berikutnya.

### 4. 🌐 Arsitektur Hibrida (Dual-Mode Deployment)
- **Mode 1: Python Flask Server (`app.py`)**: OpenCV DirectShow backend berkecepatan tinggi, MJPEG ultra-low-latency streaming.
- **Mode 2: 100% Client-Side Web (`index.html` & Netlify)**: Menggunakan MediaPipe JS CDN langsung di browser tanpa memerlukan server back-end (siap hosting gratis di Netlify/Vercel/GitHub Pages).

---

## 🏗️ Diagram Arsitektur Multi-Agent

```text
                       ┌─────────────────────────┐
                       │   Live Webcam Stream    │
                       └────────────┬────────────┘
                                    │
                                    ▼
        ┌────────────────────────────────────────────────────────┐
        │   Agent 1: Computer Vision & Spatial Stabilization     │
        │   (stabilizer.py & camera.py)                          │
        │   - 21-Joint MediaPipe Hands Extraction                │
        │   - 6-State 3D Kalman Filter Jitter Reduction          │
        │   - Anti-Gravity Canonical Normalization (Wrist L0)    │
        │   - Virtual Floating 3D Canvas Reticle Lock            │
        └────────────────────────────┬───────────────────────────┘
                                    │ Canonical Features (63D)
                                    ▼
        ┌────────────────────────────────────────────────────────┐
        │   Agent 2: Deep Learning & Zero-Error Protocol         │
        │   (model.py / classifier.js)                           │
        │   - Spatial-Temporal Neural Sequence Classifier        │
        │   - Softmax Posterior Probability Distribution         │
        │   - Strict >88% Gate:                                  │
        │       If Confidence > 0.88  ==> Verified Sign          │
        │       If Confidence <= 0.88 ==> "Signal Unclear"       │
        └────────────────────────────┬───────────────────────────┘
                                    │ Telemetry & Stream
                                    ▼
        ┌────────────────────────────────────────────────────────┐
        │   Agent 3: UI/UX & Web Integration                     │
        │   (app.py, templates/index.html, static/)              │
        │   - Cyber-Glassmorphism UI Overlay                     │
        │   - Real-time 88% Gate Meter & Dynamic Reticle         │
        │   - Web Audio Synth Chimes & Speech Audio              │
        └────────────────────────────────────────────────────────┘
```

---

## 📋 18 Isyarat Tangan yang Didukung

| No | Ikon | ID Gestur | Label | Kategori | Deskripsi / Instruksi |
|:---:|:---:|:---|:---|:---|:---|
| 1 | 👋 | `hello` | Halo / Hello | Salam | Buka kelima jari tegak lurus ke atas |
| 2 | 🙏 | `thank_you` | Terima Kasih | Kesopanan | Rapatkan kedua telapak tangan menghadap dada |
| 3 | 🤟 | `i_love_you` | I Love You | Emosi | Buka jempol ke samping, telunjuk & kelingking lurus ke atas |
| 4 | 👍 | `yes` | Setuju / Yes | Konfirmasi | Acungkan ibu jari ke atas, kepalkan keempat jari lain |
| 5 | 👎 | `no` | Tidak / No | Konfirmasi | Arahkan ibu jari ke bawah, kepalkan keempat jari lain |
| 6 | ✌️ | `peace` | Damai / Peace | Simbol | Angkat telunjuk dan jari tengah membentuk huruf V |
| 7 | 👌 | `ok` | Sempurna / OK | Konfirmasi | Sentuhkan ujung jempol dan telunjuk membentuk lingkaran |
| 8 | 🤘 | `rock_on` | Rock On | Simbol | Angkat telunjuk dan kelingking, lipat jari lainnya |
| 9 | 🤙 | `call_me` | Hubungi Saya | Komunikasi | Buka jempol dan kelingking menjauh, lipat 3 jari tengah |
| 10 | 🤲 | `help` | Bantuan / Tolong | Kebutuhan | Buka kedua telapak tangan menghadap ke atas |
| 11 | ☝️ | `num_1` | Angka 1 | Numerik | Angkat jari telunjuk lurus ke atas |
| 12 | ✌️ | `num_2` | Angka 2 | Numerik | Angkat telunjuk dan jari tengah lurus ke atas |
| 13 | 🤟 | `num_3` | Angka 3 | Numerik | Angkat jempol, telunjuk, dan jari tengah |
| 14 | 🖖 | `num_4` | Angka 4 | Numerik | Angkat 4 jari lurus, lipat ibu jari ke telapak |
| 15 | 🖐️ | `num_5` | Angka 5 | Numerik | Buka seluruh lima jari tangan |
| 16 | ✊ | `fist_a` | Huruf A / Kepalan | Alfabet | Kepalkan seluruh jari tangan dengan jempol di samping |
| 17 | ✋ | `flat_b` | Huruf B / Telapak Rapat | Alfabet | Angkat 4 jari rapat lurus, lipat ibu jari ke depan |
| 18 | 🤏 | `cup_c` | Huruf C / Setengah Lingkar | Alfabet | Lengkungkan seluruh jari membentuk setengah lingkaran |

---

## 📁 Struktur Direktori

```
zero_error_translator/
├── app.py                  # Server Flask & endpoint API Telemetry/Stream
├── camera.py               # Pipeline VideoCapture OpenCV & MediaPipe Hands
├── model.py                # Classifier AI Python & Strict >88% Gate Protocol
├── stabilizer.py           # Algoritma Anti-Gravity & 6-State 3D Kalman Filter
├── test_system.py          # Automated Test Suite (5 Test Cases)
├── requirements.txt        # Daftar dependency Python
├── run.bat                 # Script 1-klik untuk menjalankan server di Windows
├── netlify.toml            # Konfigurasi deployment hosting Netlify
├── index.html              # Standalone web app (100% In-Browser MediaPipe)
├── templates/
│   └── index.html          # Template HTML Flask Dashboard
└── static/
    ├── css/
    │   └── style.css       # Tema Cyber-Glassmorphism, animasi & HUD
    └── js/
        ├── classifier.js   # In-Browser Stabilizer & Classifier (JS Port)
        └── main.js         # Logika UI, Web Audio synth, dan event handling
```

---

## 🚀 Panduan Memulai (Quick Start)

### Persyaratan Sistem
- Python 3.10 atau 3.11+
- Webcam (kamera laptop atau USB webcam)
- Browser modern (Google Chrome, Microsoft Edge, atau Firefox)

### Opsi A: Menjalankan dengan `run.bat` (Termudah di Windows)
Cukup **klik dua kali** file:
```cmd
run.bat
```
Script ini akan otomatis mendeteksi lingkungan Python, menjalankan server Flask, dan membuka browser di `http://localhost:5000`.

### Opsi B: Menjalankan via Terminal / CMD / PowerShell
1. **Clone repository ini:**
   ```bash
   git clone https://github.com/martintomss/zero-error-and-sign-translator.git
   cd zero-error-and-sign-translator
   ```

2. **Install dependency:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Jalankan aplikasi:**
   ```bash
   python app.py
   ```

4. **Buka di browser:**
   ```
   http://localhost:5000
   ```

---

## 🧪 Pengujian Otomatis (Automated Tests)

Jalankan test suite autonomous untuk memvalidasi algoritma Kalman filter, stabilisasi spasial, dan protokol gating:

```bash
python test_system.py
```

### Hasil Validasi:
```text
--- TEST 1: Kalman Filter Jitter Damping ---
>>> PASS: Kalman Filter significantly damped coordinate jittering!

--- TEST 2: Anti-Gravity Spatial Stabilization ---
>>> PASS: Anti-Gravity Spatial Normalization locks hand into invariant 3D space!

--- TEST 3: Strict Zero-Error Protocol (>88% Threshold) ---
>>> PASS: Clear gesture exceeds 88% confidence!

--- TEST 4: Zero Guessing on Ambiguous Sign ---
>>> PASS: Ambiguous gesture safely rejected as "Signal Unclear / Analyzing"!

--- TEST 5: Pipeline Frame Processing ---
>>> PASS: Pipeline initialized and processed frames cleanly!
==========================================
>>> [SUCCESS] ALL TESTS PASSED!
==========================================
```

---

## 🌐 Deploy ke Netlify (Gratis Tanpa Server)

Aplikasi ini sudah dilengkapi file konfigurasi `netlify.toml` dan engine JavaScript MediaPipe mandiri di `index.html`.

1. Hubungkan repository GitHub ini ke akun [Netlify](https://www.netlify.com/).
2. Konfigurasi build:
   - **Publish directory**: `.` (root)
3. Deploy! Aplikasi akan langsung aktif menggunakan webcam browser klien.

---

## 📄 Lisensi

Proyek ini dirilis di bawah lisensi [MIT](LICENSE). Bebas digunakan untuk keperluan edukasi, riset, maupun pengembangan komersial.

<div align="center">
  <sub>Dibangun dengan ❤️ oleh <b>Martin Tom Samuel Simorangkir</b></sub>
</div>
