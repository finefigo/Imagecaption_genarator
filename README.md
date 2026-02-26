# 🎨 AI-Powered Image Caption Generator with Mood Customization

An intelligent Streamlit web application that generates image captions using the **BLIP** (Bootstrapped Language-Image Pretraining) model and transforms them into different emotional tones.

![Python](https://img.shields.io/badge/Python-3.9+-blue?logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red?logo=streamlit)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange?logo=pytorch)
![HuggingFace](https://img.shields.io/badge/🤗_Transformers-4.30+-yellow)

---

## ✨ Features

- **AI Image Captioning** — Generate context-aware captions using BLIP Vision Transformer
- **8 Mood Styles** — Normal, Happy, Sad, Funny, Romantic, Dramatic, Inspirational, Poetic
- **Creativity Control** — Adjustable temperature slider (0.1–1.5)
- **Beam Search** — Configurable beam width for higher quality captions
- **Download Captions** — Export captions as `.txt` files
- **CPU Optimized** — Runs on CPU with GPU auto-detection
- **Modern UI** — Dark glassmorphic theme with animations

---

## 🏗️ Architecture

```
Image Upload → Image Preprocessing → Vision Encoder (ViT)
    → Cross-Attention → Transformer Decoder → Base Caption
    → Mood Prompt Conditioning → Temperature Control
    → Mood-Enhanced Caption
```

| Component | Technology |
|---|---|
| Vision Encoder | Vision Transformer (ViT-Large, 576 patches) |
| Text Decoder | Transformer with Cross-Attention |
| Mood Engine | Prompt-based Conditional Generation |
| Frontend | Streamlit with custom CSS |
| Model | `Salesforce/blip-image-captioning-large` |

---

## 🚀 Quick Start

### 1. Clone & Navigate
```bash
cd Imagecaption_genarator(bct)
```

### 2. Create Virtual Environment
```bash
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the App
```bash
streamlit run streamlit_app.py
```

The app opens at `http://localhost:8501`. The BLIP model (~1.5GB) downloads automatically on first run.

---

## 📁 Project Structure

```
├── streamlit_app.py          # Main Streamlit application
├── requirements.txt          # Python dependencies
├── README.md                 # This file
├── CUSTOM_MODEL_GUIDE.md     # Guide to train your own model
├── models/
│   ├── __init__.py
│   ├── caption_engine.py     # BLIP model loading & generation
│   └── mood_transformer.py   # Mood transformation engine
└── .streamlit/
    └── config.toml           # Theme configuration
```

---

## 🎭 Supported Moods

| Mood | Emoji | Style |
|---|---|---|
| Normal | 📝 | Factual, objective description |
| Happy | 😊 | Joyful, uplifting, bright tone |
| Sad | 😢 | Melancholic, somber, reflective |
| Funny | 😂 | Humorous, witty, playful |
| Romantic | 💕 | Tender, dreamy, intimate |
| Dramatic | 🎭 | Intense, cinematic, powerful |
| Inspirational | ✨ | Uplifting, motivational |
| Poetic | 🌸 | Lyrical, metaphorical, artistic |

---

## 🧠 Key AI Concepts (see code comments for details)

- **Vision Transformers (ViT)** — Patch-based image encoding with self-attention
- **Cross-Attention** — Bridges image embeddings with text generation
- **Autoregressive Decoding** — Word-by-word text generation
- **Beam Search** — Explores multiple caption candidates for better quality
- **Temperature** — Controls creativity vs. accuracy tradeoff
- **Prompt Conditioning** — Steers generation toward specific emotional tones

---

## 🔮 Future Extensions

- 🌍 Multilingual captions
- 🔊 Text-to-speech output
- 📷 Webcam mode
- 📊 Batch processing
- 🤖 Custom model training (see `CUSTOM_MODEL_GUIDE.md`)

---

## 📝 License

This project is for educational purposes. BLIP model is by [Salesforce Research](https://github.com/salesforce/BLIP).
