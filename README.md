# 🎨 AI-Powered Image Caption Generator with Mood Customization

An intelligent Streamlit web application that generates creative image captions using **Google's Gemini 2.5 Flash** model and transforms them into different emotional tones through a two-stage AI pipeline.

![Python](https://img.shields.io/badge/Python-3.9+-blue?logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red?logo=streamlit)
![Gemini](https://img.shields.io/badge/Gemini_2.5_Flash-Google_AI-4285F4?logo=google)

---

## ✨ Features

- **Gemini 2.5 Flash AI** — Generates context-aware, creative captions using Google's latest multimodal model
- **Two-Stage Pipeline** — Stage A (vision analysis) + Stage B (mood rewrite) for high-quality results
- **8 Mood Styles** — Normal, Happy, Sad, Funny, Romantic, Dramatic, Inspirational, Poetic
- **Creativity Control** — Adjustable temperature slider (0.1–1.5) with mood-specific multipliers
- **Download Captions** — Export captions as `.txt` files
- **Cloud-Powered** — No local GPU required; all inference runs on Google's cloud
- **Modern UI** — Premium dark glassmorphic theme with animations

---

## 🏗️ Architecture

```
Image Upload → Gemini 2.5 Flash (Vision + Language)
    → Stage A: Vision-to-Suggestion Pass → Base Caption
    → Stage B: Text-Only Mood Rewrite → Mood-Enhanced Caption
```

### Two-Stage Pipeline

| Stage | Description | Input | Output |
|---|---|---|---|
| **Stage A** | Vision-to-Suggestion Pass | Image + prompt | Creative base caption |
| **Stage B** | Mood Transformation Pass | Base caption + mood instruction (text-only) | Mood-enhanced caption |

### Tech Stack

| Component | Technology |
|---|---|
| AI Model | Google Gemini 2.5 Flash |
| API Client | `google-genai` SDK |
| Mood Engine | Instruction-based zero-shot rewriting |
| Frontend | Streamlit with custom CSS |
| Image Processing | Pillow (PIL) |

---

## 🚀 Quick Start

### 1. Clone & Navigate
```bash
git clone https://github.com/finefigo/Imagecaption_genarator.git
cd Imagecaption_genarator
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

### 4. Set Up API Key
Get a free API key from [Google AI Studio](https://aistudio.google.com/apikey) and update it in `models/caption_engine.py`.

### 5. Run the App
```bash
streamlit run streamlit_app.py
```

The app opens at `http://localhost:8501`. No large model downloads — Gemini runs in the cloud!

---

## 📁 Project Structure

```
├── streamlit_app.py          # Main Streamlit application
├── requirements.txt          # Python dependencies
├── README.md                 # This file
├── CUSTOM_MODEL_GUIDE.md     # Guide to train your own model
├── models/
│   ├── __init__.py           # Package init
│   ├── caption_engine.py     # Gemini API client & caption generation
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

## 🧠 How It Works

### Stage A: Vision-to-Suggestion Pass
The raw image is sent to Gemini 2.5 Flash alongside a creative suggestion prompt. The model's built-in Vision Transformer (ViT) analyzes the image patches to identify subjects, lighting, and context, then generates a catchy, social-media-ready caption.

### Stage B: Mood Transformation Pass
If a mood other than "Normal" is selected, a **text-only** request is made (no image sent). The base caption from Stage A is combined with a mood-specific rewrite instruction. The model uses zero-shot conditioning to rewrite the caption while preserving the original meaning.

### Key AI Concepts
- **Multimodal Understanding** — Gemini processes both images and text natively
- **Autoregressive Decoding** — Word-by-word text generation based on image features and preceding words
- **Zero-Shot Conditioning** — No fine-tuning needed; the LLM already understands mood concepts from pretraining
- **Temperature Control** — Controls creativity vs. accuracy tradeoff, with mood-specific multipliers
- **Semantic Preservation** — Mood rewrites maintain the original caption's meaning

---

## 🔮 Future Extensions

- 🌍 Multilingual captions
- 🔊 Text-to-speech output
- 📷 Webcam mode
- 📊 Batch processing
- 🤖 Custom model training (see `CUSTOM_MODEL_GUIDE.md`)

---

## 📝 License

This project is for educational purposes. Powered by [Google Gemini](https://ai.google.dev/).
