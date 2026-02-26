

import streamlit as st
from PIL import Image
import io
import time

# ── Import AI Modules ──
from models.caption_engine import CaptionEngine
from models.mood_transformer import MoodTransformer


# =============================================================================
# PAGE CONFIGURATION & CUSTOM STYLING
# =============================================================================

st.set_page_config(
    page_title="AI Caption Generator | Mood Customization",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "About": "AI-Powered Image Caption Generator with Mood Customization. "
                 "Built with Gemini Flash 3 Preview and Streamlit."
    }
)

# ── Inject Custom CSS for Premium Dark Theme ──
st.markdown("""
<style>
    /* ── Import Google Font ── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    /* ── Global Styles ── */
    .stApp {
        font-family: 'Inter', sans-serif;
    }

    /* ── Hero Header ── */
    .hero-header {
        text-align: center;
        padding: 2rem 1rem 1rem 1rem;
        margin-bottom: 1rem;
    }
    .hero-header h1 {
        font-size: 2.8rem;
        font-weight: 800;
        background: linear-gradient(135deg, #00d4ff 0%, #7b2ff7 50%, #ff6ac1 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 0.3rem;
        letter-spacing: -0.02em;
    }
    .hero-subtitle {
        font-size: 1.1rem;
        color: #8a8a9a;
        font-weight: 300;
        letter-spacing: 0.02em;
    }

    /* ── Glassmorphism Caption Cards ── */
    .caption-card {
        background: rgba(255, 255, 255, 0.04);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 1.5rem;
        margin: 0.75rem 0;
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        transition: all 0.3s ease;
        animation: fadeInUp 0.5s ease-out;
    }
    .caption-card:hover {
        background: rgba(255, 255, 255, 0.07);
        border-color: rgba(0, 212, 255, 0.3);
        transform: translateY(-2px);
        box-shadow: 0 8px 32px rgba(0, 212, 255, 0.1);
    }
    .caption-card-label {
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        color: #00d4ff;
        margin-bottom: 0.5rem;
    }
    .caption-card-text {
        font-size: 1.15rem;
        line-height: 1.7;
        color: #e8e8f0;
        font-weight: 400;
    }

    /* ── Mood Badge ── */
    .mood-badge {
        display: inline-block;
        background: linear-gradient(135deg, #7b2ff7, #00d4ff);
        color: white;
        padding: 0.3rem 0.9rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
        margin-top: 0.5rem;
    }

    /* ── Stats Row ── */
    .stats-row {
        display: flex;
        gap: 1rem;
        margin-top: 0.75rem;
        flex-wrap: wrap;
    }
    .stat-chip {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 8px;
        padding: 0.3rem 0.7rem;
        font-size: 0.75rem;
        color: #8a8a9a;
    }

    /* ── Upload Area Styling ── */
    .upload-section {
        background: rgba(255, 255, 255, 0.02);
        border: 2px dashed rgba(0, 212, 255, 0.2);
        border-radius: 16px;
        padding: 1.5rem;
        text-align: center;
        transition: all 0.3s ease;
    }
    .upload-section:hover {
        border-color: rgba(0, 212, 255, 0.5);
        background: rgba(0, 212, 255, 0.03);
    }

    /* ── Sidebar Styling ── */
    section[data-testid="stSidebar"] {
        background: rgba(10, 10, 26, 0.95) !important;
        border-right: 1px solid rgba(255, 255, 255, 0.05);
    }
    section[data-testid="stSidebar"] .stSelectbox label,
    section[data-testid="stSidebar"] .stSlider label {
        color: #c0c0d0 !important;
        font-weight: 500;
    }

    /* ── Generate Button ── */
    .stButton > button {
        background: linear-gradient(135deg, #00d4ff 0%, #7b2ff7 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 0.7rem 2rem !important;
        font-size: 1rem !important;
        font-weight: 600 !important;
        letter-spacing: 0.02em !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 15px rgba(0, 212, 255, 0.3) !important;
        width: 100% !important;
    }
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 25px rgba(0, 212, 255, 0.4) !important;
    }

    /* ── Animations ── */
    @keyframes fadeInUp {
        from { opacity: 0; transform: translateY(20px); }
        to   { opacity: 1; transform: translateY(0); }
    }
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50%      { opacity: 0.5; }
    }

    /* ── Image Preview ── */
    .image-preview-container {
        border-radius: 16px;
        overflow: hidden;
        border: 1px solid rgba(255, 255, 255, 0.08);
        margin: 1rem 0;
    }
    .image-preview-container img {
        border-radius: 16px;
    }

    /* ── Info Box ── */
    .info-box {
        background: rgba(0, 212, 255, 0.05);
        border: 1px solid rgba(0, 212, 255, 0.15);
        border-radius: 12px;
        padding: 1rem;
        font-size: 0.85rem;
        color: #8a8a9a;
        margin: 0.5rem 0;
    }

    /* ── Footer ── */
    .app-footer {
        text-align: center;
        padding: 2rem 1rem;
        color: #555;
        font-size: 0.8rem;
        border-top: 1px solid rgba(255, 255, 255, 0.05);
        margin-top: 3rem;
    }

    /* ── Hide Streamlit Default Elements ── */
    #MainMenu {visibility: hidden;}
    header {visibility: hidden;}
    footer {visibility: hidden;}

    /* ── Divider ── */
    .section-divider {
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(0,212,255,0.3), transparent);
        margin: 1.5rem 0;
    }
</style>
""", unsafe_allow_html=True)


# =============================================================================
# MODEL LOADING (Cached - creates client only once)
# =============================================================================

@st.cache_resource(show_spinner=False)
def load_models():
    """
    Initialize and cache the AI models.

    @st.cache_resource ensures the Gemini client is created ONCE when the app
    starts and shared across all user sessions. Unlike the previous BLIP model
    which required downloading ~1.5GB, the Gemini client initializes instantly
    since all computation happens on Google's cloud.

    Returns:
        Tuple of (CaptionEngine, MoodTransformer)
    """
    engine = CaptionEngine()
    transformer = MoodTransformer(engine)
    return engine, transformer


# =============================================================================
# MAIN APPLICATION
# =============================================================================

def main():
    """Main Streamlit application entry point."""

    # ── Hero Header ──
    st.markdown("""
        <div class="hero-header">
            <h1>🎨 AI Caption Generator</h1>
            <p class="hero-subtitle">
                Generate intelligent image captions with emotional mood customization
            </p>
        </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    # ── Load Models (with loading indicator) ──
    with st.spinner("🧠 Initializing Gemini AI... This is instant!"):
        engine, transformer = load_models()

    # ── Sidebar Configuration ──
    with st.sidebar:
        st.markdown("## ⚙️ Configuration")
        st.markdown("---")

        # ── Image Upload ──
        st.markdown("### 📸 Upload Image")
        uploaded_file = st.file_uploader(
            "Drag and drop or browse",
            type=["jpg", "jpeg", "png", "webp", "bmp"],
            help="Supported formats: JPG, PNG, WebP, BMP (max 10MB)",
            label_visibility="collapsed",
        )

        st.markdown("---")

        # ── Mood Selection ──
        st.markdown("### 🎭 Select Mood")
        available_moods = transformer.get_available_moods()
        mood_options = [
            f"{transformer.get_mood_emoji(m)} {m}" for m in available_moods
        ]
        selected_mood_display = st.selectbox(
            "Choose emotional tone",
            options=mood_options,
            index=0,
            help="Select the emotional tone for your caption",
            label_visibility="collapsed",
        )
        # Extract mood name (remove emoji prefix)
        selected_mood = selected_mood_display.split(" ", 1)[1]

        st.markdown("---")

        # ── Creativity Slider ──
        st.markdown("### 🌡️ Creativity Level")
        creativity = st.slider(
            "Adjust creativity",
            min_value=0.1,
            max_value=1.5,
            value=0.7,
            step=0.1,
            help="Low = factual & safe | High = creative & expressive",
            label_visibility="collapsed",
        )

        # ── Creativity level indicator ──
        if creativity <= 0.4:
            creativity_label = "🔵 Conservative"
        elif creativity <= 0.8:
            creativity_label = "🟢 Balanced"
        elif creativity <= 1.1:
            creativity_label = "🟡 Creative"
        else:
            creativity_label = "🔴 Highly Expressive"

        st.markdown(f"""
            <div class="info-box">
                <strong>Creativity:</strong> {creativity_label}<br>
                <small>Temperature affects how adventurous the AI is with word choices.</small>
            </div>
        """, unsafe_allow_html=True)

    # ── Main Content Area ──
    if uploaded_file is not None:
        # ── Load and Display Image ──
        image = Image.open(uploaded_file).convert("RGB")

        col1, col2 = st.columns([1, 1], gap="large")

        with col1:
            st.markdown("### 📷 Uploaded Image")
            st.image(
                image,
                width='stretch',
                caption=f"📐 {image.size[0]}×{image.size[1]}px | 📁 {uploaded_file.name}",
            )

        with col2:
            st.markdown("### 🎯 Caption Generation")

            # ── Generate Button ──
            generate_clicked = st.button(
                "✨ Generate Caption",
            )

            if generate_clicked:
                # ── Step 1: Generate Base Caption ──
                with st.spinner("🔍 Analyzing image with Gemini AI..."):
                    start_time = time.time()
                    base_caption = engine.generate_caption(
                        image=image,
                        temperature=0.7,
                    )
                    base_time = time.time() - start_time

                # ── Store in session state ──
                st.session_state["base_caption"] = base_caption
                st.session_state["base_time"] = base_time

                # ── Step 2: Generate Mood Caption ──
                with st.spinner(f"🎭 Applying {selected_mood} mood transformation..."):
                    start_time = time.time()
                    result = transformer.transform(
                        base_caption=base_caption,
                        mood=selected_mood,
                        creativity=creativity,
                    )
                    mood_time = time.time() - start_time

                st.session_state["mood_result"] = result
                st.session_state["mood_time"] = mood_time
                st.session_state["total_time"] = base_time + mood_time

                st.balloons()

            # ── Display Results ──
            if "base_caption" in st.session_state:
                base_caption = st.session_state["base_caption"]
                result = st.session_state.get("mood_result", {})
                total_time = st.session_state.get("total_time", 0)

                # ── Base Caption Card ──
                st.markdown(f"""
                    <div class="caption-card">
                        <div class="caption-card-label">📝 Suggested Caption (Stage A)</div>
                        <div class="caption-card-text">{base_caption}</div>
                        <div class="stats-row">
                            <span class="stat-chip">⏱️ {st.session_state.get('base_time', 0):.1f}s</span>
                            <span class="stat-chip">📏 {len(base_caption.split())} words</span>
                        </div>
                    </div>
                """, unsafe_allow_html=True)

                # ── Mood Caption Card ──
                if result and result.get("mood") != "Normal":
                    mood_caption = result.get("mood_caption", "")
                    mood_emoji = result.get("mood_emoji", "")
                    mood_name = result.get("mood", "")

                    st.markdown(f"""
                        <div class="caption-card" style="border-color: rgba(123, 47, 247, 0.3);">
                            <div class="caption-card-label" style="color: #7b2ff7;">
                                {mood_emoji} Mood Caption ({mood_name})
                            </div>
                            <div class="caption-card-text">{mood_caption}</div>
                            <div class="stats-row">
                                <span class="stat-chip">⏱️ {st.session_state.get('mood_time', 0):.1f}s</span>
                                <span class="stat-chip">📏 {len(mood_caption.split())} words</span>
                                <span class="stat-chip">🌡️ Temp: {result.get('temperature', 0)}</span>
                                <span class="mood-badge">{mood_emoji} {mood_name}</span>
                            </div>
                        </div>
                    """, unsafe_allow_html=True)
                elif result and result.get("mood") == "Normal":
                    st.markdown(f"""
                        <div class="info-box">
                            ℹ️ <strong>Normal mood selected</strong> — showing the factual caption above. 
                            Select a different mood from the sidebar to see an emotionally enhanced version.
                        </div>
                    """, unsafe_allow_html=True)

                st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

                # ── Download Options ──
                col_dl1, col_dl2 = st.columns(2)

                with col_dl1:
                    # ── Download as Text ──
                    download_text = f"Base Caption:\n{base_caption}\n"
                    if result and result.get("mood") != "Normal":
                        download_text += f"\nMood ({result.get('mood', '')}):\n{result.get('mood_caption', '')}\n"
                        download_text += f"\nCreativity: {result.get('creativity', '')}"
                        download_text += f"\nTemperature: {result.get('temperature', '')}"

                    st.download_button(
                        label="📄 Download Caption (.txt)",
                        data=download_text,
                        file_name="ai_caption.txt",
                        mime="text/plain",
                    )

                with col_dl2:
                    # ── Performance Stats ──
                    st.markdown(f"""
                        <div class="info-box">
                            ⚡ <strong>Total time:</strong> {total_time:.1f}s<br>
                            <small>🖥️ Device: {str(engine.device).upper()}</small>
                        </div>
                    """, unsafe_allow_html=True)

    else:
        # ── Empty State ──
        st.markdown("""
            <div style="text-align: center; padding: 4rem 2rem;">
                <div style="font-size: 5rem; margin-bottom: 1rem;">📸</div>
                <h2 style="color: #c0c0d0; font-weight: 600;">Upload an Image to Begin</h2>
                <p style="color: #6a6a7a; max-width: 500px; margin: 0.5rem auto; line-height: 1.6;">
                    Use the sidebar to upload an image. The AI will analyze it using 
                    Google's Gemini model and generate intelligent captions that you can 
                    customize with different emotional moods.
                </p>
                <div class="stats-row" style="justify-content: center; margin-top: 1.5rem;">
                    <span class="stat-chip">🖼️ JPG, PNG, WebP</span>
                    <span class="stat-chip">🎭 8 Mood Styles</span>
                    <span class="stat-chip">🌡️ Adjustable Creativity</span>
                    <span class="stat-chip">⚡ Gemini Flash 3</span>
                </div>
            </div>
        """, unsafe_allow_html=True)

    # ── How It Works Section ──
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    with st.expander("🧠 How Does This Work? — AI Architecture Explained"):
        st.markdown("""
        ### The Two-Stage AI Pipeline
        
        This application uses Google's **Gemini** multimodal AI model in a 
        sophisticated two-stage pipeline:
        
        **Stage A: Vision-to-Suggestion Pass**
        - The raw image is sent to Gemini alongside a "Creative Suggestion" prompt
        - The Vision Transformer (ViT) inside Gemini analyzes the image patches 
          to identify subjects, lighting, and context
        - Output: A catchy, social-media-ready "Suggested Caption"
        - Temperature is set to **0.7** — the creative sweet spot where the AI is 
          creative but doesn't hallucinate things that aren't in the image

        **Stage B: Mood Transformation Pass (Text-Only)**
        - If you select a mood (like "Funny" or "Dramatic"), a second **text-only** 
          request is made — no image is sent
        - The base caption from Stage A + a mood-specific rewrite instruction are 
          sent to the model
        - The model uses its linguistic knowledge to "reskin" the caption while 
          keeping the original meaning (semantic preservation)
        - This is faster and cheaper since no image processing is needed

        **Temperature Control — Creativity vs. Accuracy**
        - Temperature controls the randomness of word selection during generation
        - Low temperature (0.1-0.4): Safe, predictable, factual captions
        - Medium temperature (0.5-0.8): Balanced creativity (recommended)
        - High temperature (0.9-1.5): Creative, expressive, and diverse outputs
        - For moods, temperature is scaled by a mood-specific multiplier 
          (e.g., Poetic × 1.3) for natural expressiveness
        
        **Key Concepts:**
        - *Autoregressive Decoding*: The model generates the caption word-by-word, 
          each word chosen based on image features AND preceding words
        - *Cross-Attention*: The "bridge" between vision and language — while writing, 
          the model "attends" (looks back) at specific parts of the image embeddings
        - *Zero-Shot Conditioning*: No training needed for moods — the LLM already 
          understands concepts like "Funny" or "Poetic" from its pretraining
        """)

    # ── Footer ──
    st.markdown("""
        <div class="app-footer">
            <p>🎨 <strong>AI Caption Generator</strong> — Powered by Gemini Flash 3 Preview × Streamlit</p>
            <p>Multimodal AI • Instruction-Based Generation • Mood Transformation • Cloud-Powered</p>
        </div>
    """, unsafe_allow_html=True)


# ── Run the App ──
if __name__ == "__main__":
    main()
