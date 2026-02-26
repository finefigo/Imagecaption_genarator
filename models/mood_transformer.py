

from PIL import Image
from typing import Optional


class MoodTransformer:
    

    # ── Mood Instruction Templates (Text-Only Rewrite Instructions) ──
    # Stage B: Each template tells Gemini how to "reskin" the base caption
    # from Stage A. These are TEXT-ONLY instructions — no image is sent.
    # The model uses its linguistic knowledge to rewrite while preserving
    # the original meaning (semantic preservation).
    MOOD_PROMPTS = {
        "Normal": None,  # No transformation → return base caption

        "Happy": (
            "Rewrite this caption to be incredibly joyful, vibrant, and full "
            "of positive energy. Perfect for a happy memory."
        ),

        "Sad": (
            "Rewrite this caption to be melancholic, wistful, and deeply "
            "reflective. Use soft, bittersweet language that evokes nostalgia."
        ),

        "Funny": (
            "Rewrite this caption to be hilarious, witty, or a clever pun. "
            "Make it something that would get a laugh."
        ),

        "Romantic": (
            "Rewrite this caption to be dreamy, intimate, and deeply romantic. "
            "Use soft and affectionate language."
        ),

        "Dramatic": (
            "Rewrite this caption to be intense, cinematic, and epic. "
            "Make it feel like a high-stakes movie moment."
        ),

        "Inspirational": (
            "Rewrite this caption to be uplifting, empowering, and full of hope. "
            "Make it something that motivates and inspires."
        ),

        "Poetic": (
            "Rewrite this caption as a beautiful, rhythmic, and metaphorical "
            "short poem or a piece of elegant prose."
        ),
    }

    # ── Mood-Specific Generation Parameters ──
    # Different moods benefit from different temperature baselines.
    MOOD_PARAMS = {
        "Normal":        {"temp_multiplier": 0.8},
        "Happy":         {"temp_multiplier": 1.0},
        "Sad":           {"temp_multiplier": 1.0},
        "Funny":         {"temp_multiplier": 1.2},
        "Romantic":      {"temp_multiplier": 1.1},
        "Dramatic":      {"temp_multiplier": 1.1},
        "Inspirational": {"temp_multiplier": 1.0},
        "Poetic":        {"temp_multiplier": 1.3},
    }

    # ── Mood Emojis (for UI display) ──
    MOOD_EMOJIS = {
        "Normal": "📝",
        "Happy": "😊",
        "Sad": "😢",
        "Funny": "😂",
        "Romantic": "💕",
        "Dramatic": "🎭",
        "Inspirational": "✨",
        "Poetic": "🌸",
    }

    def __init__(self, caption_engine):
        """
        Initialize the MoodTransformer with a reference to the CaptionEngine.

        We reuse the SAME Gemini client from CaptionEngine — no additional
        setup required. This is efficient because:
        1. Only one API client is created (shared connection)
        2. No extra model loading or memory usage
        3. Same API key and configuration

        Args:
            caption_engine: An initialized CaptionEngine instance
        """
        self.engine = caption_engine

    def get_available_moods(self) -> list:
        """Return list of all available mood options."""
        return list(self.MOOD_PROMPTS.keys())

    def get_mood_emoji(self, mood: str) -> str:
        """Return the emoji icon for a given mood."""
        return self.MOOD_EMOJIS.get(mood, "📝")

    def transform(
        self,
        base_caption: str,
        mood: str = "Normal",
        creativity: float = 0.7,
    ) -> dict:
        """
        Stage B: Mood Transformation Pass (text-only).

        Takes the Suggested Caption from Stage A and rewrites it with
        the desired mood/style. This is a TEXT-ONLY operation — no image
        is sent, making it faster and cheaper.

        Workflow:
        1. If mood is "Normal", return the base caption unchanged
        2. Look up the mood's rewrite instruction
        3. Calculate effective temperature from creativity × mood multiplier
        4. Send base caption + mood instruction to Gemini (text-only)
        5. Clean up the output
        6. Return both captions + metadata

        Args:
            base_caption: The suggested caption from Stage A
            mood: One of the supported moods (Normal, Happy, Sad, etc.)
            creativity: User-selected creativity level (0.1 to 1.5)

        Returns:
            Dictionary with base_caption, mood_caption, mood, emoji, etc.
        """
        mood_emoji = self.get_mood_emoji(mood)

        # ── Handle "Normal" mood (no transformation needed) ──
        if mood == "Normal" or mood not in self.MOOD_PROMPTS:
            return {
                "base_caption": base_caption,
                "mood_caption": base_caption,
                "mood": mood,
                "mood_emoji": mood_emoji,
                "creativity": creativity,
                "temperature": 1.0,
            }

        # ── Get mood-specific parameters ──
        mood_instruction = self.MOOD_PROMPTS[mood]
        params = self.MOOD_PARAMS[mood]

        # ── Calculate effective temperature ──
        # Final temperature = user's creativity × mood's base multiplier
        # Clamped to Gemini's valid range [0.0, 2.0]
        effective_temperature = min(2.0, max(0.1, creativity * params["temp_multiplier"]))

        # ── Stage B: Text-only mood rewrite (no image sent) ──
        mood_caption = self.engine.rewrite_caption(
            base_caption=base_caption,
            mood_instruction=mood_instruction,
            temperature=effective_temperature,
        )

        # ── Clean up the generated text ──
        mood_caption = self._clean_caption(mood_caption)

        return {
            "base_caption": base_caption,
            "mood_caption": mood_caption,
            "mood": mood,
            "mood_emoji": mood_emoji,
            "creativity": creativity,
            "temperature": round(effective_temperature, 2),
        }

    def _clean_caption(self, caption: str) -> str:
        """
        Clean up the generated mood caption.

        Post-processing steps:
        1. Strip whitespace
        2. Capitalize first letter
        3. Ensure proper sentence ending
        4. Remove surrounding quotes if present

        Args:
            caption: Raw generated text from Gemini

        Returns:
            Cleaned, properly formatted caption string
        """
        caption = caption.strip()

        # Remove surrounding quotes
        if caption.startswith('"') and caption.endswith('"'):
            caption = caption[1:-1].strip()

        # Capitalize first letter
        if caption:
            caption = caption[0].upper() + caption[1:]

        # Ensure the caption ends with proper punctuation
        if caption and caption[-1] not in ".!?":
            caption += "."

        return caption
