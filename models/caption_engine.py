

import time as _time
from PIL import Image
from google import genai
from google.genai import types


class CaptionEngine:
    

    # ── Gemini API Key ──
    API_KEY = "AIzaSyC7Ag7mh8qLI7cfcBUrKl_3PYsNPEpaCkc"

    # ── Max retries for rate-limited requests ──
    MAX_RETRIES = 5
    RETRY_DELAY = 35  # seconds (Gemini free-tier needs ~30s cooldown)

    def __init__(self, model_name: str = "gemini-2.5-flash"):
       
        print(f"[CaptionEngine] Initializing Gemini client...")
        print(f"[CaptionEngine] Model: {model_name}")

        self.model_name = model_name
        self.client = genai.Client(api_key=self.API_KEY)
        self.device = "Gemini API (Cloud)"

        print("[CaptionEngine] Gemini client ready!")

    def _call_with_retry(self, prompt, image, temperature, max_tokens):
        """Call the Gemini API with automatic retry on rate limit errors."""
        for attempt in range(self.MAX_RETRIES):
            try:
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=[prompt, image],
                    config=types.GenerateContentConfig(
                        temperature=temperature,
                        max_output_tokens=max_tokens,
                    ),
                )
                return response.text.strip()
            except Exception as e:
                error_str = str(e)
                if "429" in error_str or "RESOURCE_EXHAUSTED" in error_str:
                    if attempt < self.MAX_RETRIES - 1:
                        wait = self.RETRY_DELAY * (attempt + 1)
                        print(f"[CaptionEngine] Rate limited. Retrying in {wait}s... (attempt {attempt + 2}/{self.MAX_RETRIES})")
                        _time.sleep(wait)
                        continue
                raise e
        return None

    def generate_caption(
        self,
        image: Image.Image,
        temperature: float = 0.7,
        **kwargs,
    ) -> str:
        """
        Stage A: Vision-to-Suggestion Pass.

        The Vision Transformer (ViT) inside Gemini analyzes the image patches
        to identify subjects, lighting, and context. Produces a catchy,
        social-media-ready "Suggested Caption."

        Args:
            image: PIL Image object (any size — Gemini handles resizing)
            temperature: Controls randomness. 0.7 = creative sweet spot
            **kwargs: Accepts additional args for backward compatibility

        Returns:
            A catchy, creative suggested caption string
        """
        prompt = (
            "Suggest a catchy, creative, and contextually relevant caption "
            "for this image. It should be something someone would actually use "
            "on social media or in a photo album. Keep it engaging. "
            "Write only the caption, nothing else."
        )

        try:
            caption = self._call_with_retry(prompt, image, temperature, max_tokens=100)
            # Clean up: remove surrounding quotes if present
            if caption and caption.startswith('"') and caption.endswith('"'):
                caption = caption[1:-1]
            return caption or "Could not generate caption."

        except Exception as e:
            print(f"[CaptionEngine] Error generating caption: {e}")
            return f"Error generating caption: {str(e)}"

    def _call_text_only_with_retry(self, prompt, temperature, max_tokens):
        """Call the Gemini API with text-only input (no image) with retry."""
        for attempt in range(self.MAX_RETRIES):
            try:
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=[prompt],
                    config=types.GenerateContentConfig(
                        temperature=temperature,
                        max_output_tokens=max_tokens,
                    ),
                )
                return response.text.strip()
            except Exception as e:
                error_str = str(e)
                if "429" in error_str or "RESOURCE_EXHAUSTED" in error_str:
                    if attempt < self.MAX_RETRIES - 1:
                        wait = self.RETRY_DELAY * (attempt + 1)
                        print(f"[CaptionEngine] Rate limited. Retrying in {wait}s... (attempt {attempt + 2}/{self.MAX_RETRIES})")
                        _time.sleep(wait)
                        continue
                raise e
        return None

    def rewrite_caption(
        self,
        base_caption: str,
        mood_instruction: str,
        temperature: float = 0.7,
    ) -> str:
        """
        Stage B: Mood Transformation Pass (text-only, no image).

        Takes the Suggested Caption from Stage A and "reskins" it using
        the model's linguistic knowledge. This is a TEXT-ONLY request —
        no image is sent, making it faster and cheaper.

        The model uses autoregressive decoding to rewrite the caption
        word-by-word, preserving the original meaning (semantic preservation)
        while applying the desired mood/style.

        Args:
            base_caption: The suggested caption from Stage A
            mood_instruction: The mood-specific rewrite instruction
            temperature: Creativity control (adjustable via slider)

        Returns:
            Mood-transformed caption string
        """
        prompt = (
            f'Here is a caption: "{base_caption}"\n\n'
            f'{mood_instruction}\n\n'
            f'Write only the rewritten caption, nothing else.'
        )

        try:
            caption = self._call_text_only_with_retry(prompt, temperature, max_tokens=200)
            # Clean up: remove surrounding quotes if present
            if caption and caption.startswith('"') and caption.endswith('"'):
                caption = caption[1:-1]
            return caption or "Could not generate mood caption."

        except Exception as e:
            print(f"[CaptionEngine] Error generating mood caption: {e}")
            return f"Error generating mood caption: {str(e)}"
