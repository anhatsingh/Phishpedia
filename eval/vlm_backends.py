"""
Unified VLM verifier interface across three backends:

  - gemini    : Google Gemini Flash (free tier, 15 RPM / 1M tok/day).
                Set GEMINI_API_KEY (https://aistudio.google.com/apikey).
  - local     : Qwen2-VL-2B-Instruct via HuggingFace, runs on Colab T4 GPU.
                No API key, just a model download (~5 GB VRAM).
  - anthropic : Claude vision (paid). Set ANTHROPIC_API_KEY.

All backends expose:   verifier.verify(screenshot_path, url, brand) -> bool
"""
from __future__ import annotations

import base64, os, sys
from pathlib import Path
from typing import Protocol


PROMPT = """You are reviewing a webpage screenshot to decide whether it is impersonating a brand.

URL: {url}
Brand allegedly impersonated: {brand}

Reply YES only if ALL of:
1. The {brand} logo is the page's main identity (header / login form / title), NOT a third-party SSO button, ad banner, or content thumbnail.
2. The page asks for credentials, payment info, or other sensitive data (login form, password field, card form, etc.).
3. The URL clearly does NOT belong to {brand}.

Otherwise reply NO. Reply with exactly one word: YES or NO."""


class Verifier(Protocol):
    def verify(self, screenshot_path: str, url: str, brand: str) -> bool: ...


def _read_image_b64(path: str) -> tuple[bytes, str]:
    data = Path(path).read_bytes()
    media = "image/png" if path.lower().endswith(".png") else "image/jpeg"
    return data, media


# ----------------------------- Gemini (free tier) -----------------------------

class GeminiVerifier:
    def __init__(self, model: str = "gemini-2.0-flash"):
        try:
            import google.generativeai as genai
        except ImportError:
            import subprocess
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-q",
                                   "google-generativeai"])
            import google.generativeai as genai
        key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not key:
            sys.exit("[error] set GEMINI_API_KEY (free at "
                     "https://aistudio.google.com/apikey)")
        genai.configure(api_key=key)
        self._genai = genai
        self.model = genai.GenerativeModel(model)

    def verify(self, screenshot_path: str, url: str, brand: str) -> bool:
        data, media = _read_image_b64(screenshot_path)
        resp = self.model.generate_content(
            [PROMPT.format(url=url, brand=brand),
             {"mime_type": media, "data": data}],
            generation_config={"max_output_tokens": 8, "temperature": 0.0},
        )
        try:
            txt = resp.text or ""
        except Exception:
            txt = ""
        return txt.strip().upper().startswith("YES")


# ----------------------------- Local (Qwen2-VL) -------------------------------

class LocalVerifier:
    """Qwen2-VL-2B-Instruct on the available device. ~5 GB VRAM on Colab T4."""

    def __init__(self, model_id: str = "Qwen/Qwen2-VL-2B-Instruct"):
        import torch
        try:
            from transformers import (Qwen2VLForConditionalGeneration,
                                      AutoProcessor)
        except ImportError:
            import subprocess
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-q",
                                   "transformers", "accelerate", "qwen-vl-utils"])
            from transformers import (Qwen2VLForConditionalGeneration,
                                      AutoProcessor)

        self.torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.bfloat16 if self.device == "cuda" else torch.float32
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_id, torch_dtype=dtype, device_map=self.device).eval()
        self.proc = AutoProcessor.from_pretrained(model_id)

    @property
    def _no_grad(self):
        return self.torch.inference_mode()

    def verify(self, screenshot_path: str, url: str, brand: str) -> bool:
        from PIL import Image
        img = Image.open(screenshot_path).convert("RGB")
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": img},
                {"type": "text",  "text": PROMPT.format(url=url, brand=brand)},
            ],
        }]
        text = self.proc.apply_chat_template(messages, tokenize=False,
                                             add_generation_prompt=True)
        inputs = self.proc(text=[text], images=[img],
                           return_tensors="pt", padding=True).to(self.device)
        with self._no_grad:
            out = self.model.generate(**inputs, max_new_tokens=8,
                                      do_sample=False)
        gen = out[:, inputs["input_ids"].shape[1]:]
        txt = self.proc.batch_decode(gen, skip_special_tokens=True)[0]
        return txt.strip().upper().startswith("YES")


# ----------------------------- Anthropic (paid) -------------------------------

class AnthropicVerifier:
    def __init__(self, model: str = "claude-haiku-4-5"):
        if not os.environ.get("ANTHROPIC_API_KEY"):
            sys.exit("[error] set ANTHROPIC_API_KEY")
        try:
            import anthropic
        except ImportError:
            import subprocess
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-q",
                                   "anthropic"])
            import anthropic
        self.client = anthropic.Anthropic()
        self.model = model

    def verify(self, screenshot_path: str, url: str, brand: str) -> bool:
        data, media = _read_image_b64(screenshot_path)
        msg = self.client.messages.create(
            model=self.model, max_tokens=8,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image", "source": {
                        "type": "base64", "media_type": media,
                        "data": base64.standard_b64encode(data).decode()}},
                    {"type": "text",
                     "text": PROMPT.format(url=url, brand=brand)},
                ],
            }],
        )
        txt = "".join(b.text for b in msg.content if getattr(b, "type", None) == "text")
        return txt.strip().upper().startswith("YES")


# --------------------------------- Factory ------------------------------------

def get_verifier(backend: str, model: str | None = None) -> Verifier:
    backend = backend.lower()
    if backend == "gemini":
        return GeminiVerifier(model or "gemini-2.0-flash")
    if backend == "local":
        return LocalVerifier(model or "Qwen/Qwen2-VL-2B-Instruct")
    if backend == "anthropic":
        return AnthropicVerifier(model or "claude-haiku-4-5")
    sys.exit(f"[error] unknown VLM backend: {backend}")
