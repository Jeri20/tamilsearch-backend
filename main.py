# ============================================
# Tamil Search Backend (Render Production Safe)
# ============================================

from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

app = FastAPI()

# ── Enable CORS (Allow Chrome Extension) ─────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # You can restrict later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Request Model ────────────────────────────
class InputText(BaseModel):
    text: str


# ── Load Model Once at Startup ───────────────
MODEL_NAME = "google/mt5-small"

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    use_fast=False  # 🔥 IMPORTANT FIX (prevents crash)
)

print("Loading model...")
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

print("Model loaded successfully.")


# ── Translation Function ─────────────────────
def translate_text(text: str) -> str:
    prompt = f"""
You are a product search translator.
Convert Tamil or Thanglish input into a short English product search term.
Only output the search phrase.
Input: {text}
Output:
"""

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=128
    ).to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=50,
            num_beams=4
        )

    translated = tokenizer.decode(
        outputs[0],
        skip_special_tokens=True
    )

    return translated.strip()


# ── Routes ───────────────────────────────────
@app.get("/")
def root():
    return {"status": "TamilSearch backend running"}


@app.post("/translate")
def translate(data: InputText):
    result = translate_text(data.text)
    return {"translation": result}
