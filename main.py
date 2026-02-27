from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class InputText(BaseModel):
    text: str

MODEL_NAME = "google/mt5-small"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


def translate_text(text: str) -> str:
    prompt = f"""
You are a product search translator.
Convert Tamil or Thanglish input into a short English product search term.
Only output the search phrase.
Input: {text}
Output:
"""

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True).to(device)

    with torch.no_grad():
        outputs = model.generate(**inputs, max_length=50, num_beams=4)

    translated = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return translated.strip()


@app.get("/")
def root():
    return {"status": "TamilSearch backend running"}


@app.post("/translate")
def translate(data: InputText):
    result = translate_text(data.text)
    return {"translation": result}
