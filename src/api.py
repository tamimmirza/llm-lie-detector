import os
os.environ["PYTHONUTF8"] = "1"

import torch
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from huggingface_hub import login
from dotenv import load_dotenv

load_dotenv('../.env')

# HuggingFace login
HF_TOKEN = os.environ.get("HF_TOKEN")
login(token=HF_TOKEN)

# Config
MODEL_ID = "meta-llama/Llama-3.2-3B-Instruct"
ADAPTER_PATH = "/app/outputs/checkpoints"

# Find best checkpoint
checkpoints = [f for f in os.listdir(ADAPTER_PATH) if f.startswith("checkpoint")]
checkpoints.sort()
BEST_CHECKPOINT = os.path.join(ADAPTER_PATH, checkpoints[-1])

print(f"Loading model from {BEST_CHECKPOINT}...")

# Detect device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

# Load model
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
tokenizer.pad_token = tokenizer.eos_token

base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    dtype=torch.float32 if DEVICE == "cpu" else torch.bfloat16,
    device_map="auto" if DEVICE == "cuda" else None
)

model = PeftModel.from_pretrained(base_model, BEST_CHECKPOINT)
model.eval()

print("Model loaded and ready.")

# FastAPI app
app = FastAPI(
    title="LLM Lie Detector",
    description="Detects hallucinations in LLM-generated answers",
    version="1.0.0"
)

# Request/Response models
class DetectionRequest(BaseModel):
    question: str
    answer: str

class DetectionResponse(BaseModel):
    question: str
    answer: str
    verdict: str
    confidence: str

# Inference function
def predict(question: str, answer: str) -> dict:
    prompt = f"Question: {question}\nAnswer: {answer}\nVerdict:"
    
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=10,
            temperature=0.1,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    response = tokenizer.decode(
        output[0][inputs['input_ids'].shape[-1]:],
        skip_special_tokens=True
    ).strip().upper()
    
    if "HALLUCINATED" in response:
        return {"verdict": "HALLUCINATED", "confidence": "high"}
    elif "TRUTHFUL" in response:
        return {"verdict": "TRUTHFUL", "confidence": "high"}
    else:
        return {"verdict": "UNCERTAIN", "confidence": "low"}

# Routes
@app.get("/")
def root():
    return {"message": "LLM Lie Detector API", "status": "running"}

@app.get("/health")
def health():
    return {"status": "healthy", "model": "Llama-3.2-3B-Instruct + LoRA"}

@app.post("/detect", response_model=DetectionResponse)
def detect(request: DetectionRequest):
    result = predict(request.question, request.answer)
    return DetectionResponse(
        question=request.question,
        answer=request.answer,
        verdict=result["verdict"],
        confidence=result["confidence"]
    )