# LLM Lie Detector 🔍

A hallucination detection system for Large Language Models (LLMs), built with a fine-tuned Llama 3.2 3B and deployed as a production-ready REST API.

## What It Does

Given a question and an LLM-generated answer, the system predicts whether the answer is factually grounded or hallucinated. It returns a verdict, a confidence score, and supports a three-class output mode (TRUTHFUL / UNCERTAIN / HALLUCINATED) for production deployment.

## Demo

![LLM Lie Detector Demo](demo.gif)

## Key Numbers

| Metric     | Score            |
|------------|------------------|
| F1 Score   | 0.9234           |
| Accuracy   | 92.34%           |
| ROC AUC    | 0.9637           |
| Latency    | 210ms            |
| Throughput | 4.75 samples/sec |
| VRAM       | 6.45 GB          |

## Model

Fine-tuned LoRA adapter available on HuggingFace Hub:
👉 [tamimmirza/llama-3.2-3b-hallucination-detector](https://huggingface.co/tamimmirza/llama-3.2-3b-hallucination-detector)

## Stack

- **Model:** Meta Llama 3.2 3B Instruct + LoRA (PEFT)
- **Training:** SFTTrainer (trl), bfloat16, Weights & Biases
- **Data:** TruthfulQA + HaluEval (15,918 labeled pairs)
- **API:** FastAPI + uvicorn
- **Deployment:** Docker
- **Hardware:** NVIDIA RTX 4080 Laptop GPU (12GB VRAM)

## How to Run

### Option 1 -- Docker (Recommended)

```bash
git clone https://github.com/tamimmirza/llm-lie-detector.git
cd llm-lie-detector
docker run -p 8000:8000 --env-file .env tamimmirza/llm-lie-detector
```

Then visit `http://127.0.0.1:8000/docs`

### Option 2 -- Local

```bash
pip install -r requirements-api.txt
cd src
uvicorn api:app --reload
```

## API Usage

```bash
curl -X POST http://127.0.0.1:8000/detect \
  -H "Content-Type: application/json" \
  -d '{"question": "Where did fortune cookies originate?",
       "answer": "Fortune cookies originated in China."}'
```

Response:

```json
{
  "question": "Where did fortune cookies originate?",
  "answer": "Fortune cookies originated in China.",
  "verdict": "HALLUCINATED",
  "confidence": "high"
}
```

## Three-Class Output

The system supports a confidence-aware three-class mode for production deployment:

| Zone              | Confidence  | Action       |
|-------------------|-------------|--------------|
| High HALLUCINATED | > 0.85      | Auto-flag    |
| UNCERTAIN         | 0.15 - 0.85 | Human review |
| High TRUTHFUL     | < 0.15      | Auto-pass    |

Accuracy on confident predictions: **93.5%** with 23.5% abstention rate.

## Hardware Requirements

| Component             | Requirement        |
|-----------------------|--------------------|
| VRAM (minimum)        | 8 GB               |
| VRAM (used)           | 6.45 GB            |
| GPU inference latency | ~210ms             |
| CPU inference         | Supported (slower) |

## Reproducibility

```bash
# Install dependencies
pip install -r requirements-api.txt

# Set environment variables
cp .env.example .env  # Add your HF_TOKEN

# Run locally
cd src && uvicorn api:app --reload
```

All training notebooks are available in `notebooks/`. The fine-tuned adapter
is publicly available on HuggingFace Hub and can be loaded directly.

## Training Notes

Fine-tuning used LoRA (r=16, alpha=32) on 14,326 training samples with
SFTTrainer from the trl library. Training completed in approximately 54 minutes
on a single RTX 4080 Laptop GPU using bfloat16 precision.

Key issue resolved during development: the standard HuggingFace Trainer had
compatibility issues with LoRA adapters in this library stack. SFTTrainer
handles LoRA internally and resolved all training instability.

## Ongoing Investigation

As an extension of this project, we are investigating whether retrieval
augmentation improves fine-tuned hallucination detection. Early results
suggest retrieval may not always help -- full findings pending.

## Project Status

Active development. This project is part of ongoing research.
Please cite appropriately if building on this work.

## Acknowledgements

Development was conducted with assistance from Claude (Anthropic) for
guidance, debugging, and code review.
