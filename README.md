# LLM Lie Detector 🔍

A hallucination detection pipeline for Large Language Models (LLMs).

## The Problem
LLMs confidently produce incorrect information — a phenomenon known as hallucination. 
A model might tell you fortune cookies originated in China, or that the Declaration of 
Independence was signed on July 4th. Both wrong, both stated with complete confidence. 
This is one of the most critical unsolved problems in AI today.

## The Solution
This project fine-tunes a small language model to act as a hallucination detector — 
given a question and an LLM-generated answer, it predicts whether that answer is 
factually grounded or hallucinated. The finished system is wrapped in a REST API 
and shipped as a Docker container.

## Status
🚧 In active development

## Results

| Metric    | Baseline | Our Model | Improvement |
|-----------|----------|-----------|-------------|
| F1 Score  | 0.3595   | 0.9032    | +0.5438     |
| Precision | 0.2738   | 0.9078    | +0.6340     |
| Recall    | 0.5232   | 0.9033    | +0.3800     |
| Accuracy  | —        | 90.33%    | —           |

Model: Llama 3.2 3B Instruct fine-tuned with LoRA  
Dataset: TruthfulQA + HaluEval (15,918 labeled pairs)  
Validation set: 1,592 samples

## Progress

### ✅ Phase 1 — Foundations
- Explored TruthfulQA and HaluEval datasets
- Built unified labeled dataset (15,918 training pairs)
- Ran local inference with Llama 3.2 3B, observed hallucinations firsthand

### ✅ Phase 2 — Fine-tuning
- Fine-tuned Llama 3.2 3B with LoRA (trained only 0.14% of parameters)
- Achieved **90% accuracy** and **F1 score of 0.90** on validation set
- +0.54 F1 improvement over majority class baseline
- Experiment tracked with Weights & Biases

### 🔄 Phase 3 — The Product (In Progress)
- [ ] FastAPI inference endpoint
- [ ] Docker container
- [ ] Demo GIF

### ⬜ Phase 4 — Documentation & Publishing
- [ ] Polish README with architecture diagram
- [ ] Push model to HuggingFace Hub
- [ ] LinkedIn post

## Development Notes

### Training Pipeline — Issues & Pivots

**Initial approach:** Fine-tuned Llama 3.2 3B using `AutoModelForSequenceClassification` 
with manual LoRA via `get_peft_model()` and the standard HuggingFace `Trainer`.

**Issues encountered on Windows / RTX 4080 Laptop GPU (12GB VRAM):**

1. **fp16 gradient scaling crash** — The initial training config used `fp16=True` which 
caused a gradient unscaling error with LoRA adapters. Resolved by switching to `bf16=True` 
which the RTX 4080 Laptop supports natively and is more stable for LLM fine-tuning.

2. **Training running at 0.02 it/s** — After the fp16 fix, training appeared to start but 
ran at near-zero speed. Root cause: the model had silently fallen back to CPU during 
repeated kernel restarts and cell reruns. GPU showed 0.0GB allocated despite 
`torch.cuda.is_available()` returning True.

3. **Double LoRA application** — When pivoting to `SFTTrainer` from the `trl` library, 
the old `get_peft_model()` cell was still present in the notebook. This caused 
`SFTTrainer` to raise a `ValueError` as it detected an already-adapted PeftModel 
when trying to apply its own LoRA config.

**Resolution:** Switched to `SFTTrainer` from `trl`, which handles LoRA application 
internally via `peft_config`, manages the training loop cleanly, and is purpose-built 
for this exact workflow. Removed all manual LoRA cells. Training now runs at ~1.2 it/s 
on GPU as expected.