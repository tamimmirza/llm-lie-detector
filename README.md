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

## Progress
- [x] Phase 1: Data exploration and local inference complete
- [ ] Phase 2: Fine-tuning with LoRA
- [ ] Phase 3: FastAPI + Docker
- [ ] Phase 4: Documentation and publishing

## Status
🚧 In active development

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