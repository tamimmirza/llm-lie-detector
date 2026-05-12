# LLM Lie Detector 🔍

A hallucination detection pipeline for Large Language Models (LLMs), with a RAG-augmented verification study.

## The Problem

LLMs confidently produce incorrect information, a phenomenon known as hallucination.
A model might tell you fortune cookies originated in China, or that the Declaration of
Independence was signed on July 4th. Both wrong, both stated with complete confidence.
This is one of the most critical unsolved problems in AI today.

## The Solution

This project fine-tunes a small language model to act as a hallucination detector.
Given a question and an LLM-generated answer, it predicts whether that answer is
factually grounded or hallucinated. The system is wrapped in a REST API and shipped
as a Docker container. A RAG-augmented verification pipeline was built and evaluated
as part of Phase 5 -- revealing that naive retrieval actively hurts detection performance.

## Demo

![LLM Lie Detector Demo](demo.gif)

## Architecture

### Base System (v1.0)

```
┌─────────────────────────────────────────────────────────┐
│                        INPUT                             │
│            Question + LLM-Generated Answer               │
└─────────────────────────┬───────────────────────────────┘
                          │
                          ▼
          ┌───────────────────────────────┐
          │       FastAPI Endpoint         │
          │         POST /detect           │
          └───────────────┬───────────────┘
                          │
                          ▼
          ┌───────────────────────────────┐
          │  Llama 3.2 3B + LoRA Adapter  │
          │ Fine-tuned on TruthfulQA      │
          │ + HaluEval (15,918 pairs)     │
          └───────────────┬───────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────┐
│                       OUTPUT                             │
│            TRUTHFUL or HALLUCINATED                      │
│                   F1 Score: 0.92                         │
└─────────────────────────────────────────────────────────┘
```

### RAG-Augmented System (Phase 5 Study)

```
┌─────────────────────────────────────────────────────────┐
│                        INPUT                             │
│            Question + LLM-Generated Answer               │
└──────────────────────┬──────────────────────────────────┘
                       │
         ┌─────────────┴─────────────┐
         │                           │
         ▼                           ▼
┌─────────────────┐      ┌───────────────────────┐
│  RAG Retriever  │      │  Fine-tuned Detector   │
│ Wikipedia/FAISS │      │  Llama 3.2 3B + LoRA   │
└────────┬────────┘      └───────────┬────────────┘
         │                           │
         └─────────────┬─────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────┐
│                       OUTPUT                             │
│            TRUTHFUL or HALLUCINATED                      │
│               F1 Score: 0.72 (RAG hurts)                 │
└─────────────────────────────────────────────────────────┘
```

## Results

### Base Detector (System A)

| Metric    | Baseline | Our Model | Improvement |
|-----------|----------|-----------|-------------|
| F1 Score  | 0.3595   | 0.9234    | +0.5639     |
| Precision | 0.2738   | 0.9236    | +0.6498     |
| Recall    | 0.5232   | 0.9234    | +0.4002     |
| Accuracy  | --       | 92.34%    | --          |

Model: Llama 3.2 3B Instruct fine-tuned with LoRA
Dataset: TruthfulQA + HaluEval (15,918 labeled pairs)
Validation set: 1,592 samples

### RAG Ablation Study (Phase 5)

| Metric    | System A (Base) | System B (RAG) | Delta    |
|-----------|-----------------|----------------|----------|
| F1 Score  | 0.9234          | 0.7236         | -0.1998  |
| Precision | 0.9236          | 0.8046         | -0.1190  |
| Recall    | 0.9234          | 0.7356         | -0.1878  |
| Accuracy  | 92.34%          | 73.56%         | -18.78%  |

Key finding: RAG reduces F1 by 0.20 points. For every case where retrieval
helped, it hurt in 8 others (341 vs 42 cases). Naive Wikipedia retrieval
actively hurts hallucination detection.

## Model

The fine-tuned LoRA adapter is publicly available on HuggingFace Hub:
👉 [tamimmirza/llama-3.2-3b-hallucination-detector](https://huggingface.co/tamimmirza/llama-3.2-3b-hallucination-detector)

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

## Progress

### Phase 1 -- Foundations ✅
- [x] Explored TruthfulQA and HaluEval datasets
- [x] Built unified labeled dataset (15,918 training pairs)
- [x] Ran local inference with Llama 3.2 3B, observed hallucinations firsthand

### Phase 2 -- Fine-tuning ✅
- [x] Fine-tuned Llama 3.2 3B with LoRA (trained only 0.14% of parameters)
- [x] Achieved **92.34% accuracy** and **F1 score of 0.92** on full validation set
- [x] +0.56 F1 improvement over majority class baseline
- [x] Experiment tracked with Weights & Biases

### Phase 3 -- The Product ✅
- [x] Built FastAPI REST endpoint serving hallucination predictions over HTTP
- [x] Containerized with Docker -- runs with a single `docker run` command
- [x] Automatic CPU/GPU detection for portability
- [x] Tested successfully inside container
- [x] Demo GIF recorded and embedded in README

### Phase 4 -- Publishing ✅
- [x] Architecture diagram added to README
- [x] Model pushed to HuggingFace Hub with full model card

### Phase 5 -- RAG-Augmented Detection Study ✅
- [x] Built Wikipedia retrieval pipeline using wikipedia-api
- [x] Ran full ablation study on 1,592 samples: System A vs System B
- [x] Key finding: RAG reduces F1 by 0.20 -- naive retrieval hurts detection
- [x] Identified two failure modes: irrelevant retrieval and vague hallucination
- [x] Error analysis: 341 cases where RAG hurt vs 42 where it helped (8:1 ratio)
- [x] Publication-quality visualizations generated and saved to outputs/

### Phase 6 -- Extended Improvements 🔄
- [x] Confidence calibration -- ROC AUC 0.9637, Brier Score 0.1088
- [x] Inference performance metrics -- 210ms latency, 4.75 samples/sec, 6.45GB VRAM
- [x] Benchmark comparison -- DeBERTa baseline and GPT-4 judge
- [ ] Gradio web UI for interactive demo
- [ ] Multi-label output: TRUTHFUL / HALLUCINATED / UNCERTAIN
- [ ] IEEE technical report

## Key Findings

**Fine-tuned hallucination detection works well.** Llama 3.2 3B fine-tuned
with LoRA on 15,918 labeled pairs achieves 92% accuracy with 0 unparseable
predictions out of 1,592 validation samples.

**Naive RAG hurts hallucination detection.** Adding Wikipedia retrieval
reduces F1 from 0.92 to 0.72. Two failure modes identified:

1. Irrelevant retrieval -- topically adjacent but factually unrelated context
   causes the model to default to TRUTHFUL in absence of contradiction.
2. Vague hallucinations -- relevant context retrieved but the hallucinated
   answer is vague enough that no explicit contradiction appears in the summary.

**Implication for the field.** Retrieval quality and prompt design are critical
for RAG-augmented verification. Semantic retrieval precision -- returning the
specific passage containing the contradicting fact -- is a fundamentally harder
problem than standard RAG applications.

## Development Notes

### Training Pipeline -- Issues and Resolutions

**Initial approach:** Fine-tuned Llama 3.2 3B using `AutoModelForSequenceClassification`
with manual LoRA via `get_peft_model()` and the standard HuggingFace `Trainer`.

**Issues encountered on Windows / RTX 4080 Laptop GPU (12GB VRAM):**

1. **fp16 gradient scaling crash** -- Resolved by switching to `bf16=True` which
the RTX 4080 Laptop supports natively and is more stable for LLM fine-tuning.

2. **Training running at 0.02 it/s** -- Root cause: model silently fell back to CPU.
Always verify VRAM allocation before training with `torch.cuda.memory_allocated()`.

3. **Double LoRA application** -- Resolved by switching to `SFTTrainer` from `trl`
which handles LoRA internally via `peft_config`. Training runs at approximately
1.2 it/s on GPU as expected.

## Limitations

- Performs best on nuanced factual questions similar to TruthfulQA and HaluEval.
  Simple common knowledge questions may show inconsistent results.
- Confidence scores are binary (high/low) rather than continuous probabilities.
- English only -- not tested on non-English inputs.
- Docker container runs on CPU on Windows without NVIDIA Container Toolkit.

## Acknowledgements

Development was conducted with assistance from Claude (Anthropic) for guidance,
debugging, and code review.
