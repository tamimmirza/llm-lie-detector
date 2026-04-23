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