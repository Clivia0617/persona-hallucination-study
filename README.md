# Persona Injection, Hallucination, and Context Persistence in LLMs

Code and data for the ST5230 course project: an empirical study of how 
system-level persona prompts affect style and hallucination rate across 
three commercial LLMs (GPT-4o-mini, Claude 3 Haiku, Gemini 1.5 Flash).

## Overview

This repository contains the complete pipeline for:
- Generating 53,865 static-persona and 1,620 prompt-switching responses 
  across 9 persona conditions, 400 questions (TriviaQA, PopQA, MedQA), 
  and 3 commercial models.
- Judging responses with an LLM-as-a-judge pipeline (κ = 0.941 against 
  human annotation).
- Reproducing all seven pre-registered hypothesis tests (H1–H7).

## Repository Structure

```
.
├── prompts.py               # All persona prompts used in the study
├── config.py                # Model IDs, API settings, constants
├── api_client.py            # OpenRouter API wrapper
├── data_prep.py             # Question sampling from TriviaQA/PopQA/MedQA
├── experiment_rq1_rq2.py    # Static-persona experiment (RQ1/RQ2)
├── experiment_rq3.py        # In-session switching experiment (RQ3)
├── judge.py                 # LLM-as-a-judge pipeline
├── metrics.py               # HR, CS, AR, RHE, PPS computations
├── analysis.py              # Statistical tests for H1–H7
├── stage1_experiment.py     # Stage 1 runner script
├── stage2_analysis.py       # Stage 2 analysis script
├── data/                    # Sampled questions
└── results/                 # Aggregated CSVs of judged responses
```

## Setup

```bash
git clone https://github.com/Clivia0617/persona-hallucination-study.git
cd persona-hallucination-study
pip install -r requirements.txt
```

Create a `.env` file in the project root with your OpenRouter API key:

```
OPENROUTER_API_KEY=your_key_here
```

## Reproducing the Experiments

### Stage 1: Generation
```bash
python stage1_experiment.py
```
Output: judged responses in `results/`.

### Stage 2: Analysis
```bash
python stage2_analysis.py
```
Produces the tables and figures from the paper.

## Data

- `data/` — Sampled 400 questions from TriviaQA, PopQA, and MedQA 
  (produced by `data_prep.py`).
- `results/` — Aggregated CSVs of all 53,865 static and 1,620 
  switching responses, with LLM-as-a-judge verdicts attached 
  (~68 MB). These are the direct inputs to `analysis.py` and 
  reproduce all tables and figures in the paper without re-running 
  the generation pipeline.
- `logs/` — API call logs are not included in the repository 
  (regenerated automatically on re-run).

## Citation

If you use this code, please cite:

```
Du, Y. (2026). Persona Injection, Hallucination, and Context Persistence 
in Large Language Models: An Empirical Study. ST5230 Course Project, 
National University of Singapore.
```

## License

MIT License. See `LICENSE` for details.
