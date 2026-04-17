# LLM Prompting & Evaluation — Structured Experimentation

This repository contains a structured, end-to-end experimentation of prompt engineering, evaluation design, and robustness analysis using Large Language Models under real-world constraints.

📄 Full Report: :contentReference[oaicite:0]{index=0}

---

## 🎯 Objective

The goal of this project was **not to maximise accuracy**, but to:

- Design prompting strategies
- Iterate and analyse behavior
- Build reliable evaluation pipelines
- Understand trade-offs in constrained environments

---

## ⚙️ Setup & Constraints

- Model: **Gemma 2B (4-bit quantized)**
- GPU: **RTX 4060 Ti (8GB VRAM)**
- Environment: Local + remote (Parsec)
- Sample size: **15 per task (intentional for fast iteration)**

Due to API limits, the project pivoted from:
- Gemini → Claude → Gemma (local fallback)

---


---

## 🧪 Tasks Overview

### 🔹 Task 1 — Zero-shot vs Few-shot
- Few-shot significantly improved **JSON compliance**
- Insight: structure > reasoning for small models

---

### 🔹 Task 2 — Direct vs CoT
- CoT alone ≠ improvement
- CoT + Few-shot → best performance
- Insight: **reasoning helps only after format is fixed**

---

### 🔹 Task 3 — Multi-objective Assistant
- Outputs: stars + key_point + business response
- Built **LLM-as-judge evaluation**
- Major learning: **evaluation design is critical**

---

### 🔹 Task 4 — Robustness & Domain Shift

#### Robustness:
- Negation hardest
- Noise causes parsing failures
- Structured prompts improve stability

#### Domain Shift:
- Yelp → IMDB shows clear performance drop
- Domain gap > perturbation impact

---

## 📊 Key Insights

- Structured prompts outperform complex reasoning
- Small models struggle with CoT unless constrained
- Evaluation pipelines can fail silently
- Domain shift is a harder problem than robustness

---

## ⚠️ Limitations

- Small sample size (15)
- Local 2B model constraints
- LLM-as-judge introduces bias
- Mixed-domain experiment affected by inference path bug

---

## 🚀 Reproducibility

Run notebooks directly:

```bash
jupyter notebook
