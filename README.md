# ✨ Auto-Prompt Optimizer

A meta-prompting tool that takes your vague idea and turns it into a well-structured, optimized prompt — using LLMs to engineer better prompts automatically.

**Tech:** Python · Streamlit · LangChain · Groq (LLaMA 3.3 70B)

---

## What It Does

Most people write lazy prompts like *"Fix this code"* or *"Write a blog about AI"*. This tool acts as an AI prompt engineer — it interprets what you actually want, asks clarifying questions, and generates a structured prompt you can use in any LLM.

It generates **prompts, not solutions**. You take the output and run it wherever you want.

---

## How It Works

```
User Input → Interpretation → Clarification MCQs → Refined Understanding → Optimized Prompt
```

1. You describe your task in plain English (optionally upload files).
2. The system interprets your intent — identifies task type, objectives, expected output.
3. If the interpretation is off, it asks MCQ-style clarification questions and refines.
4. It generates an optimized prompt using techniques like Chain-of-Thought, role-based prompting, and the ARMS framework.
5. It also generates 4 prompt variations using different prompting techniques (CoT, Few-shot, Role-based, Spec-driven).

---

## Features

- **Prompt Quality Scoring** — Scores your prompt on Clarity, Specificity, Structure, and Completeness (0–100 each) using an LLM-as-Judge pattern. Gives you a letter grade and improvement suggestions.

- **Prompt Chain Builder** — Breaks complex tasks into a multi-step workflow (2–6 steps). Each step has a ready-to-use prompt where the output feeds into the next step via `{previous_output}`.

- **RAG-Enhanced Generation** — Upload reference docs (style guides, specs, etc.) and the system pulls relevant chunks to enrich the generated prompt with domain-specific context.

- **Injection Detection** — Two-pass security scan: regex pattern matching (17 patterns across 6 categories) + LLM contextual analysis to catch real threats and filter out false positives.

- **Interactive Editing** — Paste a specific part of the prompt to refine, or describe what you want changed in natural language.

- **Export** — Download as `.txt`, `.md`, or `.json` with all scores and analysis included.

---

## Setup

```bash
git clone https://github.com/<your-username>/prompt-optimizer.git
cd prompt-optimizer
pip install -r requirements.txt
```

Create a `.env` file:
```env
GROQ_API_KEY=gsk_your_key_here
```

Run:
```bash
streamlit run app.py
```

---

## Example

**Input:** "Write a tweet about AI"

**Output (a prompt, not a tweet):**
> Act as a Social Media Manager for a Tech Startup. Write a viral thread (max 280 chars) about the future of Agentic AI.
>
> Constraints: Use a punchy hook. Include 3 hashtags. Tone: Professional yet visionary.
>
> Steps: 1. Draft 3 hooks. 2. Pick the best. 3. Polish for readability.

---

## Project Structure

```
├── app.py              # Streamlit UI
├── optimizer.py        # Core LLM pipeline (interpretation, generation, scoring, RAG, injection)
├── requirements.txt
├── .env
└── README.md
```

---