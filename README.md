# ✨ Automated Prompt Optimizer

**Role:** AI Tools Engineer | **Tech:** Meta-Prompting, LangChain, Groq

> "Stop writing prompts from scratch. Let the AI write them for you."

## 🧠 The Concept
Most users write "lazy" prompts (e.g., *"Fix this code"*). This system uses a **Meta-Prompt**—a prompt that writes other prompts—to automatically convert a vague request into a highly structured, engineering-grade instruction set following the **CO-STAR framework** (Context, Objective, Style, Tone, Audience, Response).

## ⚙️ Architecture

1.  **Input:** User provides a raw, weak string.
2.  **The Meta-Agent:** An LLM acting as a "Senior Prompt Engineer" analyzes the input for:
    * Missing Context
    * Vague Objectives
    * Lack of Formatting
3.  **Transformation:** It rewrites the prompt adding **Chain-of-Thought (CoT)** reasoning and **Persona** adoption.
4.  **Execution:** The system runs the *new* prompt to demonstrate the superior output quality.



## 🛠️ Usage

1.  **Clone & Install**
    ```bash
    git clone <repo_url>
    cd prompt-optimizer
    pip install -r requirements.txt
    ```

2.  **Setup Keys**
    Create `.env`:
    ```env
    GROQ_API_KEY=gsk_...
    ```

3.  **Run Tool**
    ```bash
    streamlit run app.py
    ```

## 🧪 Example

* **User Input:** "Write a tweet about AI."
* **Optimized Output:**
    > "Act as a Social Media Manager for a Tech Startup. Write a viral thread (max 280 chars) about the future of Agentic AI.
    >
    > **Constraints:**
    > * Use a punchy, provocative hook.
    > * Include 3 relevant hashtags.
    > * Tone: Professional yet visionary.
    >
    > **Step-by-Step:**
    > 1. Draft 3 hooks.
    > 2. Select the best one.
    > 3. Polish for readability."