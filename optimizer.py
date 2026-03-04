import os
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# --- CONFIG ---
# We use Llama 3 70B for the "Brain" because optimization requires high reasoning.
LLM_MODEL = "llama3-70b-8192"

def get_llm():
    return ChatGroq(
        model=LLM_MODEL, 
        temperature=0.2, 
        api_key=os.getenv("GROQ_API_KEY")
    )

def optimize_prompt(original_prompt):
    """
    Takes a raw, weak prompt and transforms it into a professional engineering prompt
    using a 'Meta-Prompt' technique.
    """
    llm = get_llm()
    
    # SYSTEM PROMPT: The "Meta-Prompt"
    # This instructs the AI to act as a Senior Prompt Engineer.
    system_instruction = """
    You are a Senior Prompt Engineer and AI Optimization Expert.
    Your goal is to rewrite a given USER PROMPT into a "Perfect Prompt" 
    using the CO-STAR framework (Context, Objective, Style, Tone, Audience, Response).

    Follow these steps:
    1. ANALYZE: Identify weaknesses in the user's prompt (vague, missing constraints, no format).
    2. ENHANCE: Add a Persona (e.g., "Act as an expert...").
    3. STRUCTURE: Add Chain-of-Thought instructions (e.g., "Think step-by-step...").
    4. REFINE: Ensure the output format is strictly defined.

    OUTPUT ONLY THE OPTIMIZED PROMPT. DO NOT EXPLAIN YOUR CHANGES.
    """
    
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", system_instruction),
        ("user", "Here is the bad prompt: '{original_prompt}'\n\nRewrite it now:")
    ])
    
    chain = prompt_template | llm | StrOutputParser()
    
    return chain.invoke({"original_prompt": original_prompt})

def run_optimized_prompt(optimized_prompt):
    """
    Actually runs the new prompt to show the user the result.
    """
    llm = get_llm()
    return llm.invoke(optimized_prompt).content