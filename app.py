import streamlit as st
from optimizer import optimize_prompt, run_optimized_prompt
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(page_title="Auto-Prompt Optimizer", page_icon="✨", layout="wide")

st.title("✨ Automated Prompt Optimizer")
st.markdown("### Turn 'lazy' prompts into Engineering-Grade prompts instantly.")

# Layout: Split screen for Before vs After
col1, col2 = st.columns(2)

with col1:
    st.subheader("1. Input (The Lazy Prompt)")
    user_input = st.text_area(
        "Enter a basic idea:", 
        placeholder="e.g., Write a blog post about coffee.",
        height=150
    )
    
    generate_btn = st.button("✨ Optimize & Run", type="primary")

if generate_btn and user_input:
    with st.spinner("🕵️‍♂️ Analyzing weakness... Rewriting with Chain-of-Thought..."):
        # 1. OPTIMIZE
        better_prompt = optimize_prompt(user_input)
    
    # Show the Optimized Prompt
    st.success("Optimization Complete!")
    
    with st.expander("👁️ View the 'Perfect' Prompt (The Meta-Prompt Output)", expanded=True):
        st.code(better_prompt, language="text")
        st.caption("Copy this to use in your own projects!")

    # 2. EXECUTE (Show the difference)
    with col2:
        st.subheader("2. Result (The Output)")
        with st.spinner("Running the new prompt..."):
            result = run_optimized_prompt(better_prompt)
            st.markdown(result)