"""
app.py — Auto-Prompt Optimizer UI
===================================
Streamlit interface for the full prompt optimization pipeline.

Features:
    1. Prompt Quality Scoring (radar chart + grade)
    4. Copy-to-Clipboard & Export (.txt, .md, .json)
    9. Prompt Chain Builder (multi-step visual workflow)
   11. RAG-Enhanced Prompt Generation (reference documents)
   12. Prompt Injection Detection (regex + LLM scan)
"""

import json
import datetime
import streamlit as st
from optimizer import (
    node_generate_interpretation,
    node_ask_clarification_questions,
    node_refine_interpretation,
    node_generate_task_prompt,
    generate_related_ideas,
    generate_prerequisite_prompts,
    recommend_templates,
    generate_prompt_variations,
    get_clarification_for_prompt_part,
    refine_prompt_part,
    rewrite_prompt_from_description,
    # New feature imports
    score_prompt_quality,
    generate_prompt_chain,
    generate_rag_enhanced_prompt,
    full_injection_scan,
)
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(page_title="Auto-Prompt Optimizer", page_icon="✨", layout="wide")


# =====================================================================
# HELPER: Export prompt data
# =====================================================================

def build_export_data(prompt_text: str, interpretation: dict, scores: dict = None,
                      chain: list = None, injection: dict = None) -> dict:
    """Build a structured dict for JSON export."""
    return {
        "exported_at": datetime.datetime.now().isoformat(),
        "tool": "Auto-Prompt Optimizer",
        "task_type": interpretation.get("task_type", "unknown"),
        "interpretation": interpretation.get("interpretation", ""),
        "objectives": interpretation.get("objectives", []),
        "optimized_prompt": prompt_text,
        "quality_scores": scores,
        "prompt_chain": chain,
        "injection_scan": {
            "risk_level": injection.get("risk_level"),
            "risk_score": injection.get("risk_score"),
            "is_safe": injection.get("is_safe"),
            "summary": injection.get("summary"),
        } if injection else None,
    }


# =====================================================================
# HEADER
# =====================================================================

col1, col2 = st.columns([8, 2])
with col1:
    st.title("✨ Automated Prompt Optimizer")
    st.markdown("### Turn your idea into an Engineering-Grade prompt instantly.")
with col2:
    if st.session_state.get("stage") == "results":
        if st.button("🔄 Start Over", type="secondary", key="start_over_top"):
            keys_to_clear = [
                "stage", "user_input", "uploaded_file", "state_data",
                "clarification_answers", "suggestions", "variations",
                "current_variation_index", "current_edited_prompt",
                "last_optimized_prompt", "refinement_feedback",
                "part_clarification", "quality_scores", "prompt_chain",
                "injection_results", "reference_docs", "rag_prompt",
            ]
            for k in keys_to_clear:
                if k in st.session_state:
                    del st.session_state[k]
            st.session_state.stage = "input"
            st.rerun()


# =====================================================================
# SESSION STATE INIT
# =====================================================================

defaults = {
    "stage": "input",
    "user_input": "",
    "uploaded_file": None,
    "state_data": {},
    "clarification_answers": {},
    "suggestions": {},
    "variations": [],
    "current_variation_index": 0,
    # New feature states
    "quality_scores": None,
    "prompt_chain": None,
    "chain_outputs": None,
    "injection_results": None,
    "reference_docs": [],
    "rag_prompt": None,
}
for key, val in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = val


# =====================================================================
# STAGE 1: INPUT
# =====================================================================

if st.session_state.stage == "input":
    st.markdown("### Step 1: Describe Your Task")
    st.markdown("Tell us what you want, and optionally upload files to work with.")

    # --- File upload ---
    st.markdown("#### 📎 Optional: Upload Files")

    tab_work, tab_ref = st.tabs(["📄 Working File", "📚 Reference Documents (RAG)"])

    with tab_work:
        st.markdown("*Upload a file you want to modify or work with.*")
        uploaded_file = st.file_uploader(
            "Working file",
            type=["txt", "py", "js", "html", "css", "md", "json", "yaml", "yml",
                  "csv", "png", "jpg", "jpeg", "gif"],
            key="work_file",
        )
        if uploaded_file:
            st.success(f"📄 Uploaded: **{uploaded_file.name}**")

    with tab_ref:
        st.markdown("*Upload reference documents (style guides, specs, etc.) to enhance prompt generation with RAG.*")
        ref_files = st.file_uploader(
            "Reference documents",
            type=["txt", "md", "py", "js", "json", "yaml", "yml", "csv", "html", "css"],
            accept_multiple_files=True,
            key="ref_files",
        )
        if ref_files:
            st.success(f"📚 {len(ref_files)} reference document(s) uploaded")
            for rf in ref_files:
                st.caption(f"  • {rf.name}")

    st.markdown("---")
    st.markdown("#### ✏️ Describe Your Task")
    user_input = st.text_area(
        "What do you want to accomplish?",
        placeholder="e.g., 'Write a Python REST API with authentication', 'Create a marketing blog post about AI trends'",
        height=100,
    )

    col1, col2, col3 = st.columns([1, 1, 2])
    with col1:
        submit_btn = st.button("🚀 Process Request", type="primary", key="submit_request")
    with col2:
        if st.button("📝 Text Only", key="text_only"):
            uploaded_file = None
            submit_btn = True

    if submit_btn and user_input.strip():
        st.session_state.user_input = user_input
        st.session_state.uploaded_file = uploaded_file

        # Store reference docs
        if ref_files:
            st.session_state.reference_docs = [
                {"filename": rf.name, "content": rf.getvalue().decode("utf-8", errors="ignore")}
                for rf in ref_files
            ]
        else:
            st.session_state.reference_docs = []

        st.session_state.stage = "interpret"
        with st.spinner("🔍 Analyzing your request..."):
            state = node_generate_interpretation(user_input, uploaded_file)
            st.session_state.state_data = state
        st.rerun()


# =====================================================================
# STAGE 2: INTERPRET
# =====================================================================

elif st.session_state.stage == "interpret":
    st.markdown("### Step 2: Confirm Understanding")
    st.markdown(f"**Your request:** *{st.session_state.user_input}*")
    if st.session_state.uploaded_file:
        st.markdown(f"**📎 Working with:** *{st.session_state.uploaded_file.name}*")
    if st.session_state.reference_docs:
        st.markdown(f"**📚 Reference docs:** {len(st.session_state.reference_docs)} file(s)")

    st.markdown("---")
    interpretation = st.session_state.state_data["interpretation"]

    st.subheader("📌 Here's what I understand:")
    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown(f"**Task Type:** {interpretation.get('task_type', 'Unknown').upper()}")
        st.markdown(f"**Interpretation:** {interpretation.get('interpretation', 'N/A')}")
        if interpretation.get("objectives"):
            st.markdown("**Objectives:**")
            for obj in interpretation["objectives"]:
                st.markdown(f"- {obj}")
        st.markdown(f"**Expected Output:** {interpretation.get('key_outputs', 'N/A')}")
    with col2:
        st.info("Is this correct?")

    st.markdown("---")
    col1, col2, col3 = st.columns([1, 1, 2])
    with col1:
        if st.button("✅ Yes, That's Right!", type="primary", key="confirm_yes"):
            st.session_state.stage = "results"
            with st.spinner("📝 Generating your optimized prompt..."):
                st.session_state.state_data = node_generate_task_prompt(st.session_state.state_data)

                # If reference docs exist, also generate RAG-enhanced version
                if st.session_state.reference_docs:
                    with st.spinner("📚 Enhancing with reference documents..."):
                        rag_prompt = generate_rag_enhanced_prompt(
                            st.session_state.user_input,
                            st.session_state.reference_docs,
                            st.session_state.state_data["interpretation"],
                        )
                        st.session_state.rag_prompt = rag_prompt
            st.rerun()
    with col2:
        if st.button("❌ No, Not Quite", key="confirm_no"):
            st.session_state.stage = "clarify"
            with st.spinner("❓ Generating clarification questions..."):
                st.session_state.state_data = node_ask_clarification_questions(st.session_state.state_data)
            st.rerun()
    with col3:
        st.caption("Answer clarification questions to refine understanding")


# =====================================================================
# STAGE 3: CLARIFY
# =====================================================================

elif st.session_state.stage == "clarify":
    st.markdown("### Step 2b: Let's Refine My Understanding")
    st.markdown(f"**Your idea:** *{st.session_state.user_input}*")
    st.markdown("---")

    clarification_questions = st.session_state.state_data.get("clarification_questions", [])
    if clarification_questions:
        col1, col2 = st.columns(2)
        cols = [col1, col2]
        local_answers = {}

        for idx, q in enumerate(clarification_questions):
            with cols[idx % 2]:
                st.markdown(f"#### Q{q['id']}: {q['question']}")
                option_keys = list(q["options"].keys())
                selected = st.radio(
                    f"Answer for Q{q['id']}",
                    options=option_keys,
                    format_func=lambda x, q=q: q["options"][x],
                    key=f"clarify_q_{q['id']}",
                )
                if selected == "E":
                    custom = st.text_input("Your answer:", key=f"custom_q_{q['id']}")
                    local_answers[f"Q{q['id']}"] = custom or "Not specified"
                else:
                    local_answers[f"Q{q['id']}"] = q["options"][selected]
                st.markdown("---")

        st.session_state.clarification_answers = local_answers
        if st.button("✅ Continue", type="primary", key="submit_clarification"):
            with st.spinner("🔄 Refining understanding..."):
                st.session_state.state_data = node_refine_interpretation(
                    st.session_state.state_data, list(local_answers.values())
                )
            st.session_state.stage = "interpret"
            st.rerun()


# =====================================================================
# STAGE 4: RESULTS
# =====================================================================

elif st.session_state.stage == "results":
    interpretation = st.session_state.state_data["interpretation"]
    task_type = interpretation.get("task_type", "general")

    # --- Understanding Summary ---
    with st.expander("📌 Understanding Summary", expanded=False):
        st.markdown(f"**Task Type:** {interpretation.get('task_type', 'Unknown')}")
        st.markdown(f"**Interpretation:** {interpretation.get('interpretation', 'N/A')}")
        if interpretation.get("objectives"):
            for obj in interpretation["objectives"]:
                st.markdown(f"- {obj}")
        st.markdown(f"**Expected Output:** {interpretation.get('key_outputs', 'N/A')}")

    # --- Generate variations if needed ---
    if not st.session_state.variations:
        with st.spinner("🔄 Generating prompt variations..."):
            ctx = {"file_context": interpretation.get("file_context", ""),
                   "objectives": interpretation.get("objectives", [])}
            st.session_state.variations = generate_prompt_variations(
                st.session_state.user_input, task_type, ctx
            )

    # --- Prompt state management ---
    optimized_prompt = st.session_state.state_data.get("optimized_prompt", "")
    if "current_edited_prompt" not in st.session_state:
        st.session_state.current_edited_prompt = optimized_prompt
        st.session_state.last_optimized_prompt = optimized_prompt
    elif st.session_state.get("last_optimized_prompt") != optimized_prompt:
        st.session_state.current_edited_prompt = optimized_prompt
        st.session_state.last_optimized_prompt = optimized_prompt

    display_prompt = st.session_state.current_edited_prompt
    variations = st.session_state.variations or []

    # Build prompt list (original + RAG + variations)
    all_prompts = [{"name": "🎯 Original Optimized Prompt", "prompt": display_prompt, "type": "original"}]
    if st.session_state.rag_prompt:
        all_prompts.append({
            "name": "📚 RAG-Enhanced Prompt",
            "prompt": st.session_state.rag_prompt,
            "description": "Enhanced with context from your reference documents",
            "type": "rag",
        })
    for i, var in enumerate(variations, 1):
        all_prompts.append({
            "name": f"📋 Variation {i}: {var.get('name', 'Untitled')}",
            "prompt": var.get("prompt", ""),
            "description": var.get("description", ""),
            "type": "variation",
        })

    if st.session_state.current_variation_index >= len(all_prompts):
        st.session_state.current_variation_index = 0
    current_prompt_data = all_prompts[st.session_state.current_variation_index]

    # ==================================================================
    # MAIN PROMPT DISPLAY
    # ==================================================================

    st.markdown("### 📝 The Engineered Prompt")

    if "refinement_feedback" in st.session_state:
        st.success("✅ **Prompt Updated!** Your refinements have been applied.")

    # Navigation
    col_nav1, col_nav2, col_nav3 = st.columns([1, 3, 1])
    with col_nav1:
        if st.button("⬅️ Previous", key="prev_var"):
            st.session_state.current_variation_index = (
                (st.session_state.current_variation_index - 1) % len(all_prompts)
            )
            st.rerun()
    with col_nav2:
        st.markdown(
            f"<div style='text-align:center;padding:10px;'>"
            f"<b>{current_prompt_data['name']}</b><br>"
            f"<small>{st.session_state.current_variation_index + 1} of {len(all_prompts)}</small>"
            f"</div>",
            unsafe_allow_html=True,
        )
    with col_nav3:
        if st.button("Next ➡️", key="next_var"):
            st.session_state.current_variation_index = (
                (st.session_state.current_variation_index + 1) % len(all_prompts)
            )
            st.rerun()

    st.code(current_prompt_data["prompt"], language="text")
    if current_prompt_data.get("description"):
        st.caption(f"💡 {current_prompt_data['description']}")

    # ==================================================================
    # FEATURE 4: COPY & EXPORT
    # ==================================================================

    st.markdown("---")
    st.markdown("### 📋 Copy & Export")

    exp_col1, exp_col2, exp_col3, exp_col4 = st.columns(4)

    current_prompt_text = current_prompt_data["prompt"]

    with exp_col1:
        # Streamlit doesn't have native clipboard, so we use a download trick
        st.download_button(
            label="📋 Copy as .txt",
            data=current_prompt_text,
            file_name="optimized_prompt.txt",
            mime="text/plain",
        )

    with exp_col2:
        md_content = (
            f"# Optimized Prompt\n\n"
            f"**Task Type:** {task_type}\n\n"
            f"**Generated:** {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n"
            f"---\n\n"
            f"## Prompt\n\n```\n{current_prompt_text}\n```\n"
        )
        if st.session_state.quality_scores:
            qs = st.session_state.quality_scores
            md_content += (
                f"\n## Quality Scores\n\n"
                f"- **Overall:** {qs['overall_score']}/100 ({qs['grade']})\n"
                f"- Clarity: {qs['clarity']}/100\n"
                f"- Specificity: {qs['specificity']}/100\n"
                f"- Structure: {qs['structure']}/100\n"
                f"- Completeness: {qs['completeness']}/100\n"
            )
        st.download_button(
            label="📝 Export .md",
            data=md_content,
            file_name="optimized_prompt.md",
            mime="text/markdown",
        )

    with exp_col3:
        json_data = build_export_data(
            current_prompt_text, interpretation,
            scores=st.session_state.quality_scores,
            chain=[s for s in (st.session_state.prompt_chain or [])],
            injection=st.session_state.injection_results,
        )
        st.download_button(
            label="📦 Export .json",
            data=json.dumps(json_data, indent=2),
            file_name="optimized_prompt.json",
            mime="application/json",
        )

    with exp_col4:
        # Copy to clipboard via HTML/JS hack
        st.markdown(
            f"""
            <button onclick="navigator.clipboard.writeText(`{current_prompt_text[:500].replace('`', '')}`).then(
                () => this.innerText = '✅ Copied!',
                () => this.innerText = '❌ Failed'
            )" style="padding:8px 16px;border-radius:4px;border:1px solid #ccc;cursor:pointer;">
            📋 Copy to Clipboard
            </button>
            """,
            unsafe_allow_html=True,
        )

    # ==================================================================
    # FEATURE TABS
    # ==================================================================

    st.markdown("---")

    tab_score, tab_edit, tab_chain, tab_rag, tab_inject = st.tabs([
        "📊 Quality Score",
        "✏️ Edit Prompt",
        "🔗 Prompt Chain",
        "📚 RAG Enhancement",
        "🛡️ Injection Detection",
    ])

    # ------------------------------------------------------------------
    # TAB 1: QUALITY SCORING
    # ------------------------------------------------------------------
    with tab_score:
        st.markdown("### 📊 Prompt Quality Score")
        st.markdown("Evaluate your prompt using LLM-as-judge on 4 dimensions.")

        if st.button("🎯 Score This Prompt", type="primary", key="score_btn"):
            with st.spinner("🔍 Evaluating prompt quality..."):
                scores = score_prompt_quality(current_prompt_text, task_type)
                st.session_state.quality_scores = scores

        if st.session_state.quality_scores:
            qs = st.session_state.quality_scores

            # Grade display
            grade = qs.get("grade", "?")
            overall = qs.get("overall_score", 0)
            grade_colors = {
                "A+": "#00C851", "A": "#00C851", "B+": "#33b5e5", "B": "#33b5e5",
                "C+": "#ffbb33", "C": "#ffbb33", "D": "#ff4444", "F": "#CC0000",
            }
            color = grade_colors.get(grade, "#666")

            st.markdown(
                f"<div style='text-align:center;padding:20px;'>"
                f"<span style='font-size:72px;font-weight:bold;color:{color};'>{grade}</span>"
                f"<br><span style='font-size:24px;color:#888;'>{overall}/100</span>"
                f"<br><em>{qs.get('one_line_verdict', '')}</em>"
                f"</div>",
                unsafe_allow_html=True,
            )

            # Score bars
            st.markdown("#### Dimension Scores")
            dimensions = [
                ("Clarity", qs.get("clarity", 0)),
                ("Specificity", qs.get("specificity", 0)),
                ("Structure", qs.get("structure", 0)),
                ("Completeness", qs.get("completeness", 0)),
            ]
            for name, score in dimensions:
                bar_color = "#00C851" if score >= 80 else "#ffbb33" if score >= 60 else "#ff4444"
                st.markdown(
                    f"**{name}:** {score}/100"
                    f"<div style='background:#eee;border-radius:8px;height:18px;width:100%;'>"
                    f"<div style='background:{bar_color};border-radius:8px;height:18px;width:{score}%;'></div>"
                    f"</div>",
                    unsafe_allow_html=True,
                )
                st.markdown("")

            # Strengths & Improvements
            col_s, col_i = st.columns(2)
            with col_s:
                st.markdown("#### ✅ Strengths")
                for s in qs.get("top_strengths", []):
                    st.markdown(f"- {s}")
            with col_i:
                st.markdown("#### 🔧 Improvements")
                for imp in qs.get("improvements", []):
                    st.markdown(f"- {imp}")

    # ------------------------------------------------------------------
    # TAB 2: EDIT PROMPT
    # ------------------------------------------------------------------
    with tab_edit:
        st.markdown("### ✏️ Edit Your Prompt")

        edited_prompt = st.text_area(
            "Edit the prompt below:",
            value=st.session_state.current_edited_prompt,
            height=200,
        )
        if edited_prompt != st.session_state.current_edited_prompt:
            st.session_state.current_edited_prompt = edited_prompt

        st.markdown("---")
        st.markdown("**🎯 Refine Specific Part**")

        refine_mode = st.radio(
            "How to specify?",
            ["📋 Paste from Prompt", "✍️ Type/Describe Directly"],
            horizontal=True,
            key="refine_mode",
        )
        is_describe_mode = refine_mode == "✍️ Type/Describe Directly"

        specific_part = st.text_area(
            "Paste text to refine:" if not is_describe_mode else "Describe what to change:",
            height=100,
            key="specific_part",
            placeholder=(
                "Copy and paste exact text..." if not is_describe_mode
                else "e.g., Change the language to Python"
            ),
        )

        if st.button("❓ Get Clarification Questions", key="clarify_part") and specific_part.strip():
            with st.spinner("🤔 Generating questions..."):
                result = get_clarification_for_prompt_part(specific_part, interpretation)
                if result and "questions" in result:
                    st.session_state.part_clarification = result
                    st.rerun()

        if "part_clarification" in st.session_state:
            cdata = st.session_state.part_clarification
            if "questions" in cdata:
                for i, question in enumerate(cdata["questions"], 1):
                    st.markdown(f"**{i}. {question['question']}**")
                    st.radio(
                        f"Answer {i}:",
                        options=list(question["options"].keys()),
                        format_func=lambda x, q=question: f"{x}: {q['options'][x]}",
                        key=f"part_q_{i}",
                        label_visibility="collapsed",
                    )

            comments = st.text_area("💬 Additional comments:", height=80, key="part_comments")

            col_a, col_c = st.columns(2)
            with col_a:
                if st.button("✨ Apply Refinements", type="primary", key="apply_refine"):
                    with st.spinner("🔄 Refining..."):
                        part = specific_part.strip()
                        if not part:
                            st.error("Please provide text to refine.")
                        else:
                            current = st.session_state.current_edited_prompt
                            if is_describe_mode:
                                updated = rewrite_prompt_from_description(
                                    current, part, comments, interpretation
                                )
                            else:
                                refined = refine_prompt_part(part, comments, interpretation)
                                if part in current:
                                    updated = current.replace(part, refined)
                                else:
                                    updated = rewrite_prompt_from_description(
                                        current,
                                        f"Replace: '{part}' with improved version. Feedback: {comments}",
                                        comments, interpretation,
                                    )
                            st.session_state.current_edited_prompt = updated
                            st.session_state.state_data["optimized_prompt"] = updated
                            st.session_state.last_optimized_prompt = updated
                            st.session_state.current_variation_index = 0
                            st.session_state.refinement_feedback = {"comments": comments}
                            if "part_clarification" in st.session_state:
                                del st.session_state.part_clarification
                            # Reset quality scores since prompt changed
                            st.session_state.quality_scores = None
                            st.rerun()
            with col_c:
                if st.button("❌ Cancel", key="cancel_refine"):
                    if "part_clarification" in st.session_state:
                        del st.session_state.part_clarification
                    st.rerun()

        st.markdown("---")
        if st.button("💾 Save Edited Prompt", type="primary"):
            st.session_state.state_data["optimized_prompt"] = st.session_state.current_edited_prompt
            st.session_state.last_optimized_prompt = st.session_state.current_edited_prompt
            st.session_state.quality_scores = None  # Reset scores for new version
            if "refinement_feedback" in st.session_state:
                del st.session_state.refinement_feedback
            st.success("✅ Saved!")
            st.rerun()

    # ------------------------------------------------------------------
    # TAB 3: PROMPT CHAIN BUILDER
    # ------------------------------------------------------------------
    with tab_chain:
        st.markdown("### 🔗 Prompt Chain Builder")
        st.markdown(
            "Break your task into a multi-step **prompt workflow**. Each step generates "
            "a ready-to-use prompt whose output is designed to feed into the next step. "
            "Copy each prompt and run them sequentially in any LLM of your choice."
        )

        chain_col1, chain_col2 = st.columns([2, 1])
        with chain_col1:
            num_steps = st.slider("Number of steps:", min_value=2, max_value=6, value=4, key="chain_steps")
        with chain_col2:
            if st.button("⚡ Generate Chain", type="primary", key="gen_chain"):
                with st.spinner("🔗 Designing prompt chain..."):
                    chain = generate_prompt_chain(st.session_state.user_input, task_type, num_steps)
                    st.session_state.prompt_chain = chain

        if st.session_state.prompt_chain:
            chain = st.session_state.prompt_chain

            # Visual chain flow
            st.markdown("#### Chain Overview")
            flow_items = [f"**{s['step_number']}. {s['name']}**" for s in chain]
            st.markdown(" → ".join(flow_items))
            st.markdown("---")

            st.info(
                "💡 **How to use this chain:** Copy each prompt in order. "
                "Run Step 1 in your preferred LLM, then paste its output into "
                "the `{previous_output}` placeholder in Step 2, and so on."
            )

            # Step cards — prompts only, no execution
            for step in chain:
                step_num = step["step_number"]
                depends = step.get("depends_on")

                with st.expander(f"📝 Step {step_num}: {step['name']}", expanded=(step_num == 1)):
                    st.markdown(f"**Purpose:** {step['description']}")
                    st.markdown(f"**Expected Output:** {step['expected_output']}")
                    if depends:
                        st.caption(f"⬆️ Use output from Step {depends} as `{{previous_output}}`")

                    st.code(step["prompt_template"], language="text")

                    # Per-step copy/download
                    st.download_button(
                        f"📋 Download Step {step_num} Prompt",
                        data=step["prompt_template"],
                        file_name=f"chain_step_{step_num}_{step['name'].lower().replace(' ', '_')}.txt",
                        mime="text/plain",
                        key=f"dl_step_{step_num}",
                    )

            # Export full chain
            st.markdown("---")
            chain_export = {
                "task": st.session_state.user_input,
                "task_type": task_type,
                "total_steps": len(chain),
                "usage_instructions": (
                    "Run each step sequentially. Paste the output of each step "
                    "into the {previous_output} placeholder of the next step."
                ),
                "steps": chain,
            }
            st.download_button(
                "📦 Export Full Chain (.json)",
                data=json.dumps(chain_export, indent=2),
                file_name="prompt_chain.json",
                mime="application/json",
                key="export_chain",
            )

            # Also export as a single combined markdown doc
            md_chain = f"# Prompt Chain: {st.session_state.user_input}\n\n"
            md_chain += f"**Task Type:** {task_type}\n\n"
            md_chain += f"**Instructions:** Run each step sequentially. Paste the output of each step into the `{{previous_output}}` placeholder of the next step.\n\n---\n\n"
            for step in chain:
                md_chain += f"## Step {step['step_number']}: {step['name']}\n\n"
                md_chain += f"**Purpose:** {step['description']}\n\n"
                md_chain += f"**Expected Output:** {step['expected_output']}\n\n"
                if step.get("depends_on"):
                    md_chain += f"*Uses output from Step {step['depends_on']}*\n\n"
                md_chain += f"### Prompt\n\n```\n{step['prompt_template']}\n```\n\n---\n\n"

            st.download_button(
                "📝 Export Full Chain (.md)",
                data=md_chain,
                file_name="prompt_chain.md",
                mime="text/markdown",
                key="export_chain_md",
            )

    # ------------------------------------------------------------------
    # TAB 4: RAG ENHANCEMENT
    # ------------------------------------------------------------------
    with tab_rag:
        st.markdown("### 📚 RAG-Enhanced Prompt Generation")
        st.markdown(
            "Upload reference documents to enrich your prompt with domain-specific "
            "context, terminology, and guidelines."
        )

        if st.session_state.rag_prompt:
            st.success("✅ RAG-enhanced prompt was generated using your reference documents!")
            st.markdown("**Navigate to the '📚 RAG-Enhanced Prompt' tab above to view it.**")

            st.markdown("#### Reference Documents Used:")
            for doc in st.session_state.reference_docs:
                with st.expander(f"📄 {doc['filename']}"):
                    preview = doc["content"][:500] + ("..." if len(doc["content"]) > 500 else "")
                    st.code(preview, language="text")

            st.markdown("---")
            if st.button("🔄 Regenerate RAG Prompt", key="regen_rag"):
                with st.spinner("📚 Regenerating with reference context..."):
                    rag = generate_rag_enhanced_prompt(
                        st.session_state.user_input,
                        st.session_state.reference_docs,
                        interpretation,
                    )
                    st.session_state.rag_prompt = rag
                    st.rerun()
        else:
            if st.session_state.reference_docs:
                st.info("Reference documents are loaded. Click below to generate a RAG-enhanced prompt.")
                if st.button("📚 Generate RAG-Enhanced Prompt", type="primary", key="gen_rag"):
                    with st.spinner("📚 Generating with reference context..."):
                        rag = generate_rag_enhanced_prompt(
                            st.session_state.user_input,
                            st.session_state.reference_docs,
                            interpretation,
                        )
                        st.session_state.rag_prompt = rag
                        st.rerun()
            else:
                st.warning(
                    "No reference documents uploaded. Go back to Step 1 and upload "
                    "reference files in the '📚 Reference Documents (RAG)' tab to use this feature."
                )

            # Allow uploading references even from results page
            st.markdown("---")
            st.markdown("#### Upload References Now")
            new_refs = st.file_uploader(
                "Add reference documents:",
                type=["txt", "md", "py", "js", "json", "yaml", "csv"],
                accept_multiple_files=True,
                key="results_ref_upload",
            )
            if new_refs:
                st.session_state.reference_docs = [
                    {"filename": rf.name, "content": rf.getvalue().decode("utf-8", errors="ignore")}
                    for rf in new_refs
                ]
                st.success(f"📚 {len(new_refs)} reference document(s) loaded!")
                st.rerun()

    # ------------------------------------------------------------------
    # TAB 5: INJECTION DETECTION
    # ------------------------------------------------------------------
    with tab_inject:
        st.markdown("### 🛡️ Prompt Injection Detection")
        st.markdown(
            "Scan your prompt for potential injection vulnerabilities using a "
            "two-pass system: regex pattern matching followed by LLM contextual analysis."
        )

        scan_target = st.radio(
            "What to scan:",
            ["Current Prompt", "Custom Text"],
            horizontal=True,
            key="scan_target",
        )

        if scan_target == "Custom Text":
            scan_text = st.text_area("Enter text to scan:", height=150, key="custom_scan_text")
        else:
            scan_text = current_prompt_text

        if st.button("🔍 Run Security Scan", type="primary", key="run_scan"):
            if scan_text.strip():
                with st.spinner("🛡️ Scanning for injection patterns..."):
                    results = full_injection_scan(scan_text)
                    st.session_state.injection_results = results
            else:
                st.error("No text to scan.")

        if st.session_state.injection_results:
            res = st.session_state.injection_results
            risk = res.get("risk_level", "unknown")
            risk_score = res.get("risk_score", -1)

            # Risk level display
            risk_colors = {
                "safe": "#00C851", "low": "#33b5e5", "medium": "#ffbb33",
                "high": "#ff4444", "critical": "#CC0000",
            }
            risk_color = risk_colors.get(risk, "#888")
            risk_emoji = {"safe": "✅", "low": "🟢", "medium": "🟡", "high": "🔴", "critical": "🚨"}.get(risk, "❓")

            st.markdown(
                f"<div style='text-align:center;padding:20px;border:2px solid {risk_color};border-radius:10px;'>"
                f"<span style='font-size:48px;'>{risk_emoji}</span>"
                f"<br><span style='font-size:28px;font-weight:bold;color:{risk_color};'>{risk.upper()}</span>"
                f"<br><span style='color:#888;'>Risk Score: {risk_score}/100</span>"
                f"<br><em>{res.get('summary', '')}</em>"
                f"</div>",
                unsafe_allow_html=True,
            )

            # Regex detections
            regex_count = res.get("regex_detection_count", 0)
            if regex_count > 0:
                st.markdown(f"#### 🔎 Pattern Matches ({regex_count})")
                for d in res.get("regex_detections", []):
                    sev_color = {"high": "🔴", "medium": "🟡", "low": "🟢"}.get(d["severity"], "⚪")
                    st.markdown(
                        f"{sev_color} **{d['category']}** — {d['description']}\n"
                        f"  Matched: `{d['matched_text']}`"
                    )

            # LLM findings
            llm_a = res.get("llm_analysis", {})
            findings = llm_a.get("findings", [])
            if findings:
                st.markdown("#### 🤖 LLM Analysis Findings")
                for f in findings:
                    fp = " *(false positive)*" if f.get("is_false_positive") else ""
                    st.markdown(f"- **{f.get('type', 'unknown')}**: {f.get('description', '')}{fp}")

            # False positives
            fps = llm_a.get("false_positives", [])
            if fps:
                st.markdown("#### ✅ False Positives (safe)")
                for fp in fps:
                    st.markdown(f"- {fp}")

            # Recommendations
            recs = res.get("recommendations", [])
            if recs:
                st.markdown("#### 💡 Recommendations")
                for r in recs:
                    st.markdown(f"- {r}")

            # Safe alternative
            safe_alt = res.get("safe_alternative", "")
            if safe_alt and not res.get("is_safe", True):
                st.markdown("#### 🔧 Suggested Safe Version")
                st.code(safe_alt, language="text")
                if st.button("✅ Apply Safe Version", key="apply_safe"):
                    st.session_state.current_edited_prompt = safe_alt
                    st.session_state.state_data["optimized_prompt"] = safe_alt
                    st.session_state.last_optimized_prompt = safe_alt
                    st.session_state.current_variation_index = 0
                    st.session_state.injection_results = None
                    st.session_state.quality_scores = None
                    st.rerun()

    st.markdown("---")