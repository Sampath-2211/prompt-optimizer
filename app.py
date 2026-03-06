"""
app.py — PromptLab UI
======================
Streamlit interface for the prompt optimization pipeline.
Clean version: Input → Interpret → Clarify → Results with variations.

Author: Sampath Krishna Tekumalla
"""

import json
import datetime
import streamlit as st
from optimizer import (
    node_generate_interpretation,
    node_ask_clarification_questions,
    node_refine_interpretation,
    node_generate_task_prompt,
    generate_prompt_variations,
    score_prompt_quality,
)
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(page_title="PromptLab", page_icon="🔬", layout="wide")


# =====================================================================
# HEADER
# =====================================================================

col1, col2 = st.columns([8, 2])
with col1:
    st.title("🔬 PromptLab")
    st.markdown("### Turn your idea into an Engineering-Grade prompt instantly.")
with col2:
    if st.session_state.get("stage") == "results":
        if st.button("🔄 Start Over", type="secondary", key="start_over_top"):
            for k in list(st.session_state.keys()):
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
    "variations": [],
    "current_variation_index": 0,
    "quality_scores": None,
}
for key, val in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = val


# =====================================================================
# STAGE 1: INPUT
# =====================================================================

if st.session_state.stage == "input":
    st.markdown("### Step 1: Describe Your Task")

    st.markdown("#### 📎 Optional: Upload a file to work with")
    uploaded_file = st.file_uploader(
        "Working file",
        type=["txt", "py", "js", "html", "css", "md", "json", "yaml", "yml",
              "csv", "png", "jpg", "jpeg", "gif"],
        key="work_file",
    )
    if uploaded_file:
        st.success(f"📄 Uploaded: **{uploaded_file.name}**")

    st.markdown("---")
    st.markdown("#### ✏️ Describe Your Task")
    user_input = st.text_area(
        "What do you want to accomplish?",
        placeholder="e.g., 'Write a Python REST API with authentication', 'Create a marketing blog post about AI trends'",
        height=120,
    )

    if st.button("🚀 Generate Prompt", type="primary", key="submit_request"):
        if user_input.strip():
            st.session_state.user_input = user_input
            st.session_state.uploaded_file = uploaded_file
            st.session_state.stage = "interpret"
            with st.spinner("🔍 Analyzing your request..."):
                state = node_generate_interpretation(user_input, uploaded_file)
                st.session_state.state_data = state
            st.rerun()
        else:
            st.error("Please describe your task first.")


# =====================================================================
# STAGE 2: INTERPRET
# =====================================================================

elif st.session_state.stage == "interpret":
    st.markdown("### Step 2: Confirm Understanding")
    st.markdown(f"**Your request:** *{st.session_state.user_input}*")

    st.markdown("---")
    interpretation = st.session_state.state_data["interpretation"]

    st.subheader("📌 Here's what I understand:")
    st.markdown(f"**Task Type:** {interpretation.get('task_type', 'Unknown').upper()}")
    st.markdown(f"**Interpretation:** {interpretation.get('interpretation', 'N/A')}")
    if interpretation.get("objectives"):
        st.markdown("**Objectives:**")
        for obj in interpretation["objectives"]:
            st.markdown(f"- {obj}")
    st.markdown(f"**Expected Output:** {interpretation.get('key_outputs', 'N/A')}")

    st.markdown("---")
    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("✅ Yes, That's Right!", type="primary", key="confirm_yes"):
            st.session_state.stage = "results"
            with st.spinner("📝 Generating your optimized prompt..."):
                st.session_state.state_data = node_generate_task_prompt(st.session_state.state_data)
            st.rerun()
    with col2:
        if st.button("❌ No, Not Quite", key="confirm_no"):
            st.session_state.stage = "clarify"
            with st.spinner("❓ Generating clarification questions..."):
                st.session_state.state_data = node_ask_clarification_questions(st.session_state.state_data)
            st.rerun()

    # Rephrase option
    st.markdown("---")
    rephrase_toggle = st.toggle(
        "I want to rephrase my idea entirely",
        value=False,
        key="interpret_rephrase_toggle",
    )
    if rephrase_toggle:
        new_idea = st.text_area(
            "Describe your idea again:",
            value=st.session_state.user_input,
            height=150,
            key="interpret_rephrase_text",
        )
        if st.button("🚀 Re-analyze This", type="primary", key="interpret_rephrase_btn"):
            if new_idea.strip():
                with st.spinner("🔍 Re-analyzing your request..."):
                    st.session_state.user_input = new_idea.strip()
                    state = node_generate_interpretation(new_idea.strip(), st.session_state.uploaded_file)
                    st.session_state.state_data = state
                st.rerun()


# =====================================================================
# STAGE 3: CLARIFY
# =====================================================================

elif st.session_state.stage == "clarify":
    st.markdown("### Step 2b: Let's Refine My Understanding")
    st.markdown(f"**Your idea:** *{st.session_state.user_input}*")
    st.markdown("---")

    clarify_mode = st.toggle(
        "I want to describe my idea differently instead",
        value=False,
        key="clarify_mode_toggle",
    )

    local_answers = {}

    if not clarify_mode:
        st.markdown("#### Answer the questions below, then add any extra comments.")

        clarification_questions = st.session_state.state_data.get("clarification_questions", [])
        if clarification_questions:
            col1, col2 = st.columns(2)
            cols = [col1, col2]

            for idx, q in enumerate(clarification_questions):
                if not isinstance(q, dict):
                    continue
                qid = q.get('id', idx + 1)
                with cols[idx % 2]:
                    st.markdown(f"#### Q{qid}: {q.get('question', 'Question')}")
                    raw_options = q.get("options", {})

                    if isinstance(raw_options, dict):
                        option_map = raw_options
                    elif isinstance(raw_options, list):
                        option_map = {chr(65 + i): str(opt) for i, opt in enumerate(raw_options)}
                    else:
                        option_map = {}

                    if option_map:
                        option_keys = list(option_map.keys())
                        selected = st.radio(
                            f"Answer for Q{qid}",
                            options=option_keys,
                            format_func=lambda x, om=option_map: om.get(x, x),
                            key=f"clarify_q_{qid}",
                        )
                        local_answers[f"Q{qid}"] = option_map.get(selected, selected)

                    custom = st.text_input(
                        "Or type your own answer:",
                        key=f"custom_q_{qid}",
                        placeholder="Type here to override the selection above...",
                    )
                    if custom.strip():
                        local_answers[f"Q{qid}"] = custom.strip()

        st.markdown("---")
        extra_context = st.text_area(
            "💬 Anything else you want to add?",
            placeholder="Add more details, context, or corrections...",
            height=100,
            key="clarify_extra_context",
        )
        if extra_context.strip():
            local_answers["extra_context"] = extra_context.strip()

        st.session_state.clarification_answers = local_answers
        if st.button("✅ Continue", type="primary", key="submit_clarification"):
            with st.spinner("🔄 Refining understanding..."):
                st.session_state.state_data = node_refine_interpretation(
                    st.session_state.state_data, list(local_answers.values())
                )
            st.session_state.stage = "interpret"
            st.rerun()

    else:
        st.markdown("#### Tell me what you actually want — in your own words.")
        new_description = st.text_area(
            "Describe your idea:",
            placeholder="e.g., 'I actually want a Python CLI tool that...'",
            height=200,
            key="clarify_new_idea",
        )
        if st.button("🚀 Use This Instead", type="primary", key="submit_new_idea"):
            if new_description.strip():
                with st.spinner("🔍 Re-analyzing your request..."):
                    st.session_state.user_input = new_description.strip()
                    state = node_generate_interpretation(new_description.strip(), st.session_state.uploaded_file)
                    st.session_state.state_data = state
                st.session_state.stage = "interpret"
                st.rerun()
            else:
                st.error("Please describe your idea first.")


# =====================================================================
# STAGE 4: RESULTS
# =====================================================================

elif st.session_state.stage == "results":
    interpretation = st.session_state.state_data["interpretation"]
    task_type = interpretation.get("task_type", "general")

    with st.expander("📌 Understanding Summary", expanded=False):
        st.markdown(f"**Task Type:** {interpretation.get('task_type', 'Unknown')}")
        st.markdown(f"**Interpretation:** {interpretation.get('interpretation', 'N/A')}")
        if interpretation.get("objectives"):
            for obj in interpretation["objectives"]:
                st.markdown(f"- {obj}")

    # --- Generate variations if needed ---
    optimized_prompt = st.session_state.state_data.get("optimized_prompt", "")
    variations = st.session_state.variations or []

    # --- Generate variations if needed (uses optimized_prompt as reference) ---
    if not variations:
        with st.spinner("🔄 Generating prompt variations..."):
            ctx = {"file_context": interpretation.get("file_context", ""),
                   "objectives": interpretation.get("objectives", [])}
            st.session_state.variations = generate_prompt_variations(
                st.session_state.user_input, task_type, ctx,
                optimized_prompt=optimized_prompt,
            )
            variations = st.session_state.variations

    # Build prompt list
    all_prompts = [{"name": "🎯 Original Optimized Prompt", "prompt": optimized_prompt, "description": "Main prompt generated using ARMS framework"}]
    for i, var in enumerate(variations, 1):
        all_prompts.append({
            "name": f"📋 Variation {i}: {var.get('name', 'Untitled')}",
            "prompt": var.get("prompt", ""),
            "description": var.get("description", ""),
        })

    if st.session_state.current_variation_index >= len(all_prompts):
        st.session_state.current_variation_index = 0
    current = all_prompts[st.session_state.current_variation_index]

    # ==================================================================
    # MAIN PROMPT DISPLAY
    # ==================================================================

    st.markdown("### 📝 The Engineered Prompt")

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
            f"<b>{current['name']}</b><br>"
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

    st.code(current["prompt"], language="text")
    if current.get("description"):
        st.caption(f"💡 {current['description']}")

    # ==================================================================
    # EXPORT — Clean buttons, no broken clipboard hack
    # ==================================================================

    st.markdown("---")
    st.markdown("### 📋 Export")

    current_prompt_text = current["prompt"]
    exp_col1, exp_col2, exp_col3 = st.columns(3)

    with exp_col1:
        st.download_button(
            "📄 Download .txt",
            data=current_prompt_text,
            file_name="promptlab_output.txt",
            mime="text/plain",
        )

    with exp_col2:
        md_content = (
            f"# PromptLab Output\n\n"
            f"**Task Type:** {task_type}\n"
            f"**Generated:** {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n---\n\n"
            f"## Prompt\n\n```\n{current_prompt_text}\n```\n"
        )
        if st.session_state.quality_scores:
            qs = st.session_state.quality_scores
            det = qs.get("deterministic", {})
            gr = qs.get("grounding", {})
            md_content += (
                f"\n## Quality Scores\n\n"
                f"### LLM Assessment: {qs.get('overall_score', 'N/A')}/100 ({qs.get('grade', '?')})\n"
                f"- Clarity: {qs.get('clarity', 'N/A')} | Specificity: {qs.get('specificity', 'N/A')} "
                f"| Structure: {qs.get('structure', 'N/A')} | Completeness: {qs.get('completeness', 'N/A')}\n\n"
                f"### Deterministic Assessment: {det.get('overall_score', 'N/A')}/100\n"
                f"- Clarity: {det.get('clarity', 'N/A')} | Specificity: {det.get('specificity', 'N/A')} "
                f"| Structure: {det.get('structure', 'N/A')} | Completeness: {det.get('completeness', 'N/A')}\n\n"
                f"### Grounding Agreement: {gr.get('agreement_score', 'N/A')}%\n"
            )
        st.download_button(
            "📝 Download .md",
            data=md_content,
            file_name="promptlab_output.md",
            mime="text/markdown",
        )

    with exp_col3:
        json_data = {
            "exported_at": datetime.datetime.now().isoformat(),
            "tool": "PromptLab",
            "task_type": task_type,
            "interpretation": interpretation.get("interpretation", ""),
            "optimized_prompt": current_prompt_text,
            "quality_scores": st.session_state.quality_scores,
        }
        st.download_button(
            "📦 Download .json",
            data=json.dumps(json_data, indent=2),
            file_name="promptlab_output.json",
            mime="application/json",
        )

    # ==================================================================
    # QUALITY SCORING — inline, not in a tab
    # ==================================================================

    st.markdown("---")
    st.markdown("### 📊 Prompt Quality Score")
    st.markdown(
        "Two independent scorers: **deterministic rule-based analyzer** (instant, reproducible) "
        "and **LLM judge** (nuanced). Disagreements are flagged."
    )

    if st.button("🎯 Score This Prompt", type="primary", key="score_btn"):
        with st.spinner("🔍 Running dual evaluation..."):
            scores = score_prompt_quality(current_prompt_text, task_type)
            st.session_state.quality_scores = scores

    if st.session_state.quality_scores:
        qs = st.session_state.quality_scores
        det = qs.get("deterministic", {})
        grounding = qs.get("grounding", {})

        grade = qs.get("grade", "?")
        overall = qs.get("overall_score", 0)
        agreement = grounding.get("agreement_score", 0)
        grade_colors = {
            "A+": "#00C851", "A": "#00C851", "B+": "#33b5e5", "B": "#33b5e5",
            "C+": "#ffbb33", "C": "#ffbb33", "D": "#ff4444", "F": "#CC0000",
        }
        color = grade_colors.get(grade, "#666")
        agree_color = "#00C851" if agreement >= 80 else "#ffbb33" if agreement >= 60 else "#ff4444"
        agree_label = "Well Grounded" if agreement >= 80 else "Moderate" if agreement >= 60 else "Low Agreement"

        col_grade, col_agree = st.columns(2)
        with col_grade:
            st.markdown(
                f"<div style='text-align:center;padding:20px;border:2px solid {color};border-radius:10px;'>"
                f"<span style='font-size:60px;font-weight:bold;color:{color};'>{grade}</span>"
                f"<br><span style='font-size:22px;color:#888;'>{overall}/100</span>"
                f"<br><em>{qs.get('one_line_verdict', '')}</em>"
                f"</div>",
                unsafe_allow_html=True,
            )
        with col_agree:
            st.markdown(
                f"<div style='text-align:center;padding:20px;border:2px solid {agree_color};border-radius:10px;'>"
                f"<span style='font-size:60px;font-weight:bold;color:{agree_color};'>{agreement}%</span>"
                f"<br><span style='font-size:22px;color:#888;'>Grounding Agreement</span>"
                f"<br><em>{agree_label}</em>"
                f"</div>",
                unsafe_allow_html=True,
            )

        # Dimension comparison
        st.markdown("#### LLM vs Deterministic")
        for dim in ["clarity", "specificity", "structure", "completeness"]:
            llm_val = qs.get(dim, 50)
            det_val = det.get(dim, 50)
            diff = abs(llm_val - det_val)
            flag = " ⚠️" if diff > 20 else ""

            col_label, col_llm_bar, col_det_bar = st.columns([1, 2, 2])
            with col_label:
                st.markdown(f"**{dim.title()}**{flag}")
            with col_llm_bar:
                bc = "#00C851" if llm_val >= 80 else "#ffbb33" if llm_val >= 60 else "#ff4444"
                st.markdown(
                    f"LLM: **{llm_val}**/100"
                    f"<div style='background:#eee;border-radius:8px;height:14px;width:100%;'>"
                    f"<div style='background:{bc};border-radius:8px;height:14px;width:{llm_val}%;'></div></div>",
                    unsafe_allow_html=True,
                )
            with col_det_bar:
                bc2 = "#00C851" if det_val >= 80 else "#ffbb33" if det_val >= 60 else "#ff4444"
                st.markdown(
                    f"Rules: **{det_val}**/100"
                    f"<div style='background:#eee;border-radius:8px;height:14px;width:100%;'>"
                    f"<div style='background:{bc2};border-radius:8px;height:14px;width:{det_val}%;'></div></div>",
                    unsafe_allow_html=True,
                )

        # Disagreements
        disagreements = grounding.get("disagreements", [])
        if disagreements:
            st.markdown("#### ⚠️ Grounding Disagreements")
            for d in disagreements:
                st.warning(
                    f"**{d['dimension'].title()}** — LLM: {d['llm']}, Rules: {d['deterministic']} "
                    f"(gap: {d['difference']} pts). {d['note']}"
                )

        # Evidence
        evidence = det.get("evidence", {})
        if evidence:
            with st.expander("🔍 Structural Evidence", expanded=False):
                ev1, ev2, ev3 = st.columns(3)
                with ev1:
                    st.metric("Words", evidence.get("word_count", 0))
                    st.metric("Sentences", evidence.get("sentence_count", 0))
                with ev2:
                    st.metric("Headers", evidence.get("header_count", 0))
                    st.metric("List Items", evidence.get("list_items", 0))
                with ev3:
                    st.metric("Action Verbs", evidence.get("strong_verb_count", 0))
                    st.metric("Vague Words", evidence.get("vague_word_count", 0))

                patterns = det.get("patterns_found", [])
                missing = evidence.get("patterns_missing", [])
                if patterns:
                    st.markdown("**Detected:** " + ", ".join(f"`{p.replace('_', ' ')}`" for p in patterns))
                if missing:
                    st.markdown("**Missing:** " + ", ".join(f"`{p.replace('_', ' ')}`" for p in missing))

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

    st.markdown("---")