"""
Asclepius - Medical Prescription & Lab Report Assistant
Helps people understand medical documents in their own language (English, Telugu, Hindi).
"""

import io
import os
import re
import json
from typing import Dict, Any, List

import streamlit as st
import pandas as pd
from PIL import Image
from gtts import gTTS

import extractor
import ranges
import reminders

# Page configuration
st.set_page_config(
    page_title="Asclepius - Understand Your Health",
    page_icon="⚕️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling for Ideathon MVP Demo
st.markdown("""
<style>
    /* Global Styling */
    .main-title {
        font-size: 2.3rem;
        font-weight: 700;
        color: #0f172a;
        margin-bottom: 0.1rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    .tagline {
        font-size: 1.15rem;
        color: #0284c7;
        font-weight: 500;
        margin-bottom: 1.5rem;
    }
    .step-header {
        font-size: 1.25rem;
        font-weight: 600;
        color: #1e293b;
        background: #f1f5f9;
        padding: 0.6rem 1rem;
        border-radius: 8px;
        border-left: 5px solid #0284c7;
        margin-top: 1rem;
        margin-bottom: 1rem;
    }
    .privacy-card {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 0.8rem;
        font-size: 0.85rem;
        color: #64748b;
        margin-top: 1rem;
    }
    .badge-high {
        background-color: #fee2e2;
        color: #991b1b;
        padding: 0.2rem 0.6rem;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.85rem;
        display: inline-block;
    }
    .badge-low {
        background-color: #dbeafe;
        color: #1e40af;
        padding: 0.2rem 0.6rem;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.85rem;
        display: inline-block;
    }
    .badge-normal {
        background-color: #dcfce7;
        color: #166534;
        padding: 0.2rem 0.6rem;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.85rem;
        display: inline-block;
    }
    .badge-unknown {
        background-color: #f3f4f6;
        color: #4b5563;
        padding: 0.2rem 0.6rem;
        border-radius: 6px;
        font-size: 0.85rem;
        display: inline-block;
    }
    .conf-low {
        background-color: #fef3c7;
        color: #92400e;
        padding: 0.2rem 0.5rem;
        border-radius: 4px;
        font-weight: 600;
    }
    .conf-high {
        background-color: #e0f2fe;
        color: #0369a1;
        padding: 0.2rem 0.5rem;
        border-radius: 4px;
    }
    .question-box {
        background: #f0fdf4;
        border: 1px solid #bbf7d0;
        border-radius: 8px;
        padding: 1rem;
        margin-top: 1rem;
    }
    .question-title {
        font-weight: 600;
        color: #166534;
        font-size: 1.05rem;
        margin-bottom: 0.5rem;
    }
    .schedule-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 0.8rem 1rem;
        margin-bottom: 0.7rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .schedule-time {
        font-size: 1.1rem;
        font-weight: 700;
        color: #0284c7;
    }
</style>
""", unsafe_allow_html=True)


def init_session():
    """Initializes Streamlit session states."""
    if "extracted_data" not in st.session_state:
        st.session_state["extracted_data"] = None
    if "verified_data" not in st.session_state:
        st.session_state["verified_data"] = None
    if "active_step" not in st.session_state:
        st.session_state["active_step"] = "1. Upload"
    if "explanation_result" not in st.session_state:
        st.session_state["explanation_result"] = None
    if "audio_bytes" not in st.session_state:
        st.session_state["audio_bytes"] = None
    if "audio_lang" not in st.session_state:
        st.session_state["audio_lang"] = None
    if "image_preview" not in st.session_state:
        st.session_state["image_preview"] = None


init_session()


# --- SIDEBAR CONFIGURATION ---
with st.sidebar:
    st.markdown("### ⚙️ Settings")

    # 1. Language selector
    languages = {
        "English": "en",
        "Telugu (తెలుగు)": "te",
        "Hindi (हिन्दी)": "hi"
    }
    selected_lang_label = st.selectbox(
        "🌐 Choose Your Language",
        options=list(languages.keys()),
        index=0,
        help="Select your preferred language for medical explanations and voice output."
    )
    selected_lang_code = languages[selected_lang_label]

    st.markdown("---")

    # Vision LLM & Gemini API Key Configuration
    st.markdown("### 🤖 Gemini API Configuration")
    
    # Retrieve key from st.secrets (Streamlit Cloud / .streamlit/secrets.toml) or os.environ
    configured_key = ""
    key_source = None
    try:
        if hasattr(st, "secrets") and "GEMINI_API_KEY" in st.secrets:
            configured_key = str(st.secrets["GEMINI_API_KEY"]).strip()
            key_source = "Streamlit Secrets"
    except Exception:
        pass

    if not configured_key:
        configured_key = os.environ.get("GEMINI_API_KEY", "").strip()
        if configured_key:
            key_source = "Environment Variable"

    api_key_input = st.text_input(
        "Gemini API Key",
        value=configured_key,
        type="password",
        placeholder="AIzaSy...",
        help="Reads from Streamlit secrets or GEMINI_API_KEY env var. You can enter or override it here."
    )
    active_api_key = api_key_input.strip() if api_key_input.strip() else configured_key

    if active_api_key:
        source_label = "Custom Entry" if api_key_input.strip() != configured_key else (key_source or "Configured")
        st.caption(f"✅ Key loaded from: **{source_label}**")
    else:
        st.caption("ℹ️ No key set. Demo fixtures work without a key!")

    # Model selector (easy to change, uses constant DEFAULT_MODEL)
    model_choice = st.selectbox(
        "Vision / Reasoning Model",
        options=[extractor.DEFAULT_MODEL, "gemini-3.5-flash-lite", "gemini-3.1-pro-preview"],
        index=0,
        help="Model constant defined in extractor.py"
    )

    st.markdown("---")

    # Sample demo loaders
    st.markdown("### 🧪 Instant Demo Fixtures")
    st.caption("No image on hand? Click below to demo immediately:")
    col_demo1, col_demo2 = st.columns(2)
    with col_demo1:
        if st.button("📋 Sample Prescription", use_container_width=True):
            fixture_path = os.path.join(os.path.dirname(__file__), "fixtures", "sample_prescription.json")
            with open(fixture_path, "r", encoding="utf-8") as f:
                sample_data = json.load(f)
            st.session_state["extracted_data"] = sample_data
            st.session_state["verified_data"] = sample_data
            st.session_state["explanation_result"] = None
            st.session_state["audio_bytes"] = None
            st.session_state["image_preview"] = None
            st.session_state["active_step"] = "2. Verify & Edit"
            st.success("Loaded sample prescription!")
            st.rerun()

    with col_demo2:
        if st.button("🔬 Sample Lab Report", use_container_width=True):
            fixture_path = os.path.join(os.path.dirname(__file__), "fixtures", "sample_lab_report.json")
            with open(fixture_path, "r", encoding="utf-8") as f:
                sample_data = json.load(f)
            st.session_state["extracted_data"] = sample_data
            st.session_state["verified_data"] = sample_data
            st.session_state["explanation_result"] = None
            st.session_state["audio_bytes"] = None
            st.session_state["image_preview"] = None
            st.session_state["active_step"] = "2. Verify & Edit"
            st.success("Loaded sample lab report!")
            st.rerun()

    # Privacy Note
    st.markdown("""
    <div class="privacy-card">
        🔒 <strong>Privacy Note:</strong><br>
        Asclepius does not store your uploaded documents. Images and medical data are processed purely in-memory and never written to disk or permanent storage.
    </div>
    """, unsafe_allow_html=True)


# --- MAIN HEADER ---
st.markdown('<div class="main-title">⚕️ Asclepius</div>', unsafe_allow_html=True)
st.markdown('<div class="tagline">Understand your health, in your own language</div>', unsafe_allow_html=True)

# Step-by-Step Navigation
tabs = st.tabs([
    "1. 📤 Upload Document",
    "2. 📝 Verify & Edit",
    "3. 💡 Plain Explanation & Audio",
    "4. ⏰ Daily Schedule & Reminders"
])


# ==============================================================================
# STEP 1: UPLOAD DOCUMENT
# ==============================================================================
with tabs[0]:
    st.markdown('<div class="step-header">Step 1: Upload Prescription or Lab Report</div>', unsafe_allow_html=True)
    st.write("Take a photo with your device camera or upload an existing JPG/PNG file.")

    upload_mode = st.radio("Input Method:", ["📁 Upload Image File", "📸 Use Device Camera"], horizontal=True)

    uploaded_file = None
    if upload_mode == "📁 Upload Image File":
        uploaded_file = st.file_uploader("Upload Prescription / Lab Report (JPG, PNG)", type=["jpg", "jpeg", "png"])
    else:
        uploaded_file = st.camera_input("Capture document using camera")

    col_preview, col_action = st.columns([1, 1])

    if uploaded_file is not None:
        image_bytes = uploaded_file.getvalue()
        mime_type = uploaded_file.type or "image/jpeg"
        st.session_state["image_preview"] = image_bytes

        with col_preview:
            st.image(image_bytes, caption="Document Preview", use_container_width=True)

        with col_action:
            st.info("Document loaded! Click below to extract details with Gemini Vision.")
            if st.button("🔍 Extract & Analyze with Gemini", type="primary", use_container_width=True):
                if not active_api_key:
                    st.error("⚠️ Gemini API Key is required! Please enter your key in the sidebar.")
                else:
                    with st.spinner("Analyzing document with Gemini Vision LLM..."):
                        try:
                            extracted = extractor.extract_medical_data_from_image(
                                image_bytes=image_bytes,
                                mime_type=mime_type,
                                api_key=active_api_key,
                                model=model_choice
                            )
                            st.session_state["extracted_data"] = extracted
                            st.session_state["verified_data"] = extracted
                            st.session_state["explanation_result"] = None
                            st.session_state["audio_bytes"] = None
                            st.success("Extraction successful! Proceeding to Step 2...")
                            st.rerun()
                        except Exception as e:
                            st.error(f"Error extracting data: {str(e)}")
                            st.info("Tip: You can also use the Instant Demo Fixtures in the sidebar to test.")
    else:
        if st.session_state.get("image_preview"):
            with col_preview:
                st.image(st.session_state["image_preview"], caption="Current Document Preview", use_container_width=True)
        else:
            st.markdown("""
            > **Need a test run?**
            > Click **'📋 Sample Prescription'** or **'🔬 Sample Lab Report'** in the sidebar to test the full pipeline right away.
            """)


# ==============================================================================
# STEP 2: VERIFY & EDIT
# ==============================================================================
with tabs[1]:
    st.markdown('<div class="step-header">Step 2: Verify & Edit Extracted Data</div>', unsafe_allow_html=True)

    data = st.session_state.get("verified_data") or st.session_state.get("extracted_data")

    if not data:
        st.warning("No document has been analyzed yet. Please upload a file in Step 1 or load a sample from the sidebar.")
    else:
        doc_type = data.get("type", "prescription")
        st.write(f"**Detected Document Type:** `{doc_type.upper()}` (You can toggle below if needed)")
        new_doc_type = st.radio("Document Category:", ["prescription", "lab_report"], index=0 if doc_type == "prescription" else 1, horizontal=True)
        data["type"] = new_doc_type

        # Highlight unclear items if present
        unclear_list = data.get("unclear_items", [])
        if unclear_list:
            with st.expander("⚠️ Notes on Unclear / Ambiguous Items Detected", expanded=True):
                for item in unclear_list:
                    st.markdown(f"- ❓ {item}")
                st.caption("Please review the editable table below to correct or supply any unclear values.")

        # Editable tables
        if new_doc_type == "prescription":
            st.subheader("💊 Prescribed Medicines")
            st.caption("Review extracted medicines. Items with low confidence are highlighted. You can edit any cell directly:")

            meds_list = data.get("medicines", [])
            df_meds = pd.DataFrame(meds_list) if meds_list else pd.DataFrame(columns=["name", "dose", "frequency", "timing", "duration_days", "confidence"])

            # Ensure columns exist
            for c in ["name", "dose", "frequency", "timing", "duration_days", "confidence"]:
                if c not in df_meds.columns:
                    df_meds[c] = ""

            # Highlight low confidence items in warning
            low_conf_meds = df_meds[df_meds["confidence"].str.lower() == "low"]["name"].tolist()
            if low_conf_meds:
                st.warning(f"⚠️ Low confidence detected for: **{', '.join(low_conf_meds)}**. Please double check their dose and frequency.")

            edited_meds_df = st.data_editor(
                df_meds,
                num_rows="dynamic",
                use_container_width=True,
                column_config={
                    "name": st.column_config.TextColumn("Medicine Name", required=True),
                    "dose": st.column_config.TextColumn("Dose (e.g. 500mg)"),
                    "frequency": st.column_config.TextColumn("Frequency (e.g. TDS, 1-0-1)"),
                    "timing": st.column_config.TextColumn("Timing (e.g. After food)"),
                    "duration_days": st.column_config.TextColumn("Duration (Days)"),
                    "confidence": st.column_config.SelectboxColumn("Confidence", options=["high", "medium", "low"])
                }
            )
            data["medicines"] = edited_meds_df.to_dict(orient="records")

        else:
            st.subheader("🔬 Laboratory Test Results")
            st.caption("Review extracted test values. You can modify any row or add missing measurements:")

            labs_list = data.get("lab_values", [])
            df_labs = pd.DataFrame(labs_list) if labs_list else pd.DataFrame(columns=["test", "value", "unit", "confidence"])

            for c in ["test", "value", "unit", "confidence"]:
                if c not in df_labs.columns:
                    df_labs[c] = ""

            low_conf_labs = df_labs[df_labs["confidence"].str.lower() == "low"]["test"].tolist()
            if low_conf_labs:
                st.warning(f"⚠️ Low confidence detected for: **{', '.join(low_conf_labs)}**.")

            edited_labs_df = st.data_editor(
                df_labs,
                num_rows="dynamic",
                use_container_width=True,
                column_config={
                    "test": st.column_config.TextColumn("Test Name", required=True),
                    "value": st.column_config.TextColumn("Measured Value", required=True),
                    "unit": st.column_config.TextColumn("Unit (e.g. mg/dL, %)"),
                    "confidence": st.column_config.SelectboxColumn("Confidence", options=["high", "medium", "low"])
                }
            )
            data["lab_values"] = edited_labs_df.to_dict(orient="records")

        st.session_state["verified_data"] = data

        col_save, col_next = st.columns([1, 1])
        with col_save:
            st.success("Data is automatically saved in your active session.")
        with col_next:
            if st.button("➡️ Proceed to Explain in My Language", type="primary", use_container_width=True):
                st.session_state["active_step"] = "3. Explain"
                st.info("Now switch to Tab 3 to view the simplified explanation and audio!")


# ==============================================================================
# STEP 3: EXPLAIN & AUDIO
# ==============================================================================
with tabs[2]:
    st.markdown('<div class="step-header">Step 3: Plain Language Explanation & Voice</div>', unsafe_allow_html=True)

    verified = st.session_state.get("verified_data")
    if not verified:
        st.warning("Please verify your data in Step 2 first.")
    else:
        # Part A: Deterministic Lab Range Evaluation (Code, NOT LLM)
        if verified.get("type") == "lab_report" and verified.get("lab_values"):
            st.markdown("### 📊 Clinical Reference Range Evaluation (Deterministic Code Check)")
            st.caption("Each test value below is evaluated purely through verified reference ranges in code, NOT by the LLM:")

            eval_results = []
            for item in verified.get("lab_values", []):
                t_name = item.get("test", "")
                t_val = item.get("value", "")
                t_unit = item.get("unit", "")
                ev = ranges.evaluate_lab_value(t_name, t_val, t_unit)
                eval_results.append(ev)

            # Render colored evaluation cards
            eval_cols = st.columns(2)
            for idx, res in enumerate(eval_results):
                target_col = eval_cols[idx % 2]
                status = res["status"]
                badge_class = "badge-normal"
                if status == "High":
                    badge_class = "badge-high"
                elif status == "Low":
                    badge_class = "badge-low"
                elif status == "Unknown":
                    badge_class = "badge-unknown"

                with target_col:
                    st.markdown(f"""
                    <div style="background:#ffffff; border:1px solid #e2e8f0; border-radius:8px; padding:0.9rem; margin-bottom:0.8rem; box-shadow: 0 1px 2px rgba(0,0,0,0.04);">
                        <div style="display:flex; justify-content:space-between; align-items:center;">
                            <strong style="font-size:1.05rem; color:#1e293b;">{res['display_name']}</strong>
                            <span class="{badge_class}">{res['status_badge']}</span>
                        </div>
                        <div style="font-size:1.15rem; font-weight:700; margin:0.3rem 0; color:#0f172a;">
                            {res['raw_value']} <span style="font-size:0.9rem; font-weight:normal; color:#64748b;">{res['unit']}</span>
                        </div>
                        <div style="font-size:0.85rem; color:#475569;">
                            <strong>Standard Range:</strong> {res['ref_range_str']}
                        </div>
                        <div style="font-size:0.8rem; color:#64748b; margin-top:0.3rem;">
                            <em>{res['description']}</em>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

        st.markdown("---")

        # Part B: Multilingual Simplified Explanation (Class-8 level)
        st.markdown(f"### 💬 Patient Explanation ({selected_lang_label})")
        st.write("Asclepius explains your medical documents in simple, everyday language at a Class-8 reading level.")

        explanation_col1, explanation_col2 = st.columns([1, 1])
        with explanation_col1:
            explain_btn = st.button("✨ Generate Plain Language Explanation", type="primary", use_container_width=True)

        if explain_btn or st.session_state.get("explanation_result") is None:
            if explain_btn:
                with st.spinner(f"Asclepius is drafting a simple explanation in {selected_lang_label}..."):
                    try:
                        if active_api_key:
                            res = extractor.explain_medical_data(
                                verified_data=verified,
                                language=selected_lang_code,
                                api_key=active_api_key,
                                model=model_choice
                            )
                            st.session_state["explanation_result"] = res["explanation"]
                        else:
                            # Demo fallback explanation
                            demo_exp = extractor.get_demo_explanation(verified, selected_lang_code)
                            st.session_state["explanation_result"] = demo_exp
                    except Exception as err:
                        st.error(f"Explanation generation failed: {err}")
                        demo_exp = extractor.get_demo_explanation(verified, selected_lang_code)
                        st.session_state["explanation_result"] = demo_exp
                        st.info("Switched to built-in verified sample explanation.")

        # Display current explanation
        current_exp = st.session_state.get("explanation_result")
        if current_exp:
            st.markdown(current_exp)

            st.markdown("---")

            # Part C: Audio (gTTS Voice Output)
            st.markdown(f"### 🔊 Listen in {selected_lang_label}")
            st.caption("Play an accessible audio voice readout in your selected language.")

            # Button to generate/play audio
            if st.button("▶️ Play Audio", key="play_audio_btn"):
                with st.spinner("Synthesizing clear audio speech..."):
                    # Clean markdown to produce conversational speech
                    speech_text = re.sub(r"#+\s*", "", current_exp)
                    speech_text = re.sub(r"\*{1,2}(.*?)\*{1,2}", r"\1", speech_text)
                    speech_text = re.sub(r"^\s*[-*•]\s*", "", speech_text, flags=re.MULTILINE)
                    speech_text = re.sub(r"\n+", ". ", speech_text).strip()

                    try:
                        tts = gTTS(text=speech_text, lang=selected_lang_code, slow=False)
                        fp = io.BytesIO()
                        tts.write_to_fp(fp)
                        fp.seek(0)
                        st.session_state["audio_bytes"] = fp.read()
                        st.session_state["audio_lang"] = selected_lang_code
                    except Exception as audio_err:
                        st.error(f"Voice synthesis error: {audio_err}")

            if st.session_state.get("audio_bytes"):
                st.audio(st.session_state["audio_bytes"], format="audio/mp3")


# ==============================================================================
# STEP 4: REMINDERS & CALENDAR SCHEDULE
# ==============================================================================
with tabs[3]:
    st.markdown('<div class="step-header">Step 4: Daily Schedule & Calendar Reminders</div>', unsafe_allow_html=True)

    verified = st.session_state.get("verified_data")
    if not verified:
        st.warning("Please verify your prescription in Step 2 first.")
    else:
        medicines = verified.get("medicines", [])
        if verified.get("type") != "prescription" or not medicines:
            st.info("The current document is a Lab Report. Medicine schedules are generated when a Prescription is uploaded.")
            st.write("If this was a prescription, you can change the document category to 'Prescription' in Step 2.")
        else:
            st.write("Here is your consolidated daily timetable based on prescribed frequencies and meal timings:")

            schedule = reminders.build_daily_schedule(medicines)

            if not schedule:
                st.info("No medicine schedule could be generated. Please add medicines in Step 2.")
            else:
                # Group by slots: Morning, Afternoon, Evening, Night
                col_m, col_a, col_e, col_n = st.columns(4)
                slots_mapping = {
                    "Morning": col_m,
                    "Afternoon": col_a,
                    "Evening": col_e,
                    "Night": col_n
                }

                # Display cards
                for item in schedule:
                    slot_target = slots_mapping.get(item["slot"], col_m)
                    with slot_target:
                        st.markdown(f"""
                        <div class="schedule-card">
                            <span class="schedule-time">⏰ {item['time']}</span>
                            <div style="font-size:0.8rem; font-weight:600; color:#64748b; margin-bottom:0.3rem;">{item['slot']}</div>
                            <div style="font-weight:700; color:#0f172a; font-size:1.05rem;">{item['medicine']}</div>
                            <div style="font-size:0.9rem; color:#0369a1; font-weight:600;">Dose: {item['dose']}</div>
                            <div style="font-size:0.85rem; color:#475569; margin-top:0.2rem;">🍴 {item['timing_note']}</div>
                            <div style="font-size:0.8rem; color:#94a3b8; margin-top:0.2rem;">⏳ Course: {item['duration_days']} days</div>
                        </div>
                        """, unsafe_allow_html=True)

                st.markdown("---")

                # Download .ics calendar file
                st.subheader("📅 Sync with Your Phone Calendar")
                st.write("Download an RFC 5545 `.ics` file to import medicine alarms into Google Calendar, Apple Calendar, or Outlook.")

                ics_content = reminders.generate_ics_calendar(medicines)

                st.download_button(
                    label="📥 Download Calendar Reminders (.ics)",
                    data=ics_content,
                    file_name="asclepius_medicine_reminders.ics",
                    mime="text/calendar",
                    type="primary"
                )
                st.caption("Tip: Once downloaded, open the file on your smartphone to automatically add daily reminders with 10-minute alerts.")
