# ⚕️ Asclepius - Medical Prescription & Lab Report Assistant

> **"Understand your health, in your own language"**  
> An Ideathon MVP web application built with Streamlit, Gemini Vision LLM (`google-genai`), and gTTS.

---

## 🌟 What is Asclepius?

Medical prescriptions with handwritten doctor notes and dense laboratory diagnostic reports are notoriously difficult for patients and their families to understand—especially across regional languages.

**Asclepius** solves this by:
1. **Extracting** medicines and lab values from prescription/report photos using Gemini Vision into structured, strict JSON.
2. **Highlighting & Editing** extracted items in an interactive table, alerting the user to low-confidence or illegible doctor notes.
3. **Evaluating Lab Values deterministically in code** (`ranges.py`) against clinical reference ranges (Normal 🟢, High 🔺, Low 🔻) without LLM guesswork.
4. **Explaining** medical instructions in simple, compassionate everyday language (Class-8 reading level) in **English**, **Telugu (తెలుగు)**, or **Hindi (हिन्दी)**.
5. **Giving 3 practical questions** for the patient to ask their doctor at their next visit.
6. **Speaking aloud** the explanation using high-quality voice synthesis (`gTTS`) in the chosen language.
7. **Generating a Daily Schedule & .ICS Calendar** (`reminders.py`) that syncs medicine alarms into Google Calendar, Apple Calendar, or Outlook with 10-minute alert reminders.

---

## 🔒 Safety Guardrails & Compliance

Asclepius is built strictly following healthcare safety principles:
- **No Hallucination / Never Guess:** If handwriting is smudged or ambiguous, the model marks confidence as `low` and classifies it as `unclear` instead of guessing.
- **Never Diagnose:** Asclepius explains only what is written in the document; it never diagnoses diseases.
- **Never Alter Doses:** It never advises changing, stopping, or starting a medicine.
- **Mandatory Doctor Confirmation:** All explanations strictly conclude with: *"Please confirm with your doctor."*
- **Privacy First:** Uploaded documents are processed entirely in memory. Zero patient images or data are saved to disk or external databases.

---

## 📁 Project Structure

```
asclepius/
├── app.py                  # Main Streamlit web application & UI
├── extractor.py            # Gemini Vision API extractor & multilingual explainer
├── ranges.py               # Deterministic code-based lab reference ranges & evaluator
├── reminders.py            # Schedule builder & RFC 5545 .ics calendar generator
├── requirements.txt        # Python package dependencies
├── README.md               # Documentation & setup guide
└── fixtures/
    ├── sample_prescription.json  # Sample prescription for instant testing
    └── sample_lab_report.json    # Sample lab test report with flags for testing
```

---

## 🚀 Quick Setup & Installation

### 1. Prerequisites
- Python 3.10+ (tested on Python 3.14)
- A Google Gemini API Key ([Get one from Google AI Studio](https://aistudio.google.com/))

### 2. Clone / Navigate to Directory
```bash
cd asclepius
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Set Your Gemini API Key
**On Windows (PowerShell):**
```powershell
$env:GEMINI_API_KEY="your-gemini-api-key-here"
```

**On Linux / macOS:**
```bash
export GEMINI_API_KEY="your-gemini-api-key-here"
```

**Or using Streamlit Secrets (.streamlit/secrets.toml):**
```bash
# Copy the example secrets file
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
# Edit .streamlit/secrets.toml and insert your real GEMINI_API_KEY
```
*(Note: `.streamlit/secrets.toml` is included in `.gitignore` so your API key will never be committed or leaked to GitHub!)*

### 5. Run the Application
```bash
streamlit run app.py
# Or:
python -m streamlit run app.py
```
Then open `http://localhost:8501` in your browser.

---

## 🐙 How to Share on GitHub & Deploy to Streamlit Cloud

### Sharing Safely to GitHub
1. Initialize git and commit your files:
   ```bash
   git init
   git add .
   git commit -m "Initial commit of Asclepius MVP"
   ```
2. Link your remote repository and push:
   ```bash
   git branch -M main
   git remote add origin https://github.com/<your-username>/asclepius.git
   git push -u origin main
   ```
   > [!IMPORTANT]
   > `.gitignore` is already preconfigured to block `.streamlit/secrets.toml` and `.env` files. Your actual API key will **never** be uploaded to GitHub.

### Deploying to Streamlit Community Cloud (Free 1-Click Hosting)
1. Go to [share.streamlit.io](https://share.streamlit.io/) and connect your GitHub account.
2. Select your `asclepius` repository and specify `app.py` as the entrypoint.
3. In **Advanced Settings -> Secrets**, paste:
   ```toml
   GEMINI_API_KEY = "your-actual-gemini-api-key"
   ```
4. Click **Deploy**! Anyone visiting your public app URL can now use the live Gemini Vision extraction and translations seamlessly without needing their own key.

---

## 🧪 Demo Mode (No Image Required!)

For ideathon presentations and testing when you don't have a physical prescription photo ready:
1. In the sidebar, look for **"Instant Demo Fixtures"**.
2. Click **"📋 Sample Prescription"** to instantly test:
   - Amoxicillin, Paracetamol, Pantoprazole, Cetirizine, and an unclear cough syrup.
   - Low-confidence alert and notes on illegible doctor signature.
   - Multilingual explanation in English, Telugu, or Hindi.
   - gTTS voice readout.
   - Daily schedule timetable and `.ics` calendar download.
3. Click **"🔬 Sample Lab Report"** to test:
   - Hemoglobin (flagged 🔻 Low), Fasting Blood Sugar (flagged 🔺 High), HbA1c (flagged 🔺 High), Creatinine (flagged 🟢 Normal), Total Cholesterol (flagged 🔺 High), and TSH (flagged 🟢 Normal).
   - Code-based range evaluation with colored pills.

---

## 🛠️ Technology Stack

- **Streamlit**: Fast, reactive, responsive frontend UI with custom CSS.
- **google-genai SDK**: Official Google Gen AI SDK utilizing `gemini-3.8-flash` for high-fidelity vision and multilingual translation.
- **gTTS (Google Text-to-Speech)**: Local in-memory voice generation in English (`en`), Telugu (`te`), and Hindi (`hi`).
- **Pandas**: Dynamic tabular data editing with `st.data_editor`.
- **Pillow**: Image preview and memory stream handling.
- **RFC 5545 iCalendar**: Standard `.ics` reminders with recurrence rules and popup alarms.
