"""
Asclepius - Gemini Vision Extractor and Multilingual Explainer
Handles image extraction into strict structured JSON and simplified multilingual explanations.
"""

import os
import re
import json
import logging
from typing import Dict, Any, List, Optional

# Vision and Language Model constant - easy to change in one place
DEFAULT_MODEL: str = "gemini-3.8-flash"

logger = logging.getLogger("asclepius.extractor")

EXTRACTION_SYSTEM_PROMPT = """You are Asclepius, an expert medical document visual extraction engine.
Your sole job is to inspect doctor prescriptions or laboratory diagnostic reports and extract the structured data accurately.

CRITICAL SAFETY AND FIDELITY RULES:
1. Extract ONLY what is explicitly legible in the image.
2. NEVER diagnose any medical condition, disease, or illness.
3. NEVER suggest changing, starting, adding, or discontinuing any medicine, treatment, or dose.
4. If handwriting, words, dosages, or numbers are unclear, smudged, clipped, or ambiguous, do NOT guess. Mark the confidence as 'low' with value 'unclear', and list the specific unclear observation in 'unclear_items'.
5. Always preserve original medication names, numbers, and units as closely as written.

OUTPUT FORMAT REQUIREMENTS:
You MUST respond with STRICT, VALID JSON ONLY.
Do NOT include markdown formatting fences (do NOT use ```json or ```).
Do NOT include conversational preambles, introductory thoughts, or postscripts.

The JSON output MUST match this exact schema:
{
  "type": "prescription" | "lab_report",
  "medicines": [
    {
      "name": "Name of medicine / brand / generic",
      "dose": "Dosage (e.g. 500 mg, 1 tab, 10 ml)",
      "frequency": "Frequency (e.g. Once daily, Twice daily, TDS, 1-0-1, HS)",
      "timing": "Timing (e.g. After food, Before breakfast, Bedtime)",
      "duration_days": "Duration in days (e.g. 5, 7, 30)",
      "confidence": "high" | "medium" | "low"
    }
  ],
  "lab_values": [
    {
      "test": "Name of laboratory test (e.g. Hemoglobin, Fasting Glucose, HbA1c)",
      "value": "Measured value as written",
      "unit": "Unit of measurement (e.g. g/dL, mg/dL, %)",
      "confidence": "high" | "medium" | "low"
    }
  ],
  "unclear_items": [
    "Specific notes about any unreadable, smudged, or ambiguous text"
  ]
}

If the image is a prescription, leave 'lab_values' as an empty array [].
If the image is a lab report, leave 'medicines' as an empty array [].
"""

EXPLANATION_SYSTEM_PROMPT = """You are Asclepius, a compassionate, patient-friendly medical communicator.
Your role is to explain verified medical documents (prescriptions or lab reports) to ordinary patients in simple, reassuring, and completely understandable language (about an 8th-grade / Class-8 reading level).

CRITICAL SAFETY RULES:
1. Explain ONLY what is in the provided verified document data.
2. NEVER diagnose any disease or medical condition.
3. NEVER suggest changing, starting, or stopping any medication, treatment, or dose.
4. If an item or value is marked unclear, politely remind the patient to clarify that specific item with their doctor.
5. ALWAYS conclude the explanation with: "Please confirm with your doctor." (translated naturally into the chosen target language).

LANGUAGE AND TONE:
- English: Simple, warm, 8th-grade level, no confusing medical jargon.
- Telugu (తెలుగు): Natural, conversational, respectful Telugu (గ్రామ్య లేదా వ్యవహారిక శైలి), written in Telugu script. Avoid confusing English transliterations where common Telugu words exist.
- Hindi (हिन्दी): Simple, respectful, conversational Hindi (सरल बोलचाल की भाषा), written in Devanagari script.

OUTPUT STRUCTURE:
You must provide a clear response with:
1. Brief Document Overview (What this prescription or lab report is for).
2. Item-by-item breakdown (Plain-language explanation of each medicine's purpose/instructions or each lab test's general role in the body).
3. Exactly 3 important, thoughtful questions the patient can ask their doctor during their next visit.
4. Mandatory closing line: "Please confirm with your doctor." (or its natural Telugu/Hindi translation).
"""


def clean_json_text(raw_text: str) -> str:
    """
    Strips markdown code fences (```json, ```) and extracts raw JSON string.
    """
    text = raw_text.strip()
    # Strip markdown backticks
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE | re.MULTILINE)
    text = re.sub(r"```\s*$", "", text, flags=re.MULTILINE)
    text = text.strip()

    # If extra text surrounds JSON, try finding outermost curly braces
    if not (text.startswith("{") and text.endswith("}")):
        match = re.search(r"(\{.*\})", text, flags=re.DOTALL)
        if match:
            text = match.group(1).strip()

    return text


def sanitize_extracted_dict(data: Any) -> Dict[str, Any]:
    """
    Ensures that the extracted dictionary conforms strictly to the expected schema.
    """
    if not isinstance(data, dict):
        data = {}

    doc_type = data.get("type", "prescription")
    if doc_type not in ["prescription", "lab_report"]:
        doc_type = "prescription" if len(data.get("medicines", [])) >= len(data.get("lab_values", [])) else "lab_report"

    medicines = []
    for item in data.get("medicines", []):
        if isinstance(item, dict):
            medicines.append({
                "name": str(item.get("name") or "Unspecified Medicine"),
                "dose": str(item.get("dose") or ""),
                "frequency": str(item.get("frequency") or ""),
                "timing": str(item.get("timing") or ""),
                "duration_days": str(item.get("duration_days") or "5"),
                "confidence": str(item.get("confidence") or "medium").lower()
            })

    lab_values = []
    for item in data.get("lab_values", []):
        if isinstance(item, dict):
            lab_values.append({
                "test": str(item.get("test") or "Unknown Test"),
                "value": str(item.get("value") or ""),
                "unit": str(item.get("unit") or ""),
                "confidence": str(item.get("confidence") or "medium").lower()
            })

    unclear_items = [str(x) for x in data.get("unclear_items", []) if str(x).strip()]

    return {
        "type": doc_type,
        "medicines": medicines,
        "lab_values": lab_values,
        "unclear_items": unclear_items
    }


def resolve_api_key(passed_key: Optional[str] = None) -> Optional[str]:
    """
    Resolves the Gemini API key from direct argument, Streamlit secrets, or environment variable.
    """
    if passed_key and passed_key.strip():
        return passed_key.strip()
    try:
        import streamlit as st
        if hasattr(st, "secrets") and "GEMINI_API_KEY" in st.secrets:
            k = str(st.secrets["GEMINI_API_KEY"]).strip()
            if k:
                return k
    except Exception:
        pass
    env_k = os.environ.get("GEMINI_API_KEY", "").strip()
    return env_k if env_k else None


def extract_medical_data_from_image(
    image_bytes: bytes,
    mime_type: str = "image/jpeg",
    api_key: Optional[str] = None,
    model: str = DEFAULT_MODEL
) -> Dict[str, Any]:
    """
    Sends document image to Gemini Vision API and returns strictly validated JSON.
    Includes automated retry on JSON parsing failure.
    """
    from google import genai
    from google.genai import types

    key = resolve_api_key(api_key)
    if not key:
        raise ValueError(
            "GEMINI_API_KEY is not set. Please provide your Gemini API key in the sidebar, "
            "in .streamlit/secrets.toml, or via the GEMINI_API_KEY environment variable."
        )

    client = genai.Client(api_key=key)

    # Wrap image bytes into types.Part
    image_part = types.Part.from_bytes(data=image_bytes, mime_type=mime_type)
    prompt_text = (
        "Carefully inspect this medical document (prescription or lab report). "
        "Extract all details into STRICT JSON following your instructions."
    )

    config = types.GenerateContentConfig(
        system_instruction=EXTRACTION_SYSTEM_PROMPT,
        response_mime_type="application/json",
        temperature=0.1
    )

    try:
        response = client.models.generate_content(
            model=model,
            contents=[image_part, prompt_text],
            config=config
        )
        raw_text = response.text or ""
    except Exception as e:
        logger.error(f"Gemini API generation error: {e}")
        raise RuntimeError(f"Gemini Vision call failed: {e}")

    # Step 1: Clean and Parse JSON
    cleaned = clean_json_text(raw_text)
    try:
        parsed = json.loads(cleaned)
        return sanitize_extracted_dict(parsed)
    except json.JSONDecodeError as err:
        logger.warning(f"Initial JSON parsing failed: {err}. Initiating LLM retry...")

        # Step 2: Retry with explicit error feedback to fix bad JSON
        retry_prompt = (
            f"The following output was not valid JSON:\n\n{raw_text}\n\n"
            f"JSON Parse Error: {str(err)}\n\n"
            "Please fix the formatting and return ONLY the valid JSON object conforming to the required schema."
        )
        try:
            retry_response = client.models.generate_content(
                model=model,
                contents=retry_prompt,
                config=types.GenerateContentConfig(
                    system_instruction=EXTRACTION_SYSTEM_PROMPT,
                    response_mime_type="application/json",
                    temperature=0.0
                )
            )
            retry_cleaned = clean_json_text(retry_response.text or "")
            parsed_retry = json.loads(retry_cleaned)
            return sanitize_extracted_dict(parsed_retry)
        except Exception as retry_err:
            logger.error(f"Retry parsing also failed: {retry_err}")
            raise ValueError(
                f"Could not parse valid JSON from LLM output even after retry. Raw response snippet: {raw_text[:200]}"
            )


def explain_medical_data(
    verified_data: Dict[str, Any],
    language: str = "en",
    api_key: Optional[str] = None,
    model: str = DEFAULT_MODEL
) -> Dict[str, Any]:
    """
    Second LLM call: Generates a Class-8 level patient explanation in the selected language
    (en, te, hi) along with 3 targeted questions to ask the doctor.
    """
    from google import genai
    from google.genai import types

    lang_code = (language or "en").lower()
    lang_names = {
        "en": "English",
        "te": "Telugu (తెలుగు)",
        "hi": "Hindi (हिन्दी)"
    }
    target_lang_name = lang_names.get(lang_code, "English")

    key = resolve_api_key(api_key)
    if not key:
        raise ValueError(
            "GEMINI_API_KEY is missing. Please configure your key in the sidebar, "
            "in .streamlit/secrets.toml, or via the GEMINI_API_KEY environment variable."
        )

    client = genai.Client(api_key=key)

    data_json_str = json.dumps(verified_data, indent=2, ensure_ascii=False)

    user_prompt = f"""Target Language: {target_lang_name}

Here is the verified data extracted from the patient's medical document:
{data_json_str}

Please generate an easy-to-read explanation for this patient:
1. Explain in simple, friendly, 8th-grade level {target_lang_name}.
2. Explain each medicine (how & when to take it, what it is generally for) OR each lab test (what it measures and why it matters).
3. If any item was marked as low-confidence or unclear, advise them politely to ask their clinic to confirm.
4. Provide exactly 3 clear, actionable questions that the patient can take and ask their doctor.
5. End with the mandatory phrase: "Please confirm with your doctor." (translated naturally into {target_lang_name}).
"""

    config = types.GenerateContentConfig(
        system_instruction=EXPLANATION_SYSTEM_PROMPT,
        temperature=0.3
    )

    try:
        response = client.models.generate_content(
            model=model,
            contents=user_prompt,
            config=config
        )
        explanation_text = response.text or ""
    except Exception as e:
        logger.error(f"Explanation call failed: {e}")
        raise RuntimeError(f"Gemini Explanation call failed: {e}")

    # Mandatory closing check - ensure phrase is present
    mandatory_closing = "Please confirm with your doctor."
    if mandatory_closing.lower() not in explanation_text.lower():
        explanation_text = explanation_text.strip() + f"\n\n**Note: {mandatory_closing}**"

    return {
        "language": lang_code,
        "language_name": target_lang_name,
        "explanation": explanation_text
    }


def get_demo_explanation(verified_data: Dict[str, Any], language: str = "en") -> str:
    """
    High-fidelity built-in multilingual fallback explanations for instant ideathon demos
    even when testing without an active API key.
    """
    lang = (language or "en").lower()
    doc_type = verified_data.get("type", "prescription")

    if doc_type == "prescription":
        if lang == "te":
            return (
                "### మీ ప్రిస్క్రిప్షన్ యొక్క సులభమైన వివరణ\n\n"
                "డాక్టర్ మీకు అందించిన ప్రిస్క్రిప్షన్ ప్రకారం క్రింది మందుల వివరాలు:\n\n"
                "- **అమోక్సిసిలిన్ (Amoxicillin 500mg):** ఇది ఒక యాంటీబయోటిక్. ఇన్ఫెక్షన్ తగ్గడానికి రోజుకు 3 సార్లు (ఉదయం, మధ్యాహ్నం, రాత్రి) భోజనం చేసిన తర్వాత వేసుకోవాలి. మొత్తం 5 రోజుల కోర్సు పూర్తి చేయడం ముఖ్యం.\n"
                "- **పారాసిటమాల్ (Paracetamol 650mg):** జ్వరం లేదా ఒళ్ళు నొప్పులు ఉన్నప్పుడు మాత్రమే (అవసరమైతే) భోజనం తర్వాత వేసుకోవాలి.\n"
                "- **పాంటోప్రజోల్ (Pantoprazole 40mg):** కడుపులో ఎసిడిటీ రాకుండా ఉండటానికి ఉదయం అల్పాహారానికి ముందు ఖాళీ కడుపుతో వేసుకోవాలి.\n"
                "- **సెటిరిజిన్ (Cetirizine 10mg):** జలుబు, తుమ్ములు లేదా అలర్జీ తగ్గడానికి రాత్రి పడుకునే ముందు వేసుకోవాలి.\n"
                "- **దగ్గు మందు:** లేబుల్ అస్పష్టంగా ఉంది కాబట్టి సరైన బ్రాండ్ పేరును మీ డాక్టర్ వద్ద సరిచూసుకోవాలి.\n\n"
                "### మీ డాక్టర్‌ను అడగవలసిన 3 ప్రశ్నలు:\n"
                "1. ఈ యాంటీబయోటిక్ కోర్సును మధ్యలో జ్వరం తగ్గినా 5 రోజులు పూర్తిగా వాడాలా?\n"
                "2. దగ్గు సిరప్ ఖచ్చితమైన పేరు మరియు రోజుకు ఎన్ని సార్లు తీసుకోవాలో స్పష్టం చేయగలరా?\n"
                "3. పారాసిటమాల్ వేసుకున్నా జ్వరం తగ్గకపోతే ఎప్పుడు మిమ్మల్ని సంప్రదించాలి?\n\n"
                "**దయచేసి మీ వైద్యుడితో ధృవీకరించుకోండి.** (Please confirm with your doctor.)"
            )
        elif lang == "hi":
            return (
                "### आपकी पर्ची (Prescription) की आसान भाषा में समझ\n\n"
                "डॉक्टर द्वारा लिखी गई दवाओं की जानकारी इस प्रकार है:\n\n"
                "- **एमोक्सिसिलिन (Amoxicillin 500mg):** यह एक एंटीबायोटिक दवा है जो संक्रमण को ठीक करती है। इसे दिन में 3 बार (सुबह, दोपहर, रात) खाना खाने के बाद 5 दिनों तक नियमित रूप से लें।\n"
                "- **पैरासिटामोल (Paracetamol 650mg):** यह बुखार और दर्द की दवा है। इसे केवल जरूरत पड़ने पर (जब बुखार या दर्द हो) खाने के बाद लें।\n"
                "- **पेंटोप्राजोल (Pantoprazole 40mg):** यह पेट में गैस और एसिडिटी रोकने की दवा है। इसे सुबह नाश्ते से आधा घंटा पहले खाली पेट लें।\n"
                "- **सिटिरिजिन (Cetirizine 10mg):** यह जुकाम, छींक या एलर्जी से राहत देती है। इसे रात को सोने से पहले लें।\n"
                "- **खांसी की सिरप:** पर्ची पर इसका नाम पूरी तरह स्पष्ट नहीं है, कृपया डॉक्टर से सही नाम अवश्य पूछ लें।\n\n"
                "### अपने डॉक्टर से पूछने के लिए 3 महत्वपूर्ण सवाल:\n"
                "1. क्या आराम मिलने पर भी एंटीबायोटिक का 5 दिन का पूरा कोर्स खत्म करना जरूरी है?\n"
                "2. खांसी की सिरप की सही बोतल और सही खुराक क्या है?\n"
                "3. अगर मुझे किसी दवा से पेट खराब या उल्टी जैसा महसूस हो, तो क्या करना चाहिए?\n\n"
                "**कृपया अपने डॉक्टर से पुष्टि करें।** (Please confirm with your doctor.)"
            )
        else:
            return (
                "### Simple Explanation of Your Prescription\n\n"
                "Here is an easy-to-understand breakdown of the medicines prescribed by your doctor:\n\n"
                "- **Amoxicillin 500mg:** This is an antibiotic that fights bacterial infection. Take it 3 times daily (morning, afternoon, night) after food for 5 full days.\n"
                "- **Paracetamol (Dolo 650mg):** For fever or body ache. Take only when needed (SOS) after food.\n"
                "- **Pantoprazole 40mg:** Protects your stomach from acid and heartburn. Take once daily in the morning on an empty stomach (30 minutes before breakfast).\n"
                "- **Cetirizine 10mg:** Relieves cold symptoms, runny nose, or allergic cough. Take at bedtime.\n"
                "- **Cough Syrup:** The handwriting was unclear on the exact brand name; please verify the bottle with your pharmacist or doctor.\n\n"
                "### 3 Questions to Ask Your Doctor:\n"
                "1. Should I finish the entire 5-day antibiotic course even if my symptoms disappear early?\n"
                "2. Can you confirm the exact brand and dosage for the cough syrup?\n"
                "3. Should I avoid any specific foods while taking these medicines?\n\n"
                "**Please confirm with your doctor.**"
            )
    else:  # Lab report
        if lang == "te":
            return (
                "### మీ ల్యాబ్ నివేదిక (Lab Report) యొక్క సులభమైన వివరణ\n\n"
                "మీ రక్త పరీక్షల ఫలితాల వివరాలు క్రింది విధంగా ఉన్నాయి:\n\n"
                "- **హిమోగ్లోబిన్ (Hemoglobin 10.8 g/dL):** సాధారణ స్థాయి (12-17.5) కంటే కొంచెం తక్కువగా ఉంది. ఇది రక్తహీనత (Anemia) సంకేతం కావచ్చు.\n"
                "- **ఫాస్టింగ్ బ్లడ్ షుగర్ (128 mg/dL) & HbA1c (7.2%):** ఇవి సాధారణ పరిమితుల కంటే ఎక్కువగా ఉన్నాయి. గత 3 నెలల సగటు చక్కెర స్థాయిలు ఎక్కువగా ఉన్నట్లు సూచిస్తున్నాయి.\n"
                "- **సీరం క్రియాటినిన్ (0.9 mg/dL):** ఇది సాధారణ పరిధిలో ఉంది. మీ మూత్రపిండాల పనితీరు బాగుంది.\n"
                "- **టోటల్ కొలెస్ట్రాల్ (215 mg/dL):** సాధారణ స్థాయి (200) కంటే కొద్దిగా ఎక్కువగా ఉంది.\n"
                "- **TSH (2.8 mIU/L) & ప్లేట్‌లెట్లు (1.9 లక్షలు):** ఇవి సాధారణ పరిధిలో ఆరోగ్యంగా ఉన్నాయి.\n\n"
                "### మీ డాక్టర్‌ను అడగవలసిన 3 ప్రశ్నలు:\n"
                "1. నా షుగర్ మరియు HbA1c స్థాయిలను నియంత్రించడానికి నేను ఎటువంటి ఆహార నియమాలు పాటించాలి?\n"
                "2. హిమోగ్లోబిన్ పెరగడానికి ఐరన్ సప్లిమెంట్స్ లేదా ప్రత్యేక ఆహారం అవసరమా?\n"
                "3. కొలెస్ట్రాల్ మరియు షుగర్ నియంత్రణకు వ్యాయామం లేదా మందులు ప్రారంభించాలా?\n\n"
                "**దయచేసి మీ వైద్యుడితో ధృవీకరించుకోండి.** (Please confirm with your doctor.)"
            )
        elif lang == "hi":
            return (
                "### आपकी लैब रिपोर्ट (Lab Report) की आसान व्याख्या\n\n"
                "आपके रक्त परीक्षण के मुख्य परिणाम इस प्रकार हैं:\n\n"
                "- **हीमोग्लोबिन (Hemoglobin 10.8 g/dL):** यह सामान्य सीमा (12-17.5) से थोड़ा कम है, जो हल्के खून की कमी (एनीमिया) का संकेत हो सकता है।\n"
                "- **खाली पेट शुगर (128 mg/dL) और HbA1c (7.2%):** ये दोनों सामान्य स्तर से अधिक हैं। HbA1c दर्शाता है कि पिछले 3 महीनों का औसत शुगर स्तर बढ़ा हुआ है।\n"
                "- **सीरम क्रिएटिनिन (0.9 mg/dL):** यह बिल्कुल सामान्य है, जिसका अर्थ है कि आपकी किडनी सही तरीके से काम कर रही है।\n"
                "- **कोलेस्ट्रॉल (215 mg/dL):** यह सामान्य सीमा (200 से कम) से थोड़ा अधिक है।\n"
                "- **थायराइड (TSH 2.8) और प्लेटलेट्स (1.9 लाख):** ये दोनों पूरी तरह सामान्य और स्वस्थ सीमा में हैं।\n\n"
                "### अपने डॉक्टर से पूछने के लिए 3 जरूरी सवाल:\n"
                "1. बढ़े हुए ब्लड शुगर और HbA1c को नियंत्रित करने के लिए मुझे खान-पान में क्या बदलाव करने चाहिए?\n"
                "2. क्या हीमोग्लोबिन सुधारने के लिए मुझे आयरन की गोलियां या सिरप लेने की आवश्यकता है?\n"
                "3. क्या कोलेस्ट्रॉल और शुगर के लिए कोई नियमित दवा शुरू करने की जरूरत है?\n\n"
                "**कृपया अपने डॉक्टर से पुष्टि करें।** (Please confirm with your doctor.)"
            )
        else:
            return (
                "### Simple Explanation of Your Lab Report\n\n"
                "Here is an easy-to-understand summary of your laboratory blood test results:\n\n"
                "- **Hemoglobin (10.8 g/dL):** Slightly below normal range (12.0 - 17.5 g/dL). This may indicate mild anemia or lower red blood cell count.\n"
                "- **Fasting Blood Sugar (128 mg/dL) & HbA1c (7.2%):** Both values are elevated above normal limits. HbA1c indicates that average blood sugar over the past 3 months has been higher than normal.\n"
                "- **Serum Creatinine (0.9 mg/dL):** Comfortably within normal range (0.6 - 1.2 mg/dL), indicating healthy kidney filtration.\n"
                "- **Total Cholesterol (215 mg/dL):** Borderline high (desirable is below 200 mg/dL).\n"
                "- **TSH (2.8 mIU/L) & Platelet Count (190,000 /mcL):** Both are well within normal, healthy limits.\n\n"
                "### 3 Questions to Ask Your Doctor:\n"
                "1. What dietary and lifestyle adjustments should I make to bring my blood sugar and HbA1c into a healthy range?\n"
                "2. Do I need iron-rich foods or supplements to improve my hemoglobin level?\n"
                "3. When should I repeat these blood tests to monitor progress?\n\n"
                "**Please confirm with your doctor.**"
            )
