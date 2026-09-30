"""
Asclepius - Lab Reference Ranges and Deterministic Flagging Engine
Evaluates laboratory test values against standard clinical reference ranges using code (NOT LLM).
"""

import re
from typing import Optional, Dict, Any, Tuple

# Comprehensive dictionary of clinical reference ranges
# Standard adult reference ranges (General guidelines)
REFERENCE_RANGES: Dict[str, Dict[str, Any]] = {
    "hemoglobin": {
        "name": "Hemoglobin (Hb)",
        "aliases": ["hemoglobin", "hb", "haemoglobin", "hgb", "total hemoglobin"],
        "min": 12.0,
        "max": 17.5,
        "unit": "g/dL",
        "description": "Oxygen-carrying protein in red blood cells. Low values may indicate anemia."
    },
    "fasting_glucose": {
        "name": "Fasting Blood Sugar (Glucose)",
        "aliases": ["fasting blood sugar", "fbs", "fasting glucose", "glucose fasting", "blood sugar fasting", "glucose, fasting"],
        "min": 70.0,
        "max": 100.0,
        "unit": "mg/dL",
        "description": "Blood sugar level after 8-12 hours of overnight fasting."
    },
    "postprandial_glucose": {
        "name": "Postprandial Blood Sugar (PPBS)",
        "aliases": ["ppbs", "postprandial blood sugar", "post prandial glucose", "glucose post prandial", "pp glucose", "2hr post glucose"],
        "min": 70.0,
        "max": 140.0,
        "unit": "mg/dL",
        "description": "Blood sugar level measured 2 hours after having a meal."
    },
    "random_glucose": {
        "name": "Random Blood Sugar (RBS)",
        "aliases": ["random blood sugar", "rbs", "random glucose", "blood sugar random", "glucose, random"],
        "min": 70.0,
        "max": 140.0,
        "unit": "mg/dL",
        "description": "Blood sugar taken at any time regardless of when meals were consumed."
    },
    "hba1c": {
        "name": "HbA1c (Glycated Hemoglobin)",
        "aliases": ["hba1c", "glycated hemoglobin", "glycosylated hemoglobin", "a1c", "hemoglobin a1c"],
        "min": 4.0,
        "max": 5.6,
        "unit": "%",
        "description": "Reflects average blood sugar control over the past 2 to 3 months. Normal is below 5.7%."
    },
    "creatinine": {
        "name": "Serum Creatinine",
        "aliases": ["creatinine", "serum creatinine", "s. creatinine", "sr creatinine", "blood creatinine"],
        "min": 0.6,
        "max": 1.2,
        "unit": "mg/dL",
        "description": "Waste product filtered out by kidneys; key indicator of kidney filtration function."
    },
    "bun": {
        "name": "Blood Urea Nitrogen (BUN)",
        "aliases": ["bun", "blood urea nitrogen", "urea nitrogen", "serum urea", "blood urea"],
        "min": 7.0,
        "max": 20.0,
        "unit": "mg/dL",
        "description": "Measures the amount of urea nitrogen in blood, aiding kidney function assessment."
    },
    "cholesterol_total": {
        "name": "Total Cholesterol",
        "aliases": ["total cholesterol", "cholesterol total", "serum cholesterol", "cholesterol"],
        "min": 125.0,
        "max": 200.0,
        "unit": "mg/dL",
        "description": "Total amount of cholesterol in blood. Desirable level is below 200 mg/dL."
    },
    "cholesterol_ldl": {
        "name": "LDL Cholesterol ('Bad' Cholesterol)",
        "aliases": ["ldl", "ldl cholesterol", "cholesterol ldl", "low density lipoprotein", "bad cholesterol"],
        "min": 50.0,
        "max": 100.0,
        "unit": "mg/dL",
        "description": "Low-density lipoprotein. Lower levels are generally healthier for cardiovascular health."
    },
    "cholesterol_hdl": {
        "name": "HDL Cholesterol ('Good' Cholesterol)",
        "aliases": ["hdl", "hdl cholesterol", "cholesterol hdl", "high density lipoprotein", "good cholesterol"],
        "min": 40.0,
        "max": 60.0,
        "unit": "mg/dL",
        "description": "High-density lipoprotein. Helps remove other forms of cholesterol from bloodstream."
    },
    "triglycerides": {
        "name": "Triglycerides",
        "aliases": ["triglycerides", "serum triglycerides", "tg", "tgl"],
        "min": 50.0,
        "max": 150.0,
        "unit": "mg/dL",
        "description": "Type of fat (lipid) found in blood. High levels can indicate cardiovascular risks."
    },
    "tsh": {
        "name": "Thyroid Stimulating Hormone (TSH)",
        "aliases": ["tsh", "thyroid stimulating hormone", "s.tsh", "serum tsh", "ultrasensitive tsh"],
        "min": 0.4,
        "max": 4.5,
        "unit": "mIU/L",
        "description": "Produced by pituitary gland to stimulate thyroid gland; regulates metabolism."
    },
    "platelets": {
        "name": "Platelet Count",
        "aliases": ["platelets", "platelet count", "plt", "total platelets", "thrombocytes"],
        "min": 150000.0,
        "max": 450000.0,
        "unit": "/mcL",
        "description": "Blood cells that help blood clot and prevent bleeding."
    },
    "wbc": {
        "name": "Total White Blood Cells (WBC / TLC)",
        "aliases": ["wbc", "tlc", "white blood cells", "total leukocyte count", "leukocyte count", "white blood cell count"],
        "min": 4000.0,
        "max": 11000.0,
        "unit": "/mcL",
        "description": "Immune cells defending against infections. Elevated in infections or inflammation."
    },
    "rbc": {
        "name": "Red Blood Cell Count (RBC)",
        "aliases": ["rbc", "red blood cell count", "total rbc", "erythrocytes"],
        "min": 4.2,
        "max": 5.9,
        "unit": "million/mcL",
        "description": "Cells that carry oxygen from lungs to rest of the body."
    },
    "total_bilirubin": {
        "name": "Total Bilirubin",
        "aliases": ["total bilirubin", "bilirubin total", "serum bilirubin", "s. bilirubin"],
        "min": 0.2,
        "max": 1.2,
        "unit": "mg/dL",
        "description": "Yellowish substance made during normal breakdown of red blood cells; liver marker."
    },
    "sgpt_alt": {
        "name": "ALT (SGPT)",
        "aliases": ["sgpt", "alt", "alanine aminotransferase", "alanine transaminase", "sgpt (alt)"],
        "min": 7.0,
        "max": 56.0,
        "unit": "U/L",
        "description": "Enzyme primarily produced in liver cells; elevated when liver tissue is stressed."
    },
    "sgot_ast": {
        "name": "AST (SGOT)",
        "aliases": ["sgot", "ast", "aspartate aminotransferase", "aspartate transaminase", "sgot (ast)"],
        "min": 10.0,
        "max": 40.0,
        "unit": "U/L",
        "description": "Enzyme found in liver, heart, and muscle cells."
    },
    "alkaline_phosphatase": {
        "name": "Alkaline Phosphatase (ALP)",
        "aliases": ["alp", "alkaline phosphatase", "alk phos", "s. alp"],
        "min": 44.0,
        "max": 147.0,
        "unit": "U/L",
        "description": "Enzyme related to bile ducts and bone turnover."
    },
    "uric_acid": {
        "name": "Serum Uric Acid",
        "aliases": ["uric acid", "serum uric acid", "s. uric acid"],
        "min": 3.5,
        "max": 7.2,
        "unit": "mg/dL",
        "description": "End product of purine metabolism; elevated levels can form crystals causing gout."
    },
    "sodium": {
        "name": "Serum Sodium (Na+)",
        "aliases": ["sodium", "serum sodium", "s. sodium", "na+"],
        "min": 135.0,
        "max": 145.0,
        "unit": "mmol/L",
        "description": "Essential electrolyte maintaining blood volume, fluid balance, and nerve firing."
    },
    "potassium": {
        "name": "Serum Potassium (K+)",
        "aliases": ["potassium", "serum potassium", "s. potassium", "k+"],
        "min": 3.5,
        "max": 5.0,
        "unit": "mmol/L",
        "description": "Critical electrolyte for cardiac rhythm and muscle contractility."
    },
    "calcium": {
        "name": "Serum Calcium",
        "aliases": ["calcium", "serum calcium", "s. calcium", "ca++"],
        "min": 8.5,
        "max": 10.2,
        "unit": "mg/dL",
        "description": "Mineral critical for bone strength, nerve signaling, and muscle movement."
    },
    "vitamin_d": {
        "name": "Vitamin D (25-OH)",
        "aliases": ["vitamin d", "25-hydroxy vitamin d", "vit d", "vitamin d3", "25-oh vitamin d"],
        "min": 30.0,
        "max": 100.0,
        "unit": "ng/mL",
        "description": "Fat-soluble vitamin supporting calcium absorption and immune system."
    },
    "vitamin_b12": {
        "name": "Vitamin B12",
        "aliases": ["vitamin b12", "vit b12", "b12", "cobalamin"],
        "min": 200.0,
        "max": 900.0,
        "unit": "pg/mL",
        "description": "Essential vitamin needed for red blood cell formation and neurological health."
    },
    "esr": {
        "name": "ESR (Erythrocyte Sedimentation Rate)",
        "aliases": ["esr", "erythrocyte sedimentation rate", "sed rate"],
        "min": 0.0,
        "max": 20.0,
        "unit": "mm/hr",
        "description": "Rate at which red blood cells settle; general indicator of systemic inflammation."
    }
}


def parse_numeric_value(val: Any) -> Optional[float]:
    """
    Extract a numeric float from various string formats (e.g. '105.4', '> 140', '12 g/dL', '1.5 Lakh').
    """
    if val is None:
        return None
    if isinstance(val, (int, float)):
        return float(val)

    s = str(val).strip().replace(",", "")
    # Check for lakh multiplier (common in Indian subcontinent lab reports, e.g. 2.5 Lakh platelets)
    is_lakh = "lakh" in s.lower() or "lac" in s.lower()

    # Match numeric portion (handles decimals, negatives)
    match = re.search(r"[-+]?\d*\.?\d+", s)
    if not match:
        return None

    try:
        num = float(match.group())
        if is_lakh and num < 100:
            num = num * 100000.0
        return num
    except (ValueError, TypeError):
        return None


def match_reference_range(test_name: str) -> Optional[Tuple[str, Dict[str, Any]]]:
    """
    Find matching reference range using normalized test names and alias search.
    """
    if not test_name:
        return None

    # Normalize query string
    cleaned = re.sub(r"[^a-zA-Z0-9\s]", " ", test_name).lower().strip()
    words = cleaned.split()

    # 1. Exact alias match
    for key, info in REFERENCE_RANGES.items():
        for alias in info["aliases"]:
            clean_alias = re.sub(r"[^a-zA-Z0-9\s]", " ", alias).lower().strip()
            if cleaned == clean_alias:
                return key, info

    # 2. Substring or token containment match
    for key, info in REFERENCE_RANGES.items():
        for alias in info["aliases"]:
            clean_alias = re.sub(r"[^a-zA-Z0-9\s]", " ", alias).lower().strip()
            if clean_alias in cleaned or cleaned in clean_alias:
                return key, info

    # 3. Keyword check for specific common terms
    keyword_map = {
        "creatinine": "creatinine",
        "hba1c": "hba1c",
        "hemoglobin": "hemoglobin",
        "haemoglobin": "hemoglobin",
        "tsh": "tsh",
        "platelet": "platelets",
        "platelets": "platelets",
        "bilirubin": "total_bilirubin",
        "sgpt": "sgpt_alt",
        "sgot": "sgot_ast",
        "alt": "sgpt_alt",
        "ast": "sgot_ast",
        "fasting": "fasting_glucose",
        "postprandial": "postprandial_glucose",
        "ppbs": "postprandial_glucose",
        "triglyceride": "triglycerides",
        "triglycerides": "triglycerides",
        "cholesterol": "cholesterol_total",
        "urea": "bun",
        "uric": "uric_acid",
        "sodium": "sodium",
        "potassium": "potassium",
        "calcium": "calcium",
        "b12": "vitamin_b12",
        "vitamin d": "vitamin_d",
        "wbc": "wbc",
        "tlc": "wbc",
        "esr": "esr"
    }
    for word in words:
        if word in keyword_map:
            matched_key = keyword_map[word]
            return matched_key, REFERENCE_RANGES[matched_key]

    return None


def evaluate_lab_value(test_name: str, value_str: Any, unit_str: Optional[str] = None) -> Dict[str, Any]:
    """
    Deterministic code-based clinical evaluation of a lab test value against reference ranges.
    Returns status: 'Normal', 'Low', 'High', or 'Unknown', with UI badge colors and range strings.
    """
    parsed_num = parse_numeric_value(value_str)
    matched = match_reference_range(test_name)

    if not matched or parsed_num is None:
        return {
            "matched_key": None,
            "display_name": test_name or "Unknown Test",
            "raw_value": str(value_str) if value_str is not None else "",
            "numeric_value": parsed_num,
            "unit": unit_str or "",
            "status": "Unknown",
            "status_badge": "⚪ Unmatched",
            "badge_color": "gray",
            "ref_min": None,
            "ref_max": None,
            "ref_range_str": "Standard range not in database",
            "description": "Consult your laboratory report's listed reference interval."
        }

    key, info = matched
    ref_min = info["min"]
    ref_max = info["max"]
    expected_unit = info["unit"]
    display_unit = unit_str if unit_str else expected_unit

    ref_range_str = f"{ref_min} - {ref_max} {expected_unit}"

    if parsed_num < ref_min:
        status = "Low"
        status_badge = "🔻 Low"
        badge_color = "#2563eb"  # Blue
    elif parsed_num > ref_max:
        status = "High"
        status_badge = "🔺 High"
        badge_color = "#dc2626"  # Red
    else:
        status = "Normal"
        status_badge = "🟢 Normal"
        badge_color = "#16a34a"  # Green

    return {
        "matched_key": key,
        "display_name": info["name"],
        "raw_value": str(value_str),
        "numeric_value": parsed_num,
        "unit": display_unit,
        "status": status,
        "status_badge": status_badge,
        "badge_color": badge_color,
        "ref_min": ref_min,
        "ref_max": ref_max,
        "ref_range_str": ref_range_str,
        "description": info["description"]
    }
