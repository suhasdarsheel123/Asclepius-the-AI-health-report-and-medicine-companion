"""
End-to-end integration test for Asclepius modules.
"""

import json
import os
import io
import ranges
import reminders
import extractor
from gtts import gTTS

def test_ranges_evaluation():
    print("Testing ranges.py...")
    # Low hemoglobin
    res1 = ranges.evaluate_lab_value("Hemoglobin", "10.5", "g/dL")
    assert res1["status"] == "Low", f"Expected Low, got {res1['status']}"
    assert res1["badge_color"] == "#2563eb"

    # High fasting sugar
    res2 = ranges.evaluate_lab_value("Fasting Blood Sugar", "135", "mg/dL")
    assert res2["status"] == "High", f"Expected High, got {res2['status']}"
    assert res2["badge_color"] == "#dc2626"

    # Normal creatinine
    res3 = ranges.evaluate_lab_value("Serum Creatinine", "0.9", "mg/dL")
    assert res3["status"] == "Normal", f"Expected Normal, got {res3['status']}"
    assert res3["badge_color"] == "#16a34a"

    # High HbA1c
    res4 = ranges.evaluate_lab_value("HbA1c", "7.4", "%")
    assert res4["status"] == "High", f"Expected High, got {res4['status']}"

    # Unknown test
    res5 = ranges.evaluate_lab_value("NonExistentTest123", "99")
    assert res5["status"] == "Unknown"

    print("✓ ranges.py tests passed successfully!")

def test_reminders_and_ics():
    print("Testing reminders.py...")
    fixture_path = os.path.join(os.path.dirname(__file__), "fixtures", "sample_prescription.json")
    with open(fixture_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    schedule = reminders.build_daily_schedule(data["medicines"])
    assert len(schedule) > 0, "Schedule should contain items"
    print(f"Generated {len(schedule)} schedule dose slots.")

    ics_content = reminders.generate_ics_calendar(data["medicines"])
    assert "BEGIN:VCALENDAR" in ics_content
    assert "END:VCALENDAR" in ics_content
    assert "BEGIN:VALARM" in ics_content
    assert "RRULE:FREQ=DAILY" in ics_content
    print("✓ reminders.py tests passed successfully!")

def test_extractor_demo_and_tts():
    print("Testing extractor.py fallbacks and gTTS in English, Telugu, Hindi...")
    fixture_prescription = os.path.join(os.path.dirname(__file__), "fixtures", "sample_prescription.json")
    with open(fixture_prescription, "r", encoding="utf-8") as f:
        presc_data = json.load(f)

    for lang in ["en", "te", "hi"]:
        exp = extractor.get_demo_explanation(presc_data, lang)
        assert len(exp) > 50
        assert "doctor" in exp.lower() or "వైద్యుడితో" in exp or "डॉक्टर" in exp

        # Test TTS synthesis
        fp = io.BytesIO()
        gTTS(text=exp[:100], lang=lang).write_to_fp(fp)
        assert fp.tell() > 1000, f"gTTS output for {lang} too small"
        print(f"✓ Language '{lang}' explanation and voice generation passed.")

    fixture_lab = os.path.join(os.path.dirname(__file__), "fixtures", "sample_lab_report.json")
    with open(fixture_lab, "r", encoding="utf-8") as f:
        lab_data = json.load(f)

    for lang in ["en", "te", "hi"]:
        exp_lab = extractor.get_demo_explanation(lab_data, lang)
        assert len(exp_lab) > 50

    print("✓ extractor.py fallback & TTS tests passed successfully!")

if __name__ == "__main__":
    test_ranges_evaluation()
    test_reminders_and_ics()
    test_extractor_demo_and_tts()
    print("\n🎉 ALL TESTS PASSED SUCCESSFULLY!")
