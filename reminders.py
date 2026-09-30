"""
Asclepius - Medical Schedule & Calendar Reminders Generator
Converts medicine frequency and timing instructions into actionable daily schedules
and RFC 5545 compliant .ics calendar files.
"""

import re
import uuid
from datetime import datetime, timedelta
from typing import List, Dict, Any, Tuple


def parse_duration_days(duration_val: Any) -> int:
    """
    Parse duration into integer days. Defaults to 5 days if unspecified.
    """
    if not duration_val:
        return 5
    s = str(duration_val).lower().strip()
    # Check for months
    month_match = re.search(r"(\d+)\s*(?:month|mo)", s)
    if month_match:
        return int(month_match.group(1)) * 30
    # Check for weeks
    week_match = re.search(r"(\d+)\s*(?:week|wk)", s)
    if week_match:
        return int(week_match.group(1)) * 7
    # Check for days
    day_match = re.search(r"(\d+)\s*(?:day|d)?", s)
    if day_match:
        days = int(day_match.group(1))
        return days if days > 0 else 5
    return 5


def parse_timing_note(timing_str: Any) -> str:
    """
    Extract human-readable dietary relation (before food, after food, with meals, etc.)
    """
    if not timing_str:
        return "As advised by doctor"
    s = str(timing_str).lower().strip()
    if any(k in s for k in ["before food", "empty stomach", "before meal", "before meals", "ac", "fasting"]):
        return "Before food (30 mins prior)"
    if any(k in s for k in ["after food", "after meal", "after meals", "pc", "with food", "with meal"]):
        return "After food (within 30 mins after meal)"
    if "bedtime" in s or "night" in s or "hs" in s:
        return "At bedtime"
    return timing_str.strip()


def parse_frequency_to_times(frequency_str: Any, timing_str: Any = "") -> List[Tuple[str, str, int]]:
    """
    Convert medical frequency notation to a list of (Time string, Slot name, Sort order integer).
    Returns list of tuples: (formatted_time, slot_name, sort_order)
    """
    freq = (str(frequency_str) if frequency_str else "").lower().strip()
    timing = (str(timing_str) if timing_str else "").lower().strip()

    # Pattern 1-0-1 or 1-1-1 or 1-0-0 or 0-0-1
    pattern_match = re.search(r"([012])\s*[-–/]\s*([012])\s*[-–/]\s*([012])", freq)
    if pattern_match:
        m, a, n = int(pattern_match.group(1)), int(pattern_match.group(2)), int(pattern_match.group(3))
        times = []
        if m > 0:
            times.append(("08:00 AM", "Morning", 800))
        if a > 0:
            times.append(("02:00 PM", "Afternoon", 1400))
        if n > 0:
            times.append(("08:30 PM", "Night", 2030))
        if times:
            return times

    # 4 times daily / QID / 1-1-1-1
    if any(k in freq for k in ["qid", "four times", "4 times", "every 6 hours", "1-1-1-1"]):
        return [
            ("08:00 AM", "Morning", 800),
            ("12:30 PM", "Afternoon", 1230),
            ("05:00 PM", "Evening", 1700),
            ("09:30 PM", "Night", 2130)
        ]

    # 3 times daily / TDS / TID / 1-1-1
    if any(k in freq for k in ["tds", "tid", "thrice", "three times", "3 times", "every 8 hours"]):
        return [
            ("08:00 AM", "Morning", 800),
            ("02:00 PM", "Afternoon", 1400),
            ("08:30 PM", "Night", 2030)
        ]

    # 2 times daily / BD / BID / 1-0-1
    if any(k in freq for k in ["bd", "bid", "twice", "two times", "2 times", "every 12 hours"]):
        return [
            ("08:00 AM", "Morning", 800),
            ("08:30 PM", "Night", 2030)
        ]

    # Bedtime / HS / Night only
    if any(k in freq for k in ["hs", "bedtime", "at night", "night"]) or any(k in timing for k in ["hs", "bedtime", "night"]):
        return [
            ("09:30 PM", "Night / Bedtime", 2130)
        ]

    # Once daily / OD / QD / 1-0-0
    if any(k in freq for k in ["od", "qd", "once", "daily", "every 24 hours", "one time"]):
        if any(k in timing for k in ["night", "bedtime", "dinner", "hs"]):
            return [("09:00 PM", "Night", 2100)]
        if any(k in timing for k in ["afternoon", "lunch"]):
            return [("01:30 PM", "Afternoon", 1330)]
        return [("08:00 AM", "Morning", 800)]

    # SOS / As needed
    if any(k in freq for k in ["sos", "prn", "as needed", "when required", "if needed"]):
        return [("As Needed", "Flexible", 9999)]

    # Default fallback
    if "night" in timing:
        return [("09:00 PM", "Night", 2100)]
    return [("08:00 AM", "Morning", 800)]


def build_daily_schedule(medicines: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Builds a sorted daily medicine timetable from extracted medicine records.
    """
    schedule_items = []

    for med in medicines:
        name = med.get("name") or "Unspecified Medicine"
        dose = med.get("dose") or "As prescribed"
        freq = med.get("frequency") or "Once daily"
        timing_raw = med.get("timing") or ""
        timing_note = parse_timing_note(timing_raw)
        duration_days = parse_duration_days(med.get("duration_days"))
        confidence = med.get("confidence", "high").lower()

        times = parse_frequency_to_times(freq, timing_raw)
        for time_str, slot_name, order_val in times:
            schedule_items.append({
                "time": time_str,
                "slot": slot_name,
                "order": order_val,
                "medicine": name,
                "dose": dose,
                "frequency": freq,
                "timing_note": timing_note,
                "duration_days": duration_days,
                "confidence": confidence
            })

    # Sort primarily by time of day
    schedule_items.sort(key=lambda x: (x["order"], x["medicine"]))
    return schedule_items


def generate_ics_calendar(medicines: List[Dict[str, Any]], start_date: datetime = None) -> str:
    """
    Generates an RFC 5545 compliant .ics calendar string containing recurring
    daily reminders with 10-minute alarm popups for each medicine dose.
    """
    if start_date is None:
        start_date = datetime.now()

    # Base date starting tomorrow morning 00:00:00 for clean calendar syncing
    base_date = (start_date + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
    now_stamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")

    ics_lines = [
        "BEGIN:VCALENDAR",
        "VERSION:2.0",
        "PRODID:-//Asclepius Health//Medical Reminders//EN",
        "CALSCALE:GREGORIAN",
        "METHOD:PUBLISH",
        "X-WR-CALNAME:Asclepius Medicine Schedule",
        "X-WR-TIMEZONE:UTC"
    ]

    for med in medicines:
        name = med.get("name") or "Medicine"
        dose = med.get("dose") or "As prescribed"
        freq = med.get("frequency") or "Once daily"
        timing_raw = med.get("timing") or ""
        timing_note = parse_timing_note(timing_raw)
        duration_days = parse_duration_days(med.get("duration_days"))
        times = parse_frequency_to_times(freq, timing_raw)

        for time_str, slot, _ in times:
            if time_str == "As Needed":
                # Do not set fixed recurring alarm for SOS drugs, but provide one informational event
                event_uid = f"{uuid.uuid4()}@asclepius.health"
                dt_start = base_date.replace(hour=9, minute=0).strftime("%Y%m%dT%H%M%S")
                dt_end = base_date.replace(hour=9, minute=15).strftime("%Y%m%dT%H%M%S")

                ics_lines.extend([
                    "BEGIN:VEVENT",
                    f"UID:{event_uid}",
                    f"DTSTAMP:{now_stamp}",
                    f"DTSTART:{dt_start}",
                    f"DTEND:{dt_end}",
                    f"SUMMARY:SOS Medicine: {name} ({dose})",
                    f"DESCRIPTION:Take only as needed when symptoms occur.\\nInstruction: {timing_note}",
                    "STATUS:CONFIRMED",
                    "END:VEVENT"
                ])
                continue

            # Parse hour and minute from time_str, e.g. "08:00 AM" or "02:00 PM"
            try:
                t_obj = datetime.strptime(time_str, "%I:%M %p").time()
                event_start = base_date.replace(hour=t_obj.hour, minute=t_obj.minute)
                event_end = event_start + timedelta(minutes=15)

                dt_start_str = event_start.strftime("%Y%m%dT%H%M%S")
                dt_end_str = event_end.strftime("%Y%m%dT%H%M%S")
                event_uid = f"{uuid.uuid4()}@asclepius.health"

                ics_lines.extend([
                    "BEGIN:VEVENT",
                    f"UID:{event_uid}",
                    f"DTSTAMP:{now_stamp}",
                    f"DTSTART:{dt_start_str}",
                    f"DTEND:{dt_end_str}",
                    f"RRULE:FREQ=DAILY;COUNT={duration_days}",
                    f"SUMMARY:Take {name} - {dose}",
                    f"DESCRIPTION:Medicine: {name} ({dose})\\nTiming: {timing_note}\\nFrequency: {freq}\\nCourse: {duration_days} days.\\nAlways consult your doctor for dosage queries.",
                    "STATUS:CONFIRMED",
                    "BEGIN:VALARM",
                    "TRIGGER:-PT10M",
                    "ACTION:DISPLAY",
                    f"DESCRIPTION:Reminder: Time to take {name} ({dose})",
                    "END:VALARM",
                    "END:VEVENT"
                ])
            except Exception:
                continue

    ics_lines.append("END:VCALENDAR")
    return "\r\n".join(ics_lines)
