"""Regex/dictionary-based text feature extractors."""
import re
from claims_fe.lexicons import COMPILED, URGENCY_WORDS


def _count_matches(text: str, compiled_list) -> int:
    if not text:
        return 0
    return sum(1 for r in compiled_list if r.search(text))


def _count_all_matches(text: str, compiled_list) -> int:
    """Total match count across all patterns (not just number of patterns that hit)."""
    if not text:
        return 0
    total = 0
    for r in compiled_list:
        total += len(r.findall(text))
    return total


def _urgency_score(text: str) -> float:
    if not text:
        return 0.0
    length = max(len(text), 1)
    urgent_hits = sum(text.count(w) for w in URGENCY_WORDS)
    exclamation_count = text.count("!")
    # Caps ratio over alphabetic characters only
    alpha = [c for c in text if c.isalpha()]
    caps_ratio = (sum(1 for c in alpha if c.isupper()) / len(alpha)) if alpha else 0.0

    score = min(
        1.0,
        urgent_hits * 0.15
        + (exclamation_count / (length / 1000)) * 0.01
        + caps_ratio * 0.3,
    )
    return round(float(score), 4)


def _dollar_amounts(text: str):
    if not text:
        return []
    matches = COMPILED["dollar"].findall(text)
    amounts = []
    for m in matches:
        try:
            amounts.append(float(m.replace(",", "")))
        except ValueError:
            continue
    return amounts


def text_flag_features(claim: dict) -> dict:
    # Combine all text sources (notes carry most signal; incidents + docs are shorter)
    notes = claim.get("ConcatenatedNotes") or ""
    docs = claim.get("ConcatenatedDocs") or ""
    incidents = claim.get("ConcatenatedIncidents") or ""
    combined = " ".join(filter(None, [notes, incidents, docs]))

    dollar_amounts = _dollar_amounts(combined)

    return {
        "attorney_mentioned": _count_matches(combined, COMPILED["attorney"]) > 0,
        "attorney_phrase_count": _count_all_matches(combined, COMPILED["attorney"]),
        "siu_red_flag_count": _count_matches(combined, COMPILED["siu"]),
        "medical_severity_flag": _count_matches(combined, COMPILED["medical"]) > 0,
        "subrogation_opportunity_flag": _count_matches(combined, COMPILED["subrogation"]) > 0,
        "health_disclosure_flag": _count_matches(combined, COMPILED["health"]) > 0,
        "urgency_score": _urgency_score(combined),
        "max_dollar_amount_mentioned": max(dollar_amounts) if dollar_amounts else None,
        "dollar_amount_count": len(dollar_amounts),
    }
