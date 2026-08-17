"""Module-level lexicon constants. Loaded once; no file I/O per call."""
import re

# Attorney / legal representation
ATTORNEY_PATTERNS = [
    r"\bretained counsel\b",
    r"\bretain counsel\b",
    r"\bletter of representation\b",
    r"\bLOR\b",
    r"\bcease\s+(?:and\s+desist|communication|contact)\b",
    r"\battorney\s+(?:representing|for the)\b",
    r"\bmy attorney\b",
    r"\blaw firm\b",
    r"\bdemand letter\b",
    r"\brepresented by\b",
    r"\bour client'?s?\b",
    r"\bbodily injury claim\b",
]

# SIU red flags
SIU_PATTERNS = [
    r"\bno police report\b",
    r"\bunable to locate\b",
    r"\bcash[-\s]only\b",
    r"\bno receipt\b",
    r"\bno witness\b",
    r"\bwitness unavailable\b",
    r"\bprior claim\b",
    r"\brecent policy\b",
    r"\bstaged[-\s]loss\b",
    r"\bstaged\s+loss\b",
    r"\bsuspicious\b",
    r"\bunder investigation\b",
]

# Medical severity
MEDICAL_PATTERNS = [
    r"\bsurgery\b",
    r"\bsurgical\b",
    r"\bMRI\b",
    r"\bCT scan\b",
    r"\bER (?:visit)?\b",
    r"\bemergency room\b",
    r"\bphysical therapy\b",
    r"\bchiropractor\b",
    r"\bpermanent (?:injury|disability)\b",
    r"\bhospitalization\b",
    r"\binpatient\b",
    r"\borthopedic\b",
]

# Subrogation opportunity
SUBROGATION_PATTERNS = [
    r"\bthird[-\s]party\b",
    r"\bat fault\b",
    r"\bother driver\b",
    r"\bnegligent\b",
    r"\bdefective\b",
    r"\bmanufacturer\b",
    r"\bproduct liability\b",
    r"\bsubrogation\b",
]

# Urgency
URGENCY_WORDS = ["URGENT", "ASAP", "IMMEDIATELY", "EXPEDITE", "TIME-SENSITIVE"]

# Health disclosures (PHI)
HEALTH_PATTERNS = [
    r"\bcancer\b",
    r"\bchemotherapy\b",
    r"\bimmunocompromised\b",
    r"\bpregnant\b",
    r"\bdisabled?\b",
    r"\bdisability\b",
    r"\bmedication\b",
    r"\bdiabetes\b",
]

DOLLAR_PATTERN = r"\$\s?(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)"


def _compile_all(patterns):
    return [re.compile(p, re.IGNORECASE) for p in patterns]


COMPILED = {
    "attorney":    _compile_all(ATTORNEY_PATTERNS),
    "siu":         _compile_all(SIU_PATTERNS),
    "medical":     _compile_all(MEDICAL_PATTERNS),
    "subrogation": _compile_all(SUBROGATION_PATTERNS),
    "health":      _compile_all(HEALTH_PATTERNS),
    "dollar":      re.compile(DOLLAR_PATTERN),
}
