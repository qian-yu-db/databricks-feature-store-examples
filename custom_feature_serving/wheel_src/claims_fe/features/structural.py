"""Length / exposure / passthrough features."""


def structural_features(claim: dict) -> dict:
    docs = claim.get("ConcatenatedDocs") or ""
    notes = claim.get("ConcatenatedNotes") or ""
    incidents = claim.get("ConcatenatedIncidents") or ""
    desc = claim.get("CLM_DESC_TXT") or ""

    exposures = claim.get("Exposures") or []
    exposure_types = [e.get("EXPSR_TYP_CD") for e in exposures if e.get("EXPSR_TYP_CD")]

    return {
        "desc_char_count": len(desc),
        "docs_char_count": len(docs),
        "notes_char_count": len(notes),
        "incidents_char_count": len(incidents),
        "n_exposures": len(exposures),
        "has_dwelling": "Dwelling" in exposure_types,
        "has_contents": "Content" in exposure_types,
        "has_living_expenses": "LivingExpenses" in exposure_types,
        "exposure_types": exposure_types,
        "lob_type": claim.get("CLM_LOB_TYP_CD"),
        "loss_cause": claim.get("LOSS_CSE_TYP_CD"),
        "sub_type": claim.get("CLM_SBTYP_CD"),
        "state": claim.get("PLCY_STATE"),
        "jurisdiction_state": claim.get("JURISDCTN_STATE_TYP_CD"),
        "water_source": claim.get("WATER_SRC_TYP_CD"),
        "claim_id": claim.get("NK_CLM_ID"),
        "claim_num": claim.get("CLM_NUM"),
    }
