"""Temporal features derived from claim dates."""
from datetime import datetime


def _parse(dttm: str):
    if not dttm:
        return None
    # Handle the trailing ".0000000" style seen in Guidewire exports
    if "." in dttm:
        head, _, _tail = dttm.partition(".")
        dttm = head
    try:
        return datetime.strptime(dttm, "%Y-%m-%dT%H:%M:%S")
    except ValueError:
        return None


def temporal_features(claim: dict) -> dict:
    loss = _parse(claim.get("CLM_LOSS_DTTM"))
    reported = _parse(claim.get("CLM_RPRTD_DTTM"))

    days_loss_to_report = None
    if loss and reported:
        days_loss_to_report = (reported - loss).days

    construction_year = claim.get("PRPTY_CNSTRCTN_YR")
    property_age_at_loss = None
    if loss and isinstance(construction_year, int) and construction_year > 1800:
        property_age_at_loss = loss.year - construction_year

    return {
        "loss_date": loss.date().isoformat() if loss else None,
        "reported_date": reported.date().isoformat() if reported else None,
        "days_loss_to_report": days_loss_to_report,
        "property_age_at_loss": property_age_at_loss,
    }
