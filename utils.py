DATE_DASH_SHORT = r"\d{1,2}-\d{1,2}-\d{2}"
MONEY_LOOSE = r"\d[\d,\sOo]*(?:\.\d{2})"


def artifact_category(artifact_id: str) -> str:
    if artifact_id.startswith("BS-"):
        return "bank_statement"
    if artifact_id.startswith("BN-"):
        return "e7_banner"
    if artifact_id.startswith("RC-"):
        return "receipt"
    if artifact_id.startswith("VI-"):
        return "vendor_invoice"
    if artifact_id.startswith("WC-"):
        return "warranty_form"
    if artifact_id.startswith("T3-"):
        return "t3_doc"
    if artifact_id.startswith(("AUD-", "EMAIL-", "MEMO-", "POL-", "TRAIN-", "VC-")):
        return "t2_doc"
    return "unknown"
