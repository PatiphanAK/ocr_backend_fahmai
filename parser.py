import re
from typing import Any, Dict, List

DATE_SLASH = r"\d{1,2}/\d{1,2}/\d{4}"
MONEY = r"\d{1,3}(?:,\d{3})*(?:\.\d{2})"


def all_text(rows: List[Dict[str, Any]]) -> str:
    parts = [str(row.get("ocr_text", "")).strip() for row in rows]
    return "\n".join(part for part in parts if part)


def first_regex(pattern: str, text: str, flags: int = re.I, group: int = 0) -> str:
    match = re.search(pattern, text, flags)
    return match.group(group).strip() if match else ""


def clean_money(value: str) -> str:
    value = value.replace(" ", "").replace("O", "0").replace("o", "0")
    match = re.search(MONEY, value)
    return match.group(0) if match else ""


def normalize_date_slash(value: str) -> str:
    match = re.search(DATE_SLASH, value)
    if not match:
        return ""
    day, month, year = match.group(0).split("/")
    return f"{int(day):02d}/{int(month):02d}/{year}"


def parse_fields(ordered_rows: List[Dict[str, Any]]) -> Dict[str, str]:
    """Extract the structured invoice fields from ordered OCR rows."""
    text = all_text(ordered_rows)

    vendor_invoice_id = first_regex(r"V-\d{3}-INV-\d{4}-\d+", text)
    vendor_id = first_regex(r"\bV-\d{3}\b", vendor_invoice_id) or first_regex(
        r"\bV-\d{3}\b", text
    )
    payment_id = first_regex(r"\bBT-[A-Z0-9OoPp-]{8,}\b", text)

    period = re.search(f"({DATE_SLASH})\\s*[–\\-]\\s*({DATE_SLASH})", text)
    period_start = normalize_date_slash(period.group(1)) if period else ""
    period_end = normalize_date_slash(period.group(2)) if period else ""

    paid_amount = clean_money(first_regex(r"(?:จำนวน|amount)[^\n]{0,40}", text))
    if not paid_amount:
        moneys = re.findall(MONEY, text.replace(" ", ""))
        paid_amount = moneys[0] if moneys else ""
    if not paid_amount:
        paid_amount = clean_money(
            first_regex(r"(?:subtotal|ยอดก่อนภาษี)[^\n]{0,40}", text)
        )

    issue_line = first_regex(r"(?:วันที่ออก|invoice date|issued)[^\n]{0,40}", text)
    business_event_date = normalize_date_slash(issue_line) or normalize_date_slash(text)

    return {
        "payment_id": payment_id,
        "vendor_id": vendor_id,
        "vendor_invoice_id": vendor_invoice_id,
        "invoice_period_start": period_start,
        "invoice_period_end": period_end,
        "paid_amount_thb": paid_amount,
        "business_event_date": business_event_date,
    }
