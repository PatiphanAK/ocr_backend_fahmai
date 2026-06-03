import re
from typing import Any, Dict, List, Optional, Tuple

from rapidfuzz import fuzz

from items import (
    Account,
    BankStatementResponse,
    OCRRow,
    OpeningBalanceRow,
    Page,
    PageMarker,
    Transaction,
)

DATE_DASH_SHORT = r"\d{1,2}-\d{1,2}-\d{2}"
MONEY_LOOSE = r"\d[\d,\sOo]*(?:\.\d{2})"
PAGE_MARKER = r"(\d+)\s*/\s*(\d+)\s*\((\d+)\)"

FUZZY_THRESHOLD = 80

LABEL_OPENING_BALANCE = ["ยอดยกมา"]
LABEL_CLOSING_BALANCE = ["ยอดยกไป"]
LABEL_OWNER_BRANCH = ["สาขาเจ้าของบัญชี"]

CREDIT_KEYWORDS = ["รับโอนเงิน", "เงินเข้า", "ฝากเงิน", "ดอกเบี้ย", "รับเงิน"]
DEBIT_KEYWORDS = ["ค่าธรรมเนียม", "ถอนเงิน", "โอนเงิน", "หักบัญชี", "ชำระ"]


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------
def _fuzzy_contains(
    text: str, targets: List[str], threshold: int = FUZZY_THRESHOLD
) -> bool:
    """True if any fuzzy target appears in text (handles OCR noise)."""
    for target in targets:
        # partial_ratio: target may be a substring inside a longer OCR line
        if fuzz.partial_ratio(target, text) >= threshold:
            return True
    return False


def _clean_money(value: str) -> str:
    """Borrowed from dev. Fix common OCR confusions and keep money pattern."""
    value = value.replace(" ", "").replace("O", "0").replace("o", "0")
    match = re.search(r"\d[\d,]*(?:\.\d{2})", value)
    return match.group(0) if match else ""


def _money_values(text: str) -> List[str]:
    out = []
    for match in re.finditer(MONEY_LOOSE, text):
        cleaned = _clean_money(match.group(0))
        if cleaned:
            out.append(cleaned)
    return out


def _money_float(value: str) -> float:
    try:
        return float(value.replace(",", ""))
    except ValueError:
        return 0.0


def _normalize_date(value: str) -> str:
    """Keep DD-MM-YY as-is (zero padded). Year is ค.ศ. per spec."""
    match = re.search(DATE_DASH_SHORT, value)
    if not match:
        return ""
    day, month, year = match.group(0).split("-")
    return f"{int(day):02d}-{int(month):02d}-{year}"


# ---------------------------------------------------------------------------
# line grouping (borrowed from dev: group_text_rows)
# ---------------------------------------------------------------------------
def _line_box(line: Dict[str, Any]) -> List[float]:
    box = line.get("box") or [0, 0, 0, 0]
    return [float(x) for x in box]


def _line_center_y(line: Dict[str, Any]) -> float:
    _, y1, _, y2 = _line_box(line)
    return (y1 + y2) / 2.0


def _line_left_x(line: Dict[str, Any]) -> float:
    return _line_box(line)[0]


def _group_text_rows(
    lines: List[Dict[str, Any]], y_threshold: float
) -> List[List[Dict[str, Any]]]:
    groups: List[List[Dict[str, Any]]] = []
    for line in sorted(lines, key=lambda i: (_line_center_y(i), _line_left_x(i))):
        if (
            not groups
            or abs(_line_center_y(groups[-1][0]) - _line_center_y(line)) > y_threshold
        ):
            groups.append([line])
        else:
            groups[-1].append(line)
    return [sorted(group, key=_line_left_x) for group in groups]


# ---------------------------------------------------------------------------
# noise filter (borrowed + extended from dev: is_noise_row)
# ---------------------------------------------------------------------------
_NOISE_TOKENS = [
    "total no.",
    "total debit",
    "total credit",
    "item",
    "inquiry contact",
    "computer-generated",
    "page",
    "จำนวนเดบิต",
    "จำนวนเครดิต",
    "ยอดรวมเดบิต",
    "ยอดรวมเครดิต",
    "เอกสารนี้",
    "สอบถามข้อมูล",
    "วันที่",
    "รายการ",
]


def _is_noise_row(row_text: str) -> bool:
    lowered = row_text.lower()
    return any(tok in lowered or tok in row_text for tok in _NOISE_TOKENS)


# ---------------------------------------------------------------------------
# header parsers
# ---------------------------------------------------------------------------
def _all_text(rows: List[OCRRow]) -> str:
    return "\n".join(r.ocr_text.strip() for r in rows if r.ocr_text.strip())


def _extract_account_number(text: str) -> str:
    # BBL style: 060-3-77856-1
    match = re.search(r"\b\d{3}-\d-\d{5,}-\d\b", text)
    if match:
        return match.group(0)
    match = re.search(r"\b\d{10,14}\b", text)
    return match.group(0) if match else ""


def _extract_page_marker(text: str) -> Tuple[Page, str]:
    """Parse '1/2(8561)' style marker. Falls back to 1/1 if absent."""
    match = re.search(PAGE_MARKER, text)
    if match:
        page_no = int(match.group(1))
        page_total = int(match.group(2))
        run = match.group(3)
        raw = f"{page_no}/{page_total}({run})"
        return Page(
            page_no=page_no,
            page_total=page_total,
            page_marker_raw=raw,
            page_marker=PageMarker(page_run=run),
        ), raw
    # ASSUMPTION: no marker found -> single page. Verify against real OCR.
    return Page(
        page_no=1,
        page_total=1,
        page_marker_raw="",
        page_marker=PageMarker(page_run=""),
    ), ""


def _extract_owner_branch(text: str) -> str:
    for line in text.splitlines():
        if _fuzzy_contains(line, LABEL_OWNER_BRANCH):
            return line.strip()
    return ""


def _extract_opening_balance(rows: List[OCRRow]) -> OpeningBalanceRow:
    """Find the ยอดยกมา row (fuzzy) and pull its date + balance."""
    for row in rows:
        for raw_line in row.ocr_text.splitlines():
            if _fuzzy_contains(raw_line, LABEL_OPENING_BALANCE):
                date = _normalize_date(raw_line)
                moneys = _money_values(raw_line)
                balance = moneys[-1] if moneys else ""
                return OpeningBalanceRow(
                    label="ยอดยกมา", date=date, balance_text=balance
                )
    return OpeningBalanceRow(label="ยอดยกมา", date="", balance_text="")


# ---------------------------------------------------------------------------
# transaction row parsing
# ---------------------------------------------------------------------------
def _infer_direction(row_text: str, balance: str, prev_balance: str) -> str:
    """
    Decide credit vs debit.
    Priority 1: compare balance vs previous balance (most reliable).
    Priority 2: fuzzy keyword match on the transaction description.
    """
    if prev_balance and balance:
        cur = _money_float(balance)
        prev = _money_float(prev_balance)
        if cur > prev:
            return "credit"
        if cur < prev:
            return "debit"
    if _fuzzy_contains(row_text, CREDIT_KEYWORDS):
        return "credit"
    if _fuzzy_contains(row_text, DEBIT_KEYWORDS):
        return "debit"
    return ""


def _parse_transaction_row(
    cells: List[Dict[str, Any]], prev_balance: str
) -> Optional[Dict[str, Any]]:
    """
    Turn one visual row into transaction fields.

    ASSUMPTION (must verify against real BBL/KBank/SCB layout):
    a transaction row has at least one date + >=2 money values, where the
    LAST money is the running balance and the SECOND-LAST is the txn amount.
    This mirrors dev's parse_row_cells heuristic, which does not rely on a
    fixed debit/credit column. We assign the amount to debit_text or
    credit_text based on _infer_direction.
    """
    row_text = " ".join(str(c.get("text", "")) for c in cells).strip()

    date = _normalize_date(row_text)
    if not date or _is_noise_row(row_text):
        return None

    moneys = _money_values(row_text)
    if not moneys:
        return None

    if len(moneys) >= 2:
        amount = moneys[-2]
        balance = moneys[-1]
    else:
        amount = moneys[-1]
        balance = ""

    direction = _infer_direction(row_text, balance, prev_balance)

    # item = first non-money, non-date text token; details = the rest
    desc_parts = []
    for cell in cells:
        text = str(cell.get("text", "")).strip()
        if (
            not text
            or re.fullmatch(MONEY_LOOSE, text)
            or re.fullmatch(DATE_DASH_SHORT, text)
        ):
            continue
        desc_parts.append(text)
    item = desc_parts[0] if desc_parts else ""
    details = " ".join(desc_parts[1:]).strip() if len(desc_parts) > 1 else None

    return {
        "date": date,
        "item": item,
        "amount": amount,
        "balance": balance,
        "direction": direction,
        "details": details,
    }


def _extract_transactions(rows: List[OCRRow]) -> List[Transaction]:
    transactions: List[Transaction] = []
    row_index = 1
    prev_balance = ""

    for ocr_row in rows:
        lines = [ln for ln in ocr_row.text_lines if isinstance(ln, dict)]
        if not lines:
            continue
        # ASSUMPTION: y_threshold scales with crop height. 0.6% mirrors dev.
        crop_h = max((_line_box(ln)[3] for ln in lines), default=1200)
        y_threshold = max(12.0, crop_h * 0.006)

        for cells in _group_text_rows(lines, y_threshold):
            row_text = " ".join(str(c.get("text", "")) for c in cells)
            # skip opening/closing balance rows (handled separately)
            if _fuzzy_contains(row_text, LABEL_OPENING_BALANCE) or _fuzzy_contains(
                row_text, LABEL_CLOSING_BALANCE
            ):
                continue
            parsed = _parse_transaction_row(cells, prev_balance)
            if parsed is None:
                continue

            direction = parsed["direction"]
            transactions.append(
                Transaction(
                    row_index=row_index,
                    date=parsed["date"],
                    item=parsed["item"],
                    debit_text=parsed["amount"] if direction == "debit" else None,
                    credit_text=parsed["amount"] if direction == "credit" else None,
                    amount_direction=direction,
                    balance_text=parsed["balance"],
                    details=parsed["details"],
                )
            )
            if parsed["balance"]:
                prev_balance = parsed["balance"]
            row_index += 1

    return transactions


# ---------------------------------------------------------------------------
# public entrypoint
# ---------------------------------------------------------------------------
def parse_fields(ordered_rows: List[OCRRow]) -> BankStatementResponse:
    text = _all_text(ordered_rows)

    page, _ = _extract_page_marker(text)
    account = Account(
        account_number=_extract_account_number(text),
        owner_branch=_extract_owner_branch(text),
    )
    opening = _extract_opening_balance(ordered_rows)
    transactions = _extract_transactions(ordered_rows)

    return BankStatementResponse(
        page=page,
        account=account,
        opening_balance_row=opening,
        transactions=transactions,
    )
