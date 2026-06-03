from typing import Any, Dict, List, Optional

from pydantic import BaseModel


class APIRequest(BaseModel):
    id: str
    header: str
    transaction: str


class PageMarker(BaseModel):
    page_run: str


class Page(BaseModel):
    page_no: int
    page_total: int
    page_marker_raw: str
    page_marker: PageMarker


class Account(BaseModel):
    account_number: str
    owner_branch: str


class OpeningBalanceRow(BaseModel):
    label: str
    date: str
    balance_text: str


class Transaction(BaseModel):
    row_index: int
    date: str
    item: str
    debit_text: Optional[str] = None
    credit_text: Optional[str] = None
    amount_direction: str
    balance_text: str
    details: Optional[str] = None


class BankStatementResponse(BaseModel):
    page: Page
    account: Account
    opening_balance_row: OpeningBalanceRow
    transactions: list[Transaction]


class APIResponse(BaseModel):
    id: str
    answer: BankStatementResponse


class OCRRow(BaseModel):
    page_index: int
    label: str
    reading_order: int
    bbox: List[float]
    padded_box: List[int]
    ocr_text: str
    text_lines: List[Dict[str, Any]]
    ok: bool
    error: str
