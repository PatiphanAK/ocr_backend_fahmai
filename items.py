from pydantic import BaseModel


class APIRequest(BaseModel):
    header: str
    transaction: str


class APIResponse(BaseModel):
    payment_id: str
    vendor_id: str
    vendor_invoice_id: str
    invoice_period_start: str
    invoice_period_end: str
    paid_amount_thb: str
    business_event_date: str
