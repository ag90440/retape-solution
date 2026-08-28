from __future__ import annotations

import json
from calendar import monthrange
from dataclasses import dataclass, field
from datetime import date
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
from typing import Literal

EntryType = Literal["credit", "debit"]


@dataclass(frozen=True)
class LedgerEntry:
    date: date
    amount_cents: int
    type: EntryType


@dataclass
class Client:
    draft_amount_cents: int
    draft_day: int
    first_draft_date: date
    last_draft_date: date
    as_of_date: date
    current_balance_cents: int
    ledger: list[LedgerEntry] = field(default_factory=list)


@dataclass(init=False)
class Offer:
    creditor: str
    current_balance_cents: int
    original_balance_cents: int
    settlement_pct: float
    first_payment_date: date | None

    def __init__(
        self,
        creditor: str,
        current_balance_cents: int | None = None,
        original_balance_cents: int = 0,
        settlement_pct: float = 0.0,
        first_payment_date: date | None = None,
        creditor_balance_cents: int | None = None,
    ) -> None:
        if creditor_balance_cents is None:
            creditor_balance_cents = current_balance_cents
        elif current_balance_cents is not None and current_balance_cents != creditor_balance_cents:
            raise ValueError("Offer balance fields do not match")
        if creditor_balance_cents is None:
            raise ValueError("Offer balance is required")
        self.creditor = creditor
        self.current_balance_cents = int(creditor_balance_cents)
        self.original_balance_cents = int(original_balance_cents)
        self.settlement_pct = float(settlement_pct)
        self.first_payment_date = first_payment_date

    @property
    def creditor_balance_cents(self) -> int:
        return self.current_balance_cents


@dataclass
class CreditorRules:
    max_terms: int
    max_payments: int
    min_payment_cents: int
    max_token_pays: int
    min_payment_tiers: list[tuple[int, int]]
    even_pays: bool
    is_ballooning_allowed: bool
    max_segments: int
    bank_fee_cents: int
    program_fee_pct: float


def round_half_up(value: Decimal | float | int) -> int:
    return int(Decimal(str(value)).quantize(Decimal("1"), rounding=ROUND_HALF_UP))


def percent_cents(amount_cents: int, percentage: Decimal | float | int) -> int:
    value = Decimal(amount_cents) * Decimal(str(percentage))
    return int(value.quantize(Decimal("1"), rounding=ROUND_HALF_UP))


def end_of_month(d: date) -> date:
    return date(d.year, d.month, monthrange(d.year, d.month)[1])


def is_end_of_month(d: date) -> bool:
    return d.day == monthrange(d.year, d.month)[1]


def add_months(d: date, n: int) -> date:
    total = d.year * 12 + d.month - 1 + n
    year, month_index = divmod(total, 12)
    month = month_index + 1
    day = min(d.day, monthrange(year, month)[1])
    return date(year, month, day)


def default_first_payment_date(client: Client) -> date:
    return end_of_month(client.first_draft_date)


def monthly_payment_dates(start: date, count: int) -> list[date]:
    if count <= 0:
        return []
    use_eom = is_end_of_month(start)
    dates: list[date] = []
    for index in range(count):
        current = add_months(start, index)
        dates.append(end_of_month(current) if use_eom else current)
    return dates


def _date(value: str) -> date:
    return date.fromisoformat(value)


def load_client(path: str | Path) -> Client:
    raw = json.loads(Path(path).read_text())
    return Client(
        draft_amount_cents=int(raw["draft_amount_cents"]),
        draft_day=int(raw["draft_day"]),
        first_draft_date=_date(raw["first_draft_date"]),
        last_draft_date=_date(raw["last_draft_date"]),
        as_of_date=_date(raw["as_of_date"]),
        current_balance_cents=int(raw["current_balance_cents"]),
        ledger=[
            LedgerEntry(_date(entry["date"]), int(entry["amount_cents"]), entry["type"])
            for entry in raw.get("ledger", [])
        ],
    )


def load_offer(path: str | Path) -> Offer:
    raw = json.loads(Path(path).read_text())
    first_payment = raw.get("first_payment_date")
    balance = raw.get("creditor_balance_cents", raw.get("current_balance_cents"))
    return Offer(
        creditor=raw["creditor"],
        creditor_balance_cents=int(balance),
        original_balance_cents=int(raw["original_balance_cents"]),
        settlement_pct=float(raw["settlement_pct"]),
        first_payment_date=_date(first_payment) if first_payment else None,
    )


def load_creditor_rules(path: str | Path) -> CreditorRules:
    raw = json.loads(Path(path).read_text())
    return CreditorRules(
        max_terms=int(raw["max_terms"]),
        max_payments=int(raw["max_payments"]),
        min_payment_cents=int(raw["min_payment_cents"]),
        max_token_pays=int(raw["max_token_pays"]),
        min_payment_tiers=[(int(start), int(amount)) for start, amount in raw.get("min_payment_tiers", [])],
        even_pays=bool(raw.get("even_pays", False)),
        is_ballooning_allowed=bool(raw.get("is_ballooning_allowed", False)),
        max_segments=int(raw.get("max_segments", 4)),
        bank_fee_cents=int(raw["bank_fee_cents"]),
        program_fee_pct=float(raw["program_fee_pct"]),
    )


def load_case(case_dir: str | Path) -> tuple[Client, Offer, CreditorRules]:
    path = Path(case_dir)
    return (
        load_client(path / "client.json"),
        load_offer(path / "offer.json"),
        load_creditor_rules(path / "creditor_rules.json"),
    )


def offer_total_cents(offer: Offer) -> int:
    return percent_cents(offer.creditor_balance_cents, offer.settlement_pct)


def program_fee_cents(offer: Offer, rules: CreditorRules) -> int:
    return percent_cents(offer.original_balance_cents, rules.program_fee_pct)
