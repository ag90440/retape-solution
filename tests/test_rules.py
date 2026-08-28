from datetime import date

from feasibility.engine import evaluate_offer
from feasibility.models import Client, CreditorRules, LedgerEntry, Offer


def rules(
    max_payments: int = 1,
    minimum: int = 1,
    tokens: int = 10,
    tiers: list[tuple[int, int]] | None = None,
    even: bool = False,
    balloon: bool = False,
    segments: int = 4,
    bank: int = 0,
    fee: float = 0.0,
) -> CreditorRules:
    return CreditorRules(
        max_terms=max_payments,
        max_payments=max_payments,
        min_payment_cents=minimum,
        max_token_pays=tokens,
        min_payment_tiers=tiers or [],
        even_pays=even,
        is_ballooning_allowed=balloon,
        max_segments=segments,
        bank_fee_cents=bank,
        program_fee_pct=fee,
    )


def client(
    ledger: list[LedgerEntry],
    horizon: date,
    current: int = 0,
    as_of: date = date(2025, 12, 31),
    draft: int = 10000,
) -> Client:
    return Client(
        draft_amount_cents=draft,
        draft_day=1,
        first_draft_date=date(2026, 1, 1),
        last_draft_date=horizon,
        as_of_date=as_of,
        current_balance_cents=current,
        ledger=ledger,
    )


def offer(
    balance: int,
    first: date,
    original: int | None = None,
    settlement: float = 1.0,
) -> Offer:
    return Offer(
        creditor="Test",
        creditor_balance_cents=balance,
        original_balance_cents=balance if original is None else original,
        settlement_pct=settlement,
        first_payment_date=first,
    )


def test_same_day_credit_first_and_zero_balance():
    day = date(2026, 1, 31)
    value = client(
        [
            LedgerEntry(day, 10000, "credit"),
            LedgerEntry(day, 5000, "debit"),
        ],
        day,
    )
    result = evaluate_offer(value, offer(5000, day), rules(minimum=5000))
    assert result.feasible is True
    assert result.schedule is not None
    assert result.schedule[-1].balance_cents == 0


def test_old_ledger_rows_are_not_used_again():
    value = client(
        [LedgerEntry(date(2025, 12, 30), 10000, "debit")],
        date(2026, 1, 31),
        current=10000,
    )
    result = evaluate_offer(
        value,
        offer(10000, date(2026, 1, 31)),
        rules(minimum=10000),
    )
    assert result.feasible is True
    assert result.schedule is not None
    assert result.schedule[-1].balance_cents == 0


def test_fee_only_date_has_no_bank_fee():
    value = client(
        [
            LedgerEntry(date(2026, 1, 1), 5000, "credit"),
            LedgerEntry(date(2026, 2, 1), 5000, "credit"),
        ],
        date(2026, 2, 28),
        draft=5000,
    )
    result = evaluate_offer(
        value,
        offer(2500, date(2026, 1, 31), original=7500),
        rules(minimum=2500, bank=400, fee=1.0),
    )
    assert result.feasible is False
    value.ledger[1] = LedgerEntry(date(2026, 2, 1), 5400, "credit")
    result = evaluate_offer(
        value,
        offer(2500, date(2026, 1, 31), original=7500),
        rules(minimum=2500, bank=400, fee=1.0),
    )
    assert result.feasible is True
    assert result.schedule is not None
    assert result.schedule[-1].creditor_payment_cents == 0
    assert result.schedule[-1].program_fee_cents > 0
    assert result.schedule[-1].bank_fee_cents == 0


def test_token_limit_changes_second_payment():
    value = client(
        [
            LedgerEntry(date(2026, 1, 1), 2500, "credit"),
            LedgerEntry(date(2026, 2, 1), 2501, "credit"),
        ],
        date(2026, 2, 28),
        draft=2500,
    )
    result = evaluate_offer(
        value,
        offer(5001, date(2026, 1, 31)),
        rules(max_payments=2, minimum=2500, tokens=1, segments=2),
    )
    assert result.feasible is True
    assert result.schedule is not None
    payments = [row.creditor_payment_cents for row in result.schedule]
    assert payments == [2500, 2501]


def test_even_remainder_is_on_late_payments():
    value = client(
        [
            LedgerEntry(date(2026, 1, 1), 3333, "credit"),
            LedgerEntry(date(2026, 2, 1), 3333, "credit"),
            LedgerEntry(date(2026, 3, 1), 3334, "credit"),
        ],
        date(2026, 3, 31),
        draft=3333,
    )
    result = evaluate_offer(
        value,
        offer(10000, date(2026, 1, 31)),
        rules(max_payments=3, minimum=1, even=True),
    )
    assert result.feasible is True
    assert result.schedule is not None
    assert [row.creditor_payment_cents for row in result.schedule] == [3333, 3333, 3334]


def test_horizon_date_is_allowed():
    horizon = date(2026, 1, 31)
    value = client([LedgerEntry(horizon, 5000, "credit")], horizon)
    result = evaluate_offer(value, offer(5000, horizon), rules(minimum=5000))
    assert result.feasible is True


def test_fee_is_not_before_first_payment():
    value = client(
        [
            LedgerEntry(date(2026, 1, 1), 10000, "credit"),
            LedgerEntry(date(2026, 2, 1), 10000, "credit"),
        ],
        date(2026, 2, 28),
    )
    first = date(2026, 1, 31)
    result = evaluate_offer(
        value,
        offer(5000, first, original=10000),
        rules(max_payments=1, minimum=5000, fee=0.5),
    )
    assert result.feasible is True
    assert result.schedule is not None
    assert all(row.date >= first for row in result.schedule if row.program_fee_cents)
