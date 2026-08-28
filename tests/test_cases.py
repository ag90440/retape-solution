from __future__ import annotations

from feasibility.engine import evaluate_offer
from feasibility.models import load_case


def run_case(name: str):
    client, offer, rules = load_case(f"cases/{name}")
    return evaluate_offer(client, offer, rules)


def test_case1_feasible_even():
    result = run_case("case1_feasible_even")
    assert result.feasible is True
    assert result.pay_shape_used == "even"
    assert result.schedule is not None
    assert all(row.balance_cents >= 0 for row in result.schedule)


def test_case2_infeasible_minima():
    result = run_case("case2_infeasible_minima")
    assert result.feasible is False
    funds = result.additional_funds
    assert funds is not None
    assert funds.lump_sum.amount_cents == 10000
    assert funds.lump_sum.within_guardrail is True
    assert funds.monthly_increment.amount_cents == 2500
    assert funds.monthly_increment.num_drafts == 5
    assert funds.monthly_increment.within_guardrail is True


def test_case3_requires_balloon():
    result = run_case("case3_balloon")
    assert result.feasible is True
    assert result.pay_shape_used == "balloon"


def test_case4_tiered_minimums():
    result = run_case("case4_tiers")
    assert result.feasible is True
    assert result.pay_shape_used == "staircase"
    payments = [
        row.creditor_payment_cents
        for row in result.schedule or []
        if row.creditor_payment_cents > 0
    ]
    assert all(payment >= 5000 for payment in payments[6:])
    assert len(set(payments)) <= 2
    assert sum(payments) == 60000
