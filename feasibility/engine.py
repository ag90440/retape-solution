from __future__ import annotations

from calendar import monthrange
from dataclasses import dataclass
from datetime import date, timedelta
from math import inf, isinf

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix

from feasibility.models import (
    Client,
    CreditorRules,
    Offer,
    add_months,
    default_first_payment_date,
    end_of_month,
    is_end_of_month,
    offer_total_cents,
    percent_cents,
    program_fee_cents,
)


@dataclass
class ScheduleRow:
    date: date
    creditor_payment_cents: int
    program_fee_cents: int
    bank_fee_cents: int
    balance_cents: int


@dataclass
class FundsOption:
    amount_cents: int
    within_guardrail: bool
    reason: str
    date: date | None = None
    num_drafts: int | None = None


@dataclass
class AdditionalFunds:
    lump_sum: FundsOption
    monthly_increment: FundsOption


@dataclass
class Result:
    feasible: bool
    pay_shape_used: str | None = None
    schedule: list[ScheduleRow] | None = None
    additional_funds: AdditionalFunds | None = None

    def to_dict(self) -> dict:
        schedule = None
        if self.schedule is not None:
            schedule = [
                {
                    "date": row.date.isoformat(),
                    "creditor_payment_cents": row.creditor_payment_cents,
                    "program_fee_cents": row.program_fee_cents,
                    "bank_fee_cents": row.bank_fee_cents,
                    "balance_cents": row.balance_cents,
                }
                for row in self.schedule
            ]
        funds = None
        if self.additional_funds is not None:
            funds = {
                "lump_sum": self._option(self.additional_funds.lump_sum),
                "monthly_increment": self._option(self.additional_funds.monthly_increment),
            }
        return {
            "feasible": self.feasible,
            "pay_shape_used": self.pay_shape_used,
            "schedule": schedule,
            "additional_funds": funds,
        }

    @staticmethod
    def _option(value: FundsOption) -> dict:
        output = {
            "amount_cents": value.amount_cents,
            "within_guardrail": value.within_guardrail,
            "reason": value.reason,
        }
        if value.date is not None:
            output["date"] = value.date.isoformat()
        if value.num_drafts is not None:
            output["num_drafts"] = value.num_drafts
        return output


class _Model:
    def __init__(self) -> None:
        self.lower: list[float] = []
        self.upper: list[float] = []
        self.integrality: list[int] = []
        self.rows: list[dict[int, int]] = []
        self.row_lower: list[float] = []
        self.row_upper: list[float] = []

    def add_var(self, lower: int = 0, upper: float = inf) -> int:
        index = len(self.lower)
        self.lower.append(lower)
        self.upper.append(upper)
        self.integrality.append(1)
        return index

    def add_row(self, values: dict[int, int], lower: float = -inf, upper: float = inf) -> None:
        self.rows.append({index: value for index, value in values.items() if value})
        self.row_lower.append(lower)
        self.row_upper.append(upper)

    def solve(self, objective: dict[int, int] | None = None) -> list[int] | None:
        size = len(self.lower)
        cost = np.zeros(size, dtype=float)
        for index, value in (objective or {}).items():
            cost[index] = value
        row_indexes: list[int] = []
        column_indexes: list[int] = []
        data: list[int] = []
        for row_index, row in enumerate(self.rows):
            for column_index, value in row.items():
                row_indexes.append(row_index)
                column_indexes.append(column_index)
                data.append(value)
        matrix = coo_matrix(
            (data, (row_indexes, column_indexes)),
            shape=(len(self.rows), size),
            dtype=float,
        ).tocsr()
        result = milp(
            c=cost,
            integrality=np.array(self.integrality, dtype=int),
            bounds=Bounds(np.array(self.lower, dtype=float), np.array(self.upper, dtype=float)),
            constraints=LinearConstraint(
                matrix,
                np.array(self.row_lower, dtype=float),
                np.array(self.row_upper, dtype=float),
            ),
            options={"presolve": True, "mip_rel_gap": 0.0},
        )
        if not result.success or result.x is None:
            return None
        values = [int(round(value)) for value in result.x]
        return values if self.valid(values) else None

    def valid(self, values: list[int]) -> bool:
        for index, value in enumerate(values):
            if value < self.lower[index] - 0.01:
                return False
            if not isinf(self.upper[index]) and value > self.upper[index] + 0.01:
                return False
        for row, lower, upper in zip(self.rows, self.row_lower, self.row_upper):
            value = sum(values[index] * amount for index, amount in row.items())
            if not isinf(lower) and value < lower - 0.01:
                return False
            if not isinf(upper) and value > upper + 0.01:
                return False
        return True


@dataclass
class _Problem:
    model: _Model
    payment_indexes: list[int]
    fee_indexes: list[int]
    funding_index: int | None
    cadence: list[date]
    payment_count: int
    shape: str
    fixed_payments: bool
    funding_mode: str | None
    lump_date: date


@dataclass
class _Plan:
    payments: list[int]
    fees: list[int]
    cadence: list[date]
    payment_count: int
    shape: str
    schedule: list[ScheduleRow]

    def fee_key(self) -> tuple[int, ...]:
        total = 0
        values: list[int] = []
        for fee in self.fees:
            total += fee
            values.append(total)
        return tuple(values)

    def creditor_key(self) -> tuple[int, ...]:
        total = 0
        values: list[int] = []
        for index in range(len(self.cadence)):
            if index < self.payment_count:
                total += self.payments[index]
            values.append(total)
        return tuple(values)


def _shape(rules: CreditorRules) -> str:
    if rules.even_pays:
        return "even"
    if rules.is_ballooning_allowed:
        return "balloon"
    return "staircase"


def _cadence_dates(start: date, horizon: date) -> list[date]:
    if start > horizon:
        return []
    count = (horizon.year - start.year) * 12 + horizon.month - start.month + 2
    use_eom = is_end_of_month(start)
    values: list[date] = []
    for index in range(count):
        current = add_months(start, index)
        if use_eom:
            current = end_of_month(current)
        if current <= horizon:
            values.append(current)
    return values


def _draft_dates(client: Client) -> list[date]:
    if client.first_draft_date > client.last_draft_date:
        return []
    count = (
        (client.last_draft_date.year - client.first_draft_date.year) * 12
        + client.last_draft_date.month
        - client.first_draft_date.month
        + 1
    )
    values: list[date] = []
    for index in range(count):
        shifted = add_months(client.first_draft_date, index)
        if index == 0:
            current = client.first_draft_date
        else:
            day = min(client.draft_day, monthrange(shifted.year, shifted.month)[1])
            current = date(shifted.year, shifted.month, day)
        if client.first_draft_date <= current <= client.last_draft_date:
            values.append(current)
    return values


def _future_entries(client: Client):
    return [
        entry
        for entry in client.ledger
        if client.as_of_date < entry.date <= client.last_draft_date
    ]


def _future_drafts(client: Client) -> list[date]:
    return [
        value
        for value in _draft_dates(client)
        if client.as_of_date < value <= client.last_draft_date
    ]


def _static_floor(position: int, rules: CreditorRules) -> int:
    value = max(1, rules.min_payment_cents)
    for start, floor in rules.min_payment_tiers:
        if position >= start:
            value = max(value, floor)
    return value


def _valid_payments(payments: list[int], total: int, rules: CreditorRules, shape: str) -> bool:
    if not payments or sum(payments) != total:
        return False
    if any(value <= 0 for value in payments):
        return False
    if any(payments[index] < payments[index - 1] for index in range(1, len(payments))):
        return False
    for index, value in enumerate(payments):
        if value < _static_floor(index + 1, rules):
            return False
    if sum(value == rules.min_payment_cents for value in payments) > rules.max_token_pays:
        return False
    if shape == "even":
        base, remainder = divmod(total, len(payments))
        expected = [base] * (len(payments) - remainder) + [base + 1] * remainder
        if payments != expected:
            return False
    if shape == "staircase" and len(set(payments)) > rules.max_segments:
        return False
    return True


def _even_payments(total: int, count: int, rules: CreditorRules) -> list[int] | None:
    base, remainder = divmod(total, count)
    payments = [base] * (count - remainder) + [base + 1] * remainder
    return payments if _valid_payments(payments, total, rules, "even") else None


def _balloon_payments(total: int, count: int, rules: CreditorRules) -> list[int] | None:
    payments: list[int] = []
    token_count = 0
    previous = 0
    for index in range(count - 1):
        value = max(previous, _static_floor(index + 1, rules))
        if value == rules.min_payment_cents and token_count >= rules.max_token_pays:
            value += 1
        payments.append(value)
        previous = value
        if value == rules.min_payment_cents:
            token_count += 1
    payments.append(total - sum(payments))
    return payments if _valid_payments(payments, total, rules, "balloon") else None


def _fixed_payments(total: int, count: int, rules: CreditorRules) -> list[int] | None:
    if rules.even_pays:
        return _even_payments(total, count, rules)
    if rules.is_ballooning_allowed:
        return _balloon_payments(total, count, rules)
    return None


def _base_balances(client: Client, event_dates: list[date]) -> dict[date, int]:
    by_date: dict[date, list] = {}
    for entry in _future_entries(client):
        by_date.setdefault(entry.date, []).append(entry)
    balance = client.current_balance_cents
    values: dict[date, int] = {}
    for current in sorted(set(event_dates) | {client.as_of_date, client.last_draft_date}):
        if current > client.as_of_date:
            for entry in by_date.get(current, []):
                if entry.type == "credit":
                    balance += entry.amount_cents
            for entry in by_date.get(current, []):
                if entry.type == "debit":
                    balance -= entry.amount_cents
        values[current] = balance
    return values


def _build_problem(
    client: Client,
    offer: Offer,
    rules: CreditorRules,
    payment_count: int,
    funding_mode: str | None = None,
) -> _Problem | None:
    total = offer_total_cents(offer)
    fee_total = program_fee_cents(offer, rules)
    if total <= 0 or fee_total < 0:
        return None
    first_payment = offer.first_payment_date or default_first_payment_date(client)
    if first_payment <= client.as_of_date:
        return None
    cadence = _cadence_dates(first_payment, client.last_draft_date)
    maximum = min(rules.max_terms, rules.max_payments, len(cadence))
    if payment_count < 1 or payment_count > maximum:
        return None
    shape = _shape(rules)
    fixed = _fixed_payments(total, payment_count, rules)
    if shape != "staircase" and fixed is None:
        return None
    if shape == "staircase" and rules.max_segments < 1:
        return None
    model = _Model()
    payment_indexes: list[int] = []
    if fixed is None:
        for index in range(payment_count):
            payment_indexes.append(model.add_var(_static_floor(index + 1, rules), total))
    else:
        for value in fixed:
            payment_indexes.append(model.add_var(value, value))
    model.add_row({index: 1 for index in payment_indexes}, total, total)
    for index in range(1, payment_count):
        model.add_row(
            {payment_indexes[index]: 1, payment_indexes[index - 1]: -1},
            0,
            inf,
        )
    if shape == "staircase":
        token_indexes: list[int] = []
        token_big = max(1, total - rules.min_payment_cents + 1)
        for index, payment_index in enumerate(payment_indexes):
            if _static_floor(index + 1, rules) == rules.min_payment_cents:
                token_index = model.add_var(0, 1)
                token_indexes.append(token_index)
                model.add_row(
                    {payment_index: 1, token_index: token_big},
                    -inf,
                    rules.min_payment_cents + token_big,
                )
                model.add_row(
                    {payment_index: 1, token_index: token_big},
                    rules.min_payment_cents + 1,
                    inf,
                )
        if token_indexes:
            model.add_row(
                {index: 1 for index in token_indexes},
                -inf,
                max(0, rules.max_token_pays),
            )
        change_indexes: list[int] = []
        for index in range(1, payment_count):
            change_index = model.add_var(0, 1)
            change_indexes.append(change_index)
            model.add_row(
                {
                    payment_indexes[index]: 1,
                    payment_indexes[index - 1]: -1,
                    change_index: -total,
                },
                -inf,
                0,
            )
            model.add_row(
                {
                    payment_indexes[index]: 1,
                    payment_indexes[index - 1]: -1,
                    change_index: -1,
                },
                0,
                inf,
            )
        if change_indexes:
            model.add_row(
                {index: 1 for index in change_indexes},
                -inf,
                rules.max_segments - 1,
            )
    fee_indexes = [model.add_var(0, fee_total) for _ in cadence]
    model.add_row({index: 1 for index in fee_indexes}, fee_total, fee_total)
    funding_index = model.add_var(0, inf) if funding_mode is not None else None
    lump_date = client.as_of_date + timedelta(days=1)
    drafts = _future_drafts(client)
    event_dates = sorted(
        {
            client.as_of_date,
            client.last_draft_date,
            *cadence,
            *drafts,
            *(entry.date for entry in _future_entries(client)),
        }
    )
    if funding_mode == "lump" and lump_date <= client.last_draft_date:
        event_dates = sorted(set(event_dates) | {lump_date})
    balances = _base_balances(client, event_dates)
    for current in event_dates:
        row: dict[int, int] = {}
        for index, payment_date in enumerate(cadence[:payment_count]):
            if payment_date <= current:
                row[payment_indexes[index]] = 1
        for index, fee_date in enumerate(cadence):
            if fee_date <= current:
                row[fee_indexes[index]] = 1
        if funding_index is not None:
            if funding_mode == "lump" and lump_date <= current <= client.last_draft_date:
                row[funding_index] = -1
            if funding_mode == "monthly":
                affected = sum(value <= current for value in drafts)
                if affected:
                    row[funding_index] = -affected
        bank_count = sum(value <= current for value in cadence[:payment_count])
        model.add_row(
            row,
            -inf,
            balances[current] - bank_count * rules.bank_fee_cents,
        )
    return _Problem(
        model=model,
        payment_indexes=payment_indexes,
        fee_indexes=fee_indexes,
        funding_index=funding_index,
        cadence=cadence,
        payment_count=payment_count,
        shape=shape,
        fixed_payments=fixed is not None,
        funding_mode=funding_mode,
        lump_date=lump_date,
    )


def _simulate(
    client: Client,
    rules: CreditorRules,
    cadence: list[date],
    payments: list[int],
    fees: list[int],
    funding_mode: str | None = None,
    funding_amount: int = 0,
    lump_date: date | None = None,
) -> tuple[bool, dict[date, int]]:
    entries: dict[date, list] = {}
    for entry in _future_entries(client):
        entries.setdefault(entry.date, []).append(entry)
    drafts = set(_future_drafts(client))
    dates = {
        client.as_of_date,
        client.last_draft_date,
        *cadence,
        *drafts,
        *(entry.date for entry in _future_entries(client)),
    }
    if lump_date is not None and lump_date <= client.last_draft_date:
        dates.add(lump_date)
    cadence_index = {value: index for index, value in enumerate(cadence)}
    balance = client.current_balance_cents
    balances: dict[date, int] = {}
    for current in sorted(dates):
        if current > client.as_of_date:
            for entry in entries.get(current, []):
                if entry.type == "credit":
                    balance += entry.amount_cents
            if funding_mode == "monthly" and current in drafts:
                balance += funding_amount
            if funding_mode == "lump" and current == lump_date:
                balance += funding_amount
            for entry in entries.get(current, []):
                if entry.type == "debit":
                    balance -= entry.amount_cents
            if current in cadence_index:
                index = cadence_index[current]
                if index < len(payments):
                    balance -= payments[index]
                    balance -= rules.bank_fee_cents
                balance -= fees[index]
        balances[current] = balance
        if balance < 0:
            return False, balances
    return True, balances


def _solution_valid(
    client: Client,
    offer: Offer,
    rules: CreditorRules,
    problem: _Problem,
    solution: list[int],
    funding_amount: int = 0,
) -> tuple[bool, list[int], list[int], dict[date, int]]:
    payments = [solution[index] for index in problem.payment_indexes]
    fees = [solution[index] for index in problem.fee_indexes]
    if not _valid_payments(payments, offer_total_cents(offer), rules, problem.shape):
        return False, payments, fees, {}
    if sum(fees) != program_fee_cents(offer, rules) or any(value < 0 for value in fees):
        return False, payments, fees, {}
    valid, balances = _simulate(
        client,
        rules,
        problem.cadence,
        payments,
        fees,
        problem.funding_mode,
        funding_amount,
        problem.lump_date,
    )
    return valid, payments, fees, balances


def _make_plan(
    client: Client,
    offer: Offer,
    rules: CreditorRules,
    problem: _Problem,
    solution: list[int],
) -> _Plan | None:
    valid, payments, fees, balances = _solution_valid(client, offer, rules, problem, solution)
    if not valid:
        return None
    schedule: list[ScheduleRow] = []
    for index, current in enumerate(problem.cadence):
        payment = payments[index] if index < problem.payment_count else 0
        fee = fees[index]
        if payment == 0 and fee == 0:
            continue
        schedule.append(
            ScheduleRow(
                date=current,
                creditor_payment_cents=payment,
                program_fee_cents=fee,
                bank_fee_cents=rules.bank_fee_cents if payment else 0,
                balance_cents=balances[current],
            )
        )
    return _Plan(
        payments=payments,
        fees=fees,
        cadence=problem.cadence,
        payment_count=problem.payment_count,
        shape=problem.shape,
        schedule=schedule,
    )


def _plan_for_count(
    client: Client,
    offer: Offer,
    rules: CreditorRules,
    payment_count: int,
) -> _Plan | None:
    problem = _build_problem(client, offer, rules, payment_count)
    if problem is None:
        return None
    solution = problem.model.solve()
    if solution is None:
        return None
    fee_total = program_fee_cents(offer, rules)
    for end in range(len(problem.fee_indexes) - 1):
        indexes = problem.fee_indexes[: end + 1]
        solution = problem.model.solve({index: -1 for index in indexes})
        if solution is None:
            return None
        value = sum(solution[index] for index in indexes)
        problem.model.add_row({index: 1 for index in indexes}, value, value)
        if value == fee_total:
            break
    if not problem.fixed_payments:
        for end in range(problem.payment_count - 1):
            indexes = problem.payment_indexes[: end + 1]
            solution = problem.model.solve({index: 1 for index in indexes})
            if solution is None:
                return None
            value = sum(solution[index] for index in indexes)
            problem.model.add_row({index: 1 for index in indexes}, value, value)
    solution = problem.model.solve()
    if solution is None:
        return None
    return _make_plan(client, offer, rules, problem, solution)


def _best_plan(client: Client, offer: Offer, rules: CreditorRules) -> _Plan | None:
    first_payment = offer.first_payment_date or default_first_payment_date(client)
    cadence = _cadence_dates(first_payment, client.last_draft_date)
    maximum = min(rules.max_terms, rules.max_payments, len(cadence))
    plans = [
        plan
        for payment_count in range(1, maximum + 1)
        if (plan := _plan_for_count(client, offer, rules, payment_count)) is not None
    ]
    if not plans:
        return None
    return max(
        plans,
        key=lambda plan: (
            plan.fee_key(),
            tuple(-value for value in plan.creditor_key()),
            -plan.payment_count * rules.bank_fee_cents,
            -plan.payment_count,
        ),
    )


def _minimum_funding(
    client: Client,
    offer: Offer,
    rules: CreditorRules,
    mode: str,
) -> int | None:
    first_payment = offer.first_payment_date or default_first_payment_date(client)
    cadence = _cadence_dates(first_payment, client.last_draft_date)
    maximum = min(rules.max_terms, rules.max_payments, len(cadence))
    amounts: list[int] = []
    for payment_count in range(1, maximum + 1):
        problem = _build_problem(client, offer, rules, payment_count, mode)
        if problem is None or problem.funding_index is None:
            continue
        solution = problem.model.solve({problem.funding_index: 1})
        if solution is None:
            continue
        amount = solution[problem.funding_index]
        valid, _, _, _ = _solution_valid(client, offer, rules, problem, solution, amount)
        if valid:
            amounts.append(amount)
    return min(amounts) if amounts else None


def _funds_option(
    amount: int | None,
    limit: int,
    mode: str,
    lump_date: date,
    draft_count: int,
) -> FundsOption:
    if amount is None:
        return FundsOption(
            amount_cents=0,
            within_guardrail=False,
            reason="No amount can fix this offer.",
            date=lump_date if mode == "lump" else None,
            num_drafts=draft_count if mode == "monthly" else None,
        )
    within = amount <= limit
    return FundsOption(
        amount_cents=amount,
        within_guardrail=within,
        reason="" if within else "Amount is above the allowed limit.",
        date=lump_date if mode == "lump" else None,
        num_drafts=draft_count if mode == "monthly" else None,
    )


def evaluate_offer(client: Client, offer: Offer, rules: CreditorRules) -> Result:
    plan = _best_plan(client, offer, rules)
    if plan is not None:
        return Result(
            feasible=True,
            pay_shape_used=plan.shape,
            schedule=plan.schedule,
            additional_funds=None,
        )
    lump_amount = _minimum_funding(client, offer, rules, "lump")
    monthly_amount = _minimum_funding(client, offer, rules, "monthly")
    offer_total = offer_total_cents(offer)
    lump_limit = percent_cents(offer_total, "0.65")
    monthly_limit = max(10000, percent_cents(client.draft_amount_cents, "0.40"))
    next_day = client.as_of_date + timedelta(days=1)
    lump_date = next_day if next_day <= client.last_draft_date else client.last_draft_date
    draft_count = len(_future_drafts(client))
    return Result(
        feasible=False,
        pay_shape_used=None,
        schedule=None,
        additional_funds=AdditionalFunds(
            lump_sum=_funds_option(
                lump_amount,
                lump_limit,
                "lump",
                lump_date,
                draft_count,
            ),
            monthly_increment=_funds_option(
                monthly_amount,
                monthly_limit,
                "monthly",
                lump_date,
                draft_count,
            ),
        ),
    )
