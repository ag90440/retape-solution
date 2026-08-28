# Settlement engine

## Run

```bash
pip install -r requirements.txt
pytest -q
python run.py cases/case1_feasible_even
```

## Work flow

Money stays in integer cents. Percentage values use half-up rounding.

The engine builds the creditor cadence from the first payment date. It checks every allowed payment count.

Even plans split cents across the latest payments. Balloon plans keep early payments at the lowest valid value and put the rest in the last payment. Staircase plans use an integer model for the payment amount, token limit, tier floor, order, exact total, and segment limit.

The ledger is checked by date. Credits are added before debits. Old ledger rows are not used again.

The program fee can start on the first creditor date. The engine fixes the largest fee amount possible on each date before moving to the next date. Fee-only dates have no bank fee.

When a plan does not work, the same model finds the smallest lump sum and monthly increase.

## Assumptions

Both `creditor_balance_cents` and the old `current_balance_cents` offer field are accepted.

A lump sum is placed on the day after `as_of_date`.

Future draft dates come from the client draft schedule. The normal draft credits stay in the ledger.

A staircase segment is one distinct payment amount. A final one-month step is allowed when it stays inside the segment limit.

If a date or payment rule cannot be fixed with money, the amount is `0` and the reason says no amount can fix the offer.

## Choice

I first considered checking payment values one cent at a time. It became slow for large balances. The integer model keeps the checks exact and fast.
