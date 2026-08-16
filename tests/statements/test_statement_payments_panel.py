"""Payments panel + manual-settle integration tests.

Covers the statement page's provisional-payment feature:

- A provisional HDFC bill-payment SMS (notify-only, no ledger row) appears in
  the page's pending set, derived live by re-parsing.
- "Settle" promotes that provisional SMS into a real credit transaction and
  recomputes the cycle's paid amount exactly once.
- Once settled the SMS links to its transaction and drops off the pending set.
- A settled real credit shows in the settled set, not pending.

Card mask 1234 matches the default test CC account (account_number ...1234).
Amounts are synthetic.
"""

import datetime
from decimal import Decimal

import pytest
from sqlalchemy import select

from financial_dashboard.db import (
    SmsMessage,
    StatementUpload,
    Transaction,
)
from financial_dashboard.db.enums import PaymentStatus
from financial_dashboard.services.linker import build_link_context
from financial_dashboard.services.sms_pipeline import process_sms_row
from financial_dashboard.services.statement_payments import build_payments_view

from . import _helpers as h


# The provisional "received but not settled" HDFC template: no reference,
# carries available limit, day-granularity date.
_PROVISIONAL_BODY = (
    "DEAR HDFCBANK CARDMEMBER, PAYMENT OF Rs. 30000.00 RECEIVED TOWARDS "
    "YOUR CREDIT CARD ENDING WITH 1234 ON {day}-8-2026. "
    "YOUR AVAILABLE LIMIT IS RS. 100000.00"
)


@pytest.fixture
def _no_payment_tracking():
    """Noop override — the real reminders recompute logic runs."""
    yield


async def _seed_open_statement(maker, *, total="50,000.00", due="25/08/2026"):
    acc_id = await h.add_cc_account(maker, cards=["XXXX1234"])
    async with maker() as session:
        upload = StatementUpload(
            account_id=acc_id,
            bank="hdfc",
            filename="cc.pdf",
            file_path="/tmp/cc.pdf",
            status="imported",
            due_date=due,
            total_amount_due=total,
            payment_status=PaymentStatus.UNPAID,
            payment_paid_amount=Decimal("0"),
            created_at=datetime.datetime(2026, 8, 6, 6, 20, tzinfo=datetime.UTC),
        )
        session.add(upload)
        await session.commit()
        return upload.id, acc_id


async def _add_provisional_sms(maker, *, day=7, hour=6, minute=0):
    async with maker() as session:
        sms = SmsMessage(
            bank="hdfc",
            sender="AD-HDFCBK",
            body=_PROVISIONAL_BODY.format(day=day),
            received_at=datetime.datetime(
                2026, 8, day, hour, minute, tzinfo=datetime.UTC
            ),
            status="parsed",
        )
        session.add(sms)
        await session.commit()
        return sms.id


async def _get_upload(maker, upload_id):
    async with maker() as session:
        return await session.get(StatementUpload, upload_id)


@pytest.mark.anyio
async def test_provisional_sms_appears_as_pending(maker):
    """A provisional bill-payment SMS shows in the pending set, not settled."""
    upload_id, _ = await _seed_open_statement(maker)
    await _add_provisional_sms(maker, day=7)

    upload = await _get_upload(maker, upload_id)
    async with maker() as session:
        view = await build_payments_view(session, upload)

    assert view.settled == []
    assert len(view.pending) == 1
    pending = view.pending[0]
    assert pending.amount == "30,000.00"
    assert pending.card_mask.endswith("1234")


@pytest.mark.anyio
async def test_settle_promotes_provisional_to_real_credit(maker):
    """Settling a provisional SMS creates one real credit and moves paid."""
    upload_id, acc_id = await _seed_open_statement(maker)
    sms_id = await _add_provisional_sms(maker, day=7)

    # Promote via the same pipeline path the web handler uses.
    async with maker() as session, session.begin():
        sms = await session.get(SmsMessage, sms_id)
        link_ctx = await build_link_context(session)
        outcome = await process_sms_row(session, sms, link_ctx, settle_provisional=True)
    assert outcome.transaction_id is not None
    assert outcome.pending_payment_check is not None

    # Fire the recompute hook (web handler does this post-commit).
    from financial_dashboard.services.reminders import check_payment_received

    await check_payment_received(*outcome.pending_payment_check)

    # Exactly one credit row exists, linked to the SMS.
    async with maker() as session:
        credits = (
            (
                await session.execute(
                    select(Transaction).where(
                        Transaction.account_id == acc_id,
                        Transaction.direction == "credit",
                    )
                )
            )
            .scalars()
            .all()
        )
        assert len(credits) == 1
        sms = await session.get(SmsMessage, sms_id)
        assert sms.transaction_id == credits[0].id

    # Paid amount moved by the settled amount.
    upload = await _get_upload(maker, upload_id)
    assert Decimal(str(upload.payment_paid_amount)) == Decimal("30000.00")

    # The settled payment now shows in settled, and pending is empty.
    async with maker() as session:
        view = await build_payments_view(session, upload)
    assert len(view.settled) == 1
    assert view.settled[0].amount == "30,000.00"
    assert view.pending == []


@pytest.mark.anyio
async def test_settle_is_idempotent_on_linked_sms(maker):
    """A second settle of an already-linked SMS makes no second credit."""
    _, acc_id = await _seed_open_statement(maker)
    sms_id = await _add_provisional_sms(maker, day=7)

    async def _settle():
        async with maker() as session, session.begin():
            sms = await session.get(SmsMessage, sms_id)
            link_ctx = await build_link_context(session)
            return await process_sms_row(
                session, sms, link_ctx, settle_provisional=True
            )

    first = await _settle()
    assert first.transaction_id is not None
    # Second settle of the now-linked SMS must not create a second credit. The
    # merge matcher sees the identical existing credit and defers (no row).
    second = await _settle()
    assert second.transaction_id is None

    async with maker() as session:
        credits = (
            (
                await session.execute(
                    select(Transaction).where(
                        Transaction.account_id == acc_id,
                        Transaction.direction == "credit",
                    )
                )
            )
            .scalars()
            .all()
        )
        assert len(credits) == 1


@pytest.mark.anyio
async def test_two_same_amount_provisionals_both_pending(maker):
    """Two identical-amount same-day provisionals are two distinct pending
    rows (each its own SMS), not collapsed."""
    upload_id, _ = await _seed_open_statement(maker)
    await _add_provisional_sms(maker, day=7, minute=7)
    await _add_provisional_sms(maker, day=7, minute=11)

    upload = await _get_upload(maker, upload_id)
    async with maker() as session:
        view = await build_payments_view(session, upload)

    assert len(view.pending) == 2
    assert {p.amount for p in view.pending} == {"30,000.00"}


async def _add_settled_credit(maker, acc_id, *, amount, day=8):
    """A real bank-settled CC-payment credit already in the ledger."""
    async with maker() as session:
        txn = Transaction(
            account_id=acc_id,
            bank="hdfc",
            email_type="hdfc_cc_payment_received_alert",
            direction="credit",
            amount=Decimal(str(amount)),
            transaction_date=datetime.date(2026, 8, day),
            counterparty="Payment",
            reference_number="REF123",
        )
        session.add(txn)
        await session.commit()
        return txn.id


@pytest.mark.anyio
async def test_settled_credit_cancels_matching_provisional(maker):
    """A provisional whose real settlement already landed is not pending.

    The settlement is a different, linked credit row; the provisional stays
    unlinked. It must be cancelled against the settled credit, not shown as a
    phantom pending payment (which would double-count if settled)."""
    upload_id, acc_id = await _seed_open_statement(maker)
    await _add_provisional_sms(maker, day=7)
    await _add_settled_credit(maker, acc_id, amount="30000.00", day=8)

    upload = await _get_upload(maker, upload_id)
    async with maker() as session:
        view = await build_payments_view(session, upload)

    assert len(view.settled) == 1
    assert view.pending == []


@pytest.mark.anyio
async def test_two_provisionals_one_settled_leaves_one_pending(maker):
    """Two equal provisionals with one real settlement: exactly one stays
    pending (the settled budget cancels only one)."""
    upload_id, acc_id = await _seed_open_statement(maker)
    await _add_provisional_sms(maker, day=7, minute=7)
    await _add_provisional_sms(maker, day=7, minute=11)
    await _add_settled_credit(maker, acc_id, amount="30000.00", day=8)

    upload = await _get_upload(maker, upload_id)
    async with maker() as session:
        view = await build_payments_view(session, upload)

    assert len(view.settled) == 1
    assert len(view.pending) == 1


@pytest.mark.anyio
async def test_is_settleable_rejects_non_provisional(maker):
    """The settle validator rejects a non-payment SMS and a wrong-bank SMS."""
    from financial_dashboard.services.statement_payments import (
        is_settleable_provisional,
    )

    upload_id, _ = await _seed_open_statement(maker)
    upload = await _get_upload(maker, upload_id)

    # A plain spend SMS (not a bill payment) must be rejected.
    async with maker() as session:
        spend = SmsMessage(
            bank="hdfc",
            sender="VK-HDFCBK",
            body="Spent Rs.500 From HDFC Bank Card x1234 At Zomato On 2026-08-07:14:23:00 Bal Rs.1000",
            received_at=datetime.datetime(2026, 8, 7, 9, 0, tzinfo=datetime.UTC),
            status="parsed",
        )
        session.add(spend)
        await session.commit()
        spend_id = spend.id

    async with maker() as session:
        sms = await session.get(SmsMessage, spend_id)
        assert await is_settleable_provisional(session, sms, upload) is False


@pytest.mark.anyio
async def test_maskless_provisional_rejected_when_card_unmatched(maker):
    """A provisional whose parsed card does not match the account is rejected
    even for an otherwise-valid payment shape (strict card check on settle)."""
    from financial_dashboard.services.statement_payments import (
        is_settleable_provisional,
    )

    upload_id, _ = await _seed_open_statement(maker)  # card ...1234
    upload = await _get_upload(maker, upload_id)

    # A provisional for a DIFFERENT card (9999) on the same bank.
    async with maker() as session:
        other = SmsMessage(
            bank="hdfc",
            sender="AD-HDFCBK",
            body=(
                "DEAR HDFCBANK CARDMEMBER, PAYMENT OF Rs. 30000.00 RECEIVED "
                "TOWARDS YOUR CREDIT CARD ENDING WITH 9999 ON 7-8-2026. "
                "YOUR AVAILABLE LIMIT IS RS. 100000.00"
            ),
            received_at=datetime.datetime(2026, 8, 7, 9, 0, tzinfo=datetime.UTC),
            status="parsed",
        )
        session.add(other)
        await session.commit()
        other_id = other.id

    async with maker() as session:
        sms = await session.get(SmsMessage, other_id)
        assert await is_settleable_provisional(session, sms, upload) is False
