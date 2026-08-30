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
    # A second settle reuses the same equal-balance credit.
    second = await _settle()
    assert second.transaction_id == first.transaction_id

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


async def _add_settled_credit(
    maker, acc_id, *, amount, day=8, ref="REF123", card_mask="1234"
):
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
            reference_number=ref,
            card_mask=card_mask,
        )
        session.add(txn)
        await session.commit()
        return txn.id


@pytest.mark.anyio
async def test_a_settlement_the_next_day_hides_its_provisional(maker):
    """The bank settles a day or two later. That is the normal shape.

    The old code required the two to look simultaneous. That left completed
    payments on the page. One provisional and one credit of the same amount,
    on the same card, inside the settlement window, are the same payment.
    """
    upload_id, acc_id = await _seed_open_statement(maker)
    await _add_provisional_sms(maker, day=7)
    await _add_settled_credit(maker, acc_id, amount="30000.00", day=8)

    upload = await _get_upload(maker, upload_id)
    async with maker() as session:
        view = await build_payments_view(session, upload)

    assert len(view.settled) == 1
    assert view.pending == []


@pytest.mark.anyio
async def test_a_credit_before_the_provisional_hides_nothing(maker):
    """A settlement never precedes the payment it settles."""
    upload_id, acc_id = await _seed_open_statement(maker)
    await _add_provisional_sms(maker, day=7)
    await _add_settled_credit(maker, acc_id, amount="30000.00", day=6)

    upload = await _get_upload(maker, upload_id)
    async with maker() as session:
        view = await build_payments_view(session, upload)

    assert len(view.pending) == 1


@pytest.mark.anyio
async def test_a_credit_long_after_the_provisional_hides_nothing(maker):
    """Beyond the window the credit is a different payment."""
    upload_id, acc_id = await _seed_open_statement(maker)
    await _add_provisional_sms(maker, day=7)
    await _add_settled_credit(maker, acc_id, amount="30000.00", day=14)

    upload = await _get_upload(maker, upload_id)
    async with maker() as session:
        view = await build_payments_view(session, upload)

    assert len(view.pending) == 1


@pytest.mark.anyio
async def test_a_credit_read_off_the_statement_also_settles(maker):
    """Some payments reach the ledger only as a statement credit.

    They mean the money arrived just as much as a bank alert does, and the
    displayed settled list excludes them, so they must be looked up separately.
    """
    upload_id, acc_id = await _seed_open_statement(maker)
    await _add_provisional_sms(maker, day=7)
    async with maker() as session:
        session.add(
            Transaction(
                account_id=acc_id,
                bank="hdfc",
                email_type="cc_statement",
                direction="credit",
                amount=Decimal("30000.00"),
                transaction_date=datetime.date(2026, 8, 8),
                counterparty="CC PAYMENT",
                card_mask="1234",
                category="credit_card_payment",
            )
        )
        await session.commit()

    upload = await _get_upload(maker, upload_id)
    async with maker() as session:
        view = await build_payments_view(session, upload)

    # The statement credit is not a bank alert, so the page does not list it.
    assert view.settled == []
    assert view.pending == []


@pytest.mark.anyio
async def test_two_provisionals_and_two_settlements_hide_both(maker):
    """Same amount twice is indistinguishable, but the counts agree."""
    upload_id, acc_id = await _seed_open_statement(maker)
    await _add_provisional_sms(maker, day=7, minute=7)
    await _add_provisional_sms(maker, day=7, minute=11)
    await _add_settled_credit(maker, acc_id, amount="30000.00", day=8, ref="REF-A")
    await _add_settled_credit(maker, acc_id, amount="30000.00", day=8, ref="REF-B")

    upload = await _get_upload(maker, upload_id)
    async with maker() as session:
        view = await build_payments_view(session, upload)

    assert len(view.settled) == 2
    assert view.pending == []


@pytest.mark.anyio
async def test_two_provisionals_one_settled_keeps_both_pending(maker):
    """One credit cannot settle two payments.

    Which of the two it settled is unknowable, so neither is hidden. Hiding an
    arbitrary one would hide a genuinely unpaid payment half the time.
    """
    upload_id, acc_id = await _seed_open_statement(maker)
    await _add_provisional_sms(maker, day=7, minute=7)
    await _add_provisional_sms(maker, day=7, minute=11)
    await _add_settled_credit(maker, acc_id, amount="30000.00", day=8)

    upload = await _get_upload(maker, upload_id)
    async with maker() as session:
        view = await build_payments_view(session, upload)

    assert len(view.settled) == 1
    assert len(view.pending) == 2


@pytest.mark.anyio
async def test_confident_settled_match_hides_provisional(maker):
    """Equal authoritative balance and identity hide the settled provisional."""
    upload_id, acc_id = await _seed_open_statement(maker)
    await _add_provisional_sms(maker, day=7, hour=6, minute=0)
    async with maker() as session:
        session.add(
            Transaction(
                account_id=acc_id,
                bank="hdfc",
                email_type="hdfc_cc_payment_received_alert",
                direction="credit",
                amount=Decimal("30000.00"),
                currency="INR",
                transaction_date=datetime.date(2026, 8, 7),
                transaction_time=datetime.time(11, 30),
                card_mask="1234",
                balance=Decimal("100000.00"),
                source="email",
            )
        )
        await session.commit()

    upload = await _get_upload(maker, upload_id)
    async with maker() as session:
        view = await build_payments_view(session, upload)

    assert len(view.settled) == 1
    assert view.pending == []


async def _seed_second_statement(maker, acc_id, *, created_at, total="50,000.00"):
    """A later statement for the SAME account, opening the next cycle."""

    async with maker() as session:
        upload = StatementUpload(
            account_id=acc_id,
            bank="hdfc",
            filename="cc2.pdf",
            file_path="/tmp/cc2.pdf",
            status="imported",
            due_date="25/09/2026",
            total_amount_due=total,
            payment_status=PaymentStatus.UNPAID,
            payment_paid_amount=Decimal("0"),
            created_at=created_at,
        )
        session.add(upload)
        await session.commit()
        return upload.id


async def _add_null_date_credit(maker, acc_id, *, amount, created_at):
    """A date-less real CC-payment credit with a controlled created_at."""
    async with maker() as session:
        txn = Transaction(
            account_id=acc_id,
            bank="hdfc",
            email_type="hdfc_cc_payment_received_alert",
            direction="credit",
            amount=Decimal(str(amount)),
            transaction_date=None,
            counterparty="Payment",
        )
        txn.created_at = created_at
        session.add(txn)
        await session.commit()
        return txn.id


@pytest.mark.anyio
async def test_null_date_credit_bounded_to_its_cycle_by_created_at(maker):
    """A date-less settled credit created after the next statement shows in the
    NEXT cycle's settled list only, never the old one. The old cycle bounds
    date-less rows by ``created_at`` (< the next statement's ``created_at``), so
    it does not double-count."""
    old_id, acc_id = await _seed_open_statement(maker)  # created 2026-08-06
    newer_id = await _seed_second_statement(
        maker,
        acc_id,
        created_at=datetime.datetime(2026, 9, 6, 6, 20, tzinfo=datetime.UTC),
    )
    await _add_null_date_credit(
        maker,
        acc_id,
        amount="30000.00",
        created_at=datetime.datetime(2026, 9, 10, 6, 0, tzinfo=datetime.UTC),
    )

    old_upload = await _get_upload(maker, old_id)
    newer_upload = await _get_upload(maker, newer_id)
    async with maker() as session:
        old_view = await build_payments_view(session, old_upload)
        newer_view = await build_payments_view(session, newer_upload)

    assert old_view.settled == []
    assert len(newer_view.settled) == 1
    assert newer_view.settled[0].amount == "30,000.00"


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


@pytest.mark.anyio
async def test_a_credit_on_another_card_hides_nothing(maker):
    """A second card's payment must not settle this card's."""
    upload_id, acc_id = await _seed_open_statement(maker)
    await _add_provisional_sms(maker, day=7)
    await _add_settled_credit(maker, acc_id, amount="30000.00", day=8, card_mask="9999")

    upload = await _get_upload(maker, upload_id)
    async with maker() as session:
        view = await build_payments_view(session, upload)

    assert len(view.pending) == 1


@pytest.mark.anyio
async def test_a_maskless_credit_hides_nothing(maker):
    """An unknown card is not evidence. It must not settle anything."""
    upload_id, acc_id = await _seed_open_statement(maker)
    await _add_provisional_sms(maker, day=7)
    await _add_settled_credit(maker, acc_id, amount="30000.00", day=8, card_mask=None)

    upload = await _get_upload(maker, upload_id)
    async with maker() as session:
        view = await build_payments_view(session, upload)

    assert len(view.pending) == 1


@pytest.mark.anyio
async def test_a_statement_cashback_hides_nothing(maker):
    """A statement imports cashback too. Cashback settles no bill."""
    upload_id, acc_id = await _seed_open_statement(maker)
    await _add_provisional_sms(maker, day=7)
    async with maker() as session:
        session.add(
            Transaction(
                account_id=acc_id,
                bank="hdfc",
                email_type="cc_statement",
                direction="credit",
                amount=Decimal("30000.00"),
                transaction_date=datetime.date(2026, 8, 8),
                counterparty="CARD CASHBACK CREDIT",
                card_mask="1234",
                category="cashback_rewards",
            )
        )
        await session.commit()

    upload = await _get_upload(maker, upload_id)
    async with maker() as session:
        view = await build_payments_view(session, upload)

    assert len(view.pending) == 1


@pytest.mark.anyio
async def test_each_provisional_needs_its_own_dated_credit(maker):
    """Two credits near the first payment do not settle a much later one."""
    upload_id, acc_id = await _seed_open_statement(maker)
    await _add_provisional_sms(maker, day=5)
    await _add_provisional_sms(maker, day=14)
    await _add_settled_credit(maker, acc_id, amount="30000.00", day=6, ref="REF-A")
    await _add_settled_credit(maker, acc_id, amount="30000.00", day=7, ref="REF-B")

    upload = await _get_upload(maker, upload_id)
    async with maker() as session:
        view = await build_payments_view(session, upload)

    assert len(view.pending) == 2
