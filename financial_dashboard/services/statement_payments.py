"""Payment view-model for the CC statement detail page.

The statement page shows three things for one CC cycle:
- a payment-status line (paid / total / remaining),
- the real credit transactions received against the cycle (settled), and
- provisional bill-payment SMS not yet in the ledger (pending), each with a
  manual "settle" action.

A provisional CC-payment SMS is notify-only in the pipeline: it makes no
Transaction row. So the pending set is not a stored table. It is derived live
by re-parsing the account's unlinked SMS in the cycle window and keeping the
ones the parser calls a provisional credit. Once a row settles it links to a
Transaction and drops off the pending set on its own. This is deliberately
self-healing: no staging table to clean up.
"""

import logging
from dataclasses import dataclass
from typing import NamedTuple
from datetime import date as datetime_date, datetime, timedelta, timezone
from decimal import Decimal, InvalidOperation

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from bank_sms_parser import ParseError, UnsupportedSmsTypeError, parse_sms

from financial_dashboard.db import (
    Account,
    SmsMessage,
    StatementUpload,
    Transaction,
)
from financial_dashboard.services.cc_cycle import (
    CycleWindow,
    cc_cycle_window,
    transactions_in_cycle,
)
from financial_dashboard.services.cc_disambiguation import (
    is_cc_payment_received_email,
)
from financial_dashboard.services.sms_pipeline import parsed_sms_to_txn_data
from financial_dashboard.services.statements.cc import parse_cc_amount

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SettledPayment:
    """A real credit transaction counted against this cycle."""

    txn_id: int
    date: str
    amount: str
    counterparty: str


@dataclass(frozen=True)
class PendingPayment:
    """A provisional bill-payment SMS not yet in the ledger."""

    sms_id: int
    amount: str
    date: str
    card_mask: str
    available_limit: str | None


# The email_type a credit carries when it was read off the statement PDF rather
# than from a bank alert.
_STATEMENT_CREDIT_EMAIL_TYPE = "cc_statement"

# A bank settles a bill payment on its own schedule. HDFC has taken up to two
# days, so a settlement is looked for on the provisional's day and the four
# days after it.
_SETTLEMENT_WINDOW_DAYS = 4

# The category the classifier gives a credit that paid a card bill. A statement
# also imports cashback and refund credits, and those settle nothing.
_CARD_PAYMENT_CATEGORY = "credit_card_payment"


class _Settlement(NamedTuple):
    """A credit that could be the settlement of some provisional payment."""

    amount: Decimal
    card_last4: str
    date: datetime_date | None


@dataclass(frozen=True)
class PaymentsView:
    """Everything the statement page's Payments panel needs."""

    settled: list[SettledPayment]
    pending: list[PendingPayment]
    remaining: str | None


def _fmt_amount(value) -> str:
    return f"{Decimal(str(value)):,.2f}"


async def _settled_payments(
    session: AsyncSession,
    upload: StatementUpload,
    window: CycleWindow,
) -> list[SettledPayment]:
    """Real credit transactions that count toward this cycle's paid amount.

    Mirrors ``_qualifying_payment_credit_sum``: credit rows the classifier calls
    a CC bill-payment, in the same due-date-anchored cycle window.
    """
    rows = (
        (
            await session.execute(
                select(Transaction)
                .where(
                    Transaction.account_id == upload.account_id,
                    Transaction.direction == "credit",
                    transactions_in_cycle(window),
                )
                .order_by(Transaction.transaction_date.asc(), Transaction.id.asc())
            )
        )
        .scalars()
        .all()
    )

    settled: list[SettledPayment] = []
    for row in rows:
        if not is_cc_payment_received_email(row.email_type):
            continue
        settled.append(
            SettledPayment(
                txn_id=row.id,
                date=row.transaction_date.isoformat() if row.transaction_date else "",
                amount=_fmt_amount(row.amount),
                counterparty=row.counterparty or "",
            )
        )
    return settled


async def _pending_payments(
    session: AsyncSession,
    upload: StatementUpload,
    window: CycleWindow,
) -> list[PendingPayment]:
    """Provisional bill-payment SMS for this card, not yet in the ledger.

    Fetches the account's unlinked SMS in the cycle window, then re-parses each
    body. Keeps only the ones the parser calls a provisional credit that the
    classifier reads as a CC bill-payment. Re-parsing is required: the parser's
    ledger role is not stored on the SMS row. The candidate set is small (one
    card's unlinked SMS in one cycle), so parsing on page load is cheap.

    A provisional whose real settlement already landed must not show as pending.
    The settlement is a different SMS linked to its own credit row, so the
    provisional stays unlinked. Suppress it only when a credit of the same
    amount, on the same card, is dated inside the settlement window.
    """
    account = await session.get(Account, upload.account_id)
    if account is None:
        return []

    # ``received_at`` is UTC; the cycle bounds are calendar dates whose real
    # day is IST. A payment near IST-midnight is stored on the adjacent UTC
    # date, so a tight UTC-midnight window would drop it. Pad the window by a
    # day on each edge to cover the ~5.5h skew. Over-inclusion is safe: a
    # provisional pulled in from an adjacent cycle still needs its own
    # settlement to be absent;
    # the worst case is showing a genuinely-unsettled neighbour payment, which
    # settling attributes to the correct cycle anyway. A row still needs a
    # matching card and ``transaction_id IS NULL``.
    lower = datetime.combine(
        window.start - timedelta(days=1), datetime.min.time(), tzinfo=timezone.utc
    )
    conditions = [
        func.lower(SmsMessage.bank) == account.bank.lower(),
        SmsMessage.transaction_id.is_(None),
        SmsMessage.received_at >= lower,
    ]
    if window.end is not None:
        upper = datetime.combine(
            window.end + timedelta(days=1), datetime.min.time(), tzinfo=timezone.utc
        )
        conditions.append(SmsMessage.received_at < upper)

    rows = (
        (
            await session.execute(
                select(SmsMessage)
                .where(*conditions)
                .order_by(SmsMessage.received_at.asc())
            )
        )
        .scalars()
        .all()
    )

    candidates = []
    for sms in rows:
        parsed = _parse_provisional_cc_payment(sms)
        if parsed is None:
            continue
        txn_data, ledger_role = parsed
        # Only true provisionals settle here. A restatement echoes a payment
        # another message already carries, so it must not become a ledger row.
        if ledger_role != "provisional":
            continue
        if not is_cc_payment_received_email(txn_data.get("email_type")):
            continue
        # A provisional SMS for a DIFFERENT card on the same bank must not show
        # here. Confirm the parsed card belongs to this account's card set.
        if not _card_belongs_to_account(txn_data.get("card_mask"), account):
            continue
        candidates.append((sms, txn_data))

    settlements = await _settlements_for_suppression(
        session, upload, window, await _provisional_sms_ids(session, upload, window)
    )
    hidden = _settled_sms_ids(candidates, settlements)

    pending: list[PendingPayment] = []
    for sms, txn_data in candidates:
        if sms.id in hidden:
            continue
        txn_date = txn_data.get("transaction_date")
        balance = txn_data.get("balance")
        pending.append(
            PendingPayment(
                sms_id=sms.id,
                amount=_fmt_amount(txn_data["amount"]),
                date=txn_date.isoformat() if txn_date else "",
                card_mask=txn_data.get("card_mask") or "",
                available_limit=_fmt_amount(balance) if balance is not None else None,
            )
        )
    return pending


def _last4(value: str | None) -> str:
    return "".join(ch for ch in (value or "") if ch.isdigit())[-4:]


async def _provisional_sms_ids(
    session: AsyncSession,
    upload: StatementUpload,
    window: CycleWindow,
) -> set[int]:
    """Every provisional payment SMS in the cycle, linked or not.

    A provisional settled by hand becomes a linked credit. That credit must not
    then be read as the settlement of some other provisional, so the ids are
    needed whether or not the row still counts as pending.
    """
    account = await session.get(Account, upload.account_id)
    if account is None:
        return set()
    lower = datetime.combine(
        window.start - timedelta(days=1), datetime.min.time(), tzinfo=timezone.utc
    )
    conditions = [
        func.lower(SmsMessage.bank) == account.bank.lower(),
        SmsMessage.received_at >= lower,
    ]
    if window.end is not None:
        upper = datetime.combine(
            window.end + timedelta(days=_SETTLEMENT_WINDOW_DAYS),
            datetime.min.time(),
            tzinfo=timezone.utc,
        )
        conditions.append(SmsMessage.received_at < upper)

    rows = (
        (await session.execute(select(SmsMessage).where(*conditions))).scalars().all()
    )
    ids: set[int] = set()
    for sms in rows:
        parsed = _parse_provisional_cc_payment(sms)
        if parsed is not None and parsed[1] == "provisional":
            ids.add(sms.id)
    return ids


async def _settlements_for_suppression(
    session: AsyncSession,
    upload: StatementUpload,
    window: CycleWindow,
    provisional_sms_ids: set[int],
) -> list[_Settlement]:
    """Credits that could settle a provisional payment in this cycle.

    Wider than the settled list the page shows. A payment can reach the ledger
    as a bank alert or as a credit read off the statement PDF, and both mean the
    money arrived. The displayed list keeps only the alerts, so it cannot answer
    this question on its own.
    """
    # A payment made on the cycle's last day settles after the cycle closes, so
    # the search runs past the end. The cycle predicate alone would miss it.
    upper = None
    if window.end is not None:
        upper = window.end + timedelta(days=_SETTLEMENT_WINDOW_DAYS)
    conditions = [
        Transaction.account_id == upload.account_id,
        Transaction.direction == "credit",
        Transaction.transaction_date >= window.start - timedelta(days=1),
    ]
    if upper is not None:
        conditions.append(Transaction.transaction_date < upper)

    rows = (
        (await session.execute(select(Transaction).where(*conditions))).scalars().all()
    )

    out: list[_Settlement] = []
    for row in rows:
        if is_cc_payment_received_email(row.email_type):
            pass
        elif (
            row.email_type == _STATEMENT_CREDIT_EMAIL_TYPE
            and row.category == _CARD_PAYMENT_CATEGORY
        ):
            # A statement imports cashback and refund credits too. Only the ones
            # the classifier calls a card payment settle anything.
            pass
        else:
            continue
        if row.sms_message_id in provisional_sms_ids:
            # This credit IS a provisional that somebody settled by hand. It
            # cannot also be the settlement of a different provisional.
            continue
        out.append(
            _Settlement(
                amount=Decimal(str(row.amount)),
                card_last4=_last4(row.card_mask),
                date=row.transaction_date,
            )
        )
    return out


def _settled_sms_ids(candidates, settlements: list[_Settlement]) -> set[int]:
    """Say which provisionals already have their settlement in the ledger.

    Two provisionals for the same card and amount are indistinguishable, so the
    decision is made per group. Every member must get its own credit, dated
    inside that member's own window. A group where even one member goes without
    stays wholly visible: one of them is genuinely unpaid, and hiding an
    arbitrary one would hide the wrong payment half the time.
    """
    groups: dict[tuple, list] = {}
    for sms, txn_data in candidates:
        key = (_last4(txn_data.get("card_mask")), Decimal(str(txn_data["amount"])))
        groups.setdefault(key, []).append((sms, txn_data))

    unused = list(settlements)
    hidden: set[int] = set()
    for (card_last4, amount), members in sorted(groups.items()):
        claimed = _claim_one_each(members, amount, card_last4, unused)
        if claimed is None:
            continue
        for used in claimed:
            unused.remove(used)
        hidden.update(sms.id for sms, _ in members)
    return hidden


def _claim_one_each(members, amount, card_last4, unused):
    """Give every member its own credit, or return None.

    Members are taken oldest first, and each takes the earliest credit it can
    use. A later member can only want a later credit, so taking the earliest
    never denies one that a member behind it needed.
    """
    available = sorted(
        (s for s in unused if s.amount == amount and _card_matches(card_last4, s)),
        key=lambda s: (s.date is None, s.date),
    )
    claimed = []
    for _, txn_data in sorted(
        members,
        key=lambda m: (
            m[1].get("transaction_date") is None,
            m[1].get("transaction_date"),
        ),
    ):
        on = txn_data.get("transaction_date")
        for candidate in available:
            if _within_settlement_window(candidate.date, on):
                claimed.append(candidate)
                available.remove(candidate)
                break
        else:
            return None
    return claimed


def _card_matches(card_last4: str, settlement: _Settlement) -> bool:
    """A credit settles a card's payment only when both name that card.

    An unknown last-4 on either side is not evidence of a match. Treating it as
    one lets a second card's credit hide this card's payment.
    """
    return bool(card_last4) and card_last4 == settlement.card_last4


def _within_settlement_window(settled_on, provisional_on) -> bool:
    """True if a credit is dated on or just after the provisional.

    A settlement never precedes the payment it settles, and the bank does not
    take forever. An undated row on either side proves nothing, so it does not
    match.
    """
    if settled_on is None or provisional_on is None:
        return False
    return 0 <= (settled_on - provisional_on).days <= _SETTLEMENT_WINDOW_DAYS


def _parse_provisional_cc_payment(sms: SmsMessage):
    """Re-parse one SMS. Return ``(txn_data, ledger_role)`` or ``None``.

    Side-effect free. Swallows parser errors: a body that no longer parses is
    simply not a pending payment.
    """
    received_at = sms.received_at
    if received_at is not None and received_at.tzinfo is None:
        received_at = received_at.replace(tzinfo=timezone.utc)
    try:
        parsed = parse_sms(
            sms.bank, sms.body, sender=sms.sender, received_at=received_at
        )
    except ParseError, UnsupportedSmsTypeError:
        return None

    txn_data = parsed_sms_to_txn_data(parsed, sms)
    if txn_data is None or txn_data.get("direction") != "credit":
        return None
    return txn_data, parsed.ledger_role


def _account_masks(account: Account) -> list[str]:
    masks = [account.account_number or ""]
    masks.extend(card.card_mask or "" for card in getattr(account, "cards", []) or [])
    return masks


def _card_belongs_to_account(
    card_mask: str | None, account: Account, *, strict: bool = False
) -> bool:
    """True if a parsed card mask matches this account or one of its cards.

    ``strict=False`` (display path): a missing or unreadable mask is treated as
    belonging. The pipeline resolves such rows to the single-CC account anyway,
    and excluding them would hide a real pending payment.

    ``strict=True`` (settle path): a missing or unreadable mask must NOT match
    an arbitrary account — settling creates a real credit. Such a mask is
    accepted only when the account has exactly one known card mask (a single-CC
    account where the payment unambiguously belongs), and rejected otherwise.
    """
    readable = [
        m4
        for m in _account_masks(account)
        if (m4 := "".join(ch for ch in m if ch.isdigit())[-4:])
    ]
    last4 = "".join(ch for ch in (card_mask or "") if ch.isdigit())[-4:]
    if not last4:
        if not strict:
            return True
        # Strict: accept a maskless payment only for an unambiguous single card.
        return len(set(readable)) == 1
    return last4 in readable


async def is_settleable_provisional(
    session: AsyncSession, sms: SmsMessage, upload: StatementUpload
) -> bool:
    """True if ``sms`` is a provisional CC bill-payment for ``upload``'s account.

    The settle route accepts a raw SMS id from the form, so it must confirm the
    SMS really is a provisional CC-payment credit for this statement's card
    before promoting it. This rejects a restatement, a non-payment SMS, a
    different card, or an arbitrary id. It does NOT re-check the cycle window:
    settling a slightly out-of-window provisional the user can see is harmless
    and the paid-amount recompute scopes the credit to the right cycle anyway.
    """
    if sms.transaction_id is not None:
        return False
    account = await session.get(Account, upload.account_id)
    if account is None or account.type != "credit_card":
        return False
    if sms.bank.casefold() != account.bank.casefold():
        return False
    parsed = _parse_provisional_cc_payment(sms)
    if parsed is None:
        return False
    txn_data, ledger_role = parsed
    if ledger_role != "provisional":
        return False
    if not is_cc_payment_received_email(txn_data.get("email_type")):
        return False
    return _card_belongs_to_account(txn_data.get("card_mask"), account, strict=True)


async def build_payments_view(
    session: AsyncSession, upload: StatementUpload
) -> PaymentsView:
    """Assemble the Payments panel view-model for one CC statement.

    Returns empty settled/pending and no remaining when the statement has no
    parseable total due (the panel then renders only what it can).
    """
    window = await cc_cycle_window(session, upload)
    settled = await _settled_payments(session, upload, window)
    pending = await _pending_payments(session, upload, window)

    remaining: str | None = None
    if upload.total_amount_due is not None:
        try:
            amount_due = parse_cc_amount(upload.total_amount_due)
        except ValueError, InvalidOperation:
            amount_due = None
        if amount_due is not None:
            paid = Decimal(str(upload.payment_paid_amount or 0))
            remaining = _fmt_amount(max(amount_due - paid, Decimal("0")))

    return PaymentsView(settled=settled, pending=pending, remaining=remaining)
