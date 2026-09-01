"""Bank-internal credit rows on a CC statement must not become transactions.

HSBC prints each billed EMI instalment twice in ``PURCHASES & INSTALLMENTS``:
a ``CR`` row that moves the instalment off the loan ledger, then the debit
that bills it. cc-parser keeps the ``CR`` row in ``payments_refunds`` for
observability and tags it ``credit_reasons="emi_installment_transfer"``. The
reconciler must drop that row; importing it creates a phantom
credit-card-payment credit that mirrors the instalment debit.
"""

from types import SimpleNamespace

import pytest
from cc_parser.parsers.models import Transaction as CcTransaction
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from financial_dashboard.db import Account, Base, StatementUpload, Transaction
from financial_dashboard.services.statements import cc as cc_module
from financial_dashboard.services.statements.cc import (
    import_missing_cc_txns,
    is_internal_transfer_credit,
    load_account_card_masks,
    reconcile_statement,
)

ACCOUNT_ID = 1
CARD_NUMBER = "4000XXXXXXXX0001"
INSTALMENT = "TEST MERCHANT CC000000000000 1ST OF 3 INSTALLMENTS PRINCIPAL"


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture
async def session_factory(monkeypatch):
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    maker = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    monkeypatch.setattr(cc_module, "async_session", maker)
    yield maker
    await engine.dispose()


def _row(direction: str, credit_reasons: str | None = None) -> CcTransaction:
    return CcTransaction(
        date="15/07/2026",
        narration=INSTALMENT,
        amount="300.00",
        card_number=CARD_NUMBER,
        transaction_type=direction,
        credit_reasons=credit_reasons,
    )


def _parsed(debits: list[CcTransaction], credits: list[CcTransaction]):
    return SimpleNamespace(
        bank="hsbc",
        transactions=debits,
        payments_refunds=credits,
        payments_refunds_total="0.00",
        card_summaries=[],
        possible_adjustment_pairs=[],
        overall_total="0.00",
        overall_reward_points="0",
    )


def test_only_the_tagged_transfer_is_internal():
    assert is_internal_transfer_credit(_row("credit", "emi_installment_transfer"))
    assert not is_internal_transfer_credit(_row("credit", "cr_marker"))
    assert not is_internal_transfer_credit(_row("credit"))
    assert not is_internal_transfer_credit(SimpleNamespace(narration="x"))


@pytest.mark.anyio
async def test_emi_transfer_credit_is_neither_matched_nor_imported(session_factory):
    async with session_factory() as session:
        session.add(
            Account(
                id=ACCOUNT_ID,
                bank="hsbc",
                label="HSBC Credit Card",
                type="credit_card",
                account_number=CARD_NUMBER,
            )
        )
        await session.commit()

    parsed = _parsed(
        debits=[_row("debit")],
        credits=[
            _row("credit", "emi_installment_transfer"),
            # A genuine credit on the same day and amount still imports.
            CcTransaction(
                date="15/07/2026",
                narration="BBPS PMT TESTREF",
                amount="300.00",
                card_number=CARD_NUMBER,
                transaction_type="credit",
                credit_reasons="cr_marker",
            ),
        ],
    )

    async with session_factory() as session:
        card_masks = await load_account_card_masks(session, ACCOUNT_ID)
    recon = reconcile_statement(parsed, [], ACCOUNT_ID, card_masks)

    assert recon["matched"] == []
    assert [(e["direction"], e["narration"]) for e in recon["missing"]] == [
        ("debit", INSTALMENT),
        ("credit", "BBPS PMT TESTREF"),
    ]

    async with session_factory() as session:
        upload = StatementUpload(
            account_id=ACCOUNT_ID,
            bank="hsbc",
            filename="statement.pdf",
            file_path="/nonexistent/statement.pdf",
            status="parsed",
        )
        session.add(upload)
        await session.flush()
        account = await session.get(Account, ACCOUNT_ID)
        await import_missing_cc_txns(session, upload, parsed, account, recon)
        await session.commit()

    async with session_factory() as session:
        rows = list((await session.execute(select(Transaction))).scalars().all())
    assert sorted((r.direction, r.counterparty) for r in rows) == [
        ("credit", "BBPS PMT TESTREF"),
        ("debit", INSTALMENT),
    ]
