"""Tests for POST /api/transactions/{id}/exclude.

The endpoint sets the flag to a given value, not a toggle. A repeated request
lands the same state. The flag drops a row from the cashflow report only. The
row keeps its category and every other view. Each test reads the state back
through the API, not the database.
"""

import datetime
from decimal import Decimal

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

import financial_dashboard.core.deps as core_deps
from financial_dashboard.api import router as api_router
from financial_dashboard.core.deps import get_session
from financial_dashboard.db import Base, Transaction


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture
async def session_maker(monkeypatch):
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    maker = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    monkeypatch.setattr(core_deps, "async_session", maker)
    yield maker
    await engine.dispose()


def _build_test_app(maker):
    app = FastAPI()
    app.include_router(api_router)

    async def _override():
        async with maker() as s:
            yield s

    app.dependency_overrides[get_session] = _override
    return app


async def _seed(maker) -> int:
    async with maker() as session:
        txn = Transaction(
            bank="hdfc",
            email_type="x",
            direction="debit",
            amount=Decimal("24995"),
            currency="INR",
            category="shopping",
            transaction_date=datetime.date(2026, 6, 12),
        )
        session.add(txn)
        await session.commit()
        return txn.id


async def _read_flag(client, txn_id: int) -> bool:
    """Read the flag back through the list API, the way a client sees it."""
    r = await client.get(f"/api/transactions?transaction_id={txn_id}")
    assert r.status_code == 200, r.text
    return r.json()["items"][0]["exclude_from_cashflow"]


@pytest.mark.anyio
async def test_set_and_clear_is_observable_and_idempotent(session_maker):
    txn_id = await _seed(session_maker)
    app = _build_test_app(session_maker)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        # A fresh row reads as included.
        assert await _read_flag(c, txn_id) is False

        # Set it, twice. The repeat lands the same state. It does not toggle.
        for _ in range(2):
            r = await c.post(
                f"/api/transactions/{txn_id}/exclude",
                json={"exclude_from_cashflow": True},
            )
            assert r.status_code == 200, r.text
            assert r.json() == {"ok": True, "exclude_from_cashflow": True}
        assert await _read_flag(c, txn_id) is True

        # Clear it.
        r = await c.post(
            f"/api/transactions/{txn_id}/exclude",
            json={"exclude_from_cashflow": False},
        )
        assert r.status_code == 200, r.text
        assert r.json() == {"ok": True, "exclude_from_cashflow": False}
        assert await _read_flag(c, txn_id) is False


@pytest.mark.anyio
async def test_missing_transaction_is_404(session_maker):
    app = _build_test_app(session_maker)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        r = await c.post(
            "/api/transactions/999999/exclude",
            json={"exclude_from_cashflow": True},
        )
        assert r.status_code == 404
