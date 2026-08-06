"""Display helpers for transaction tables."""

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from financial_dashboard.db import Transaction
from financial_dashboard.services.cashflow.buckets import LABEL_OVERRIDES


def category_label(slug: str) -> str:
    """Return a user-facing label for a category slug."""
    if slug in LABEL_OVERRIDES:
        return LABEL_OVERRIDES[slug]
    return slug.replace("_", " ").title()


async def hydrate_reconciliation_transactions(
    session: AsyncSession, recon: dict
) -> None:
    """Add current transaction display fields to reconciliation entries."""
    referenced_entries: list[tuple[dict, int]] = []
    for entry in recon.get("matched", []):
        transaction_id = entry.get("db_txn_id")
        if isinstance(transaction_id, int):
            referenced_entries.append((entry, transaction_id))
    for entry in recon.get("missing", []):
        transaction_id = entry.get("imported_txn_id")
        if entry.get("imported") and isinstance(transaction_id, int):
            referenced_entries.append((entry, transaction_id))

    if not referenced_entries:
        return

    transaction_ids = {transaction_id for _, transaction_id in referenced_entries}
    result = await session.execute(
        select(Transaction.id, Transaction.category, Transaction.note).where(
            Transaction.id.in_(transaction_ids)
        )
    )
    display_by_id = {
        row.id: (row.category, row.note, category_label(row.category))
        if row.category and row.category != "unknown"
        else (row.category, row.note, "Uncategorized")
        for row in result
    }

    for entry, transaction_id in referenced_entries:
        display = display_by_id.get(transaction_id)
        if display is None:
            continue
        category, note, label = display
        entry["transaction_category"] = category
        entry["transaction_category_label"] = label
        entry["transaction_note"] = note
