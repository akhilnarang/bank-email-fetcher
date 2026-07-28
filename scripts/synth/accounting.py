"""Shared posting-sign rules for synthetic ledger corpora."""

from scripts.synth.models import SynthTransaction

_SOURCE_PREFIXES = ("Assets:Bank:", "Liabilities:Card:")


def _is_source_account(account: str) -> bool:
    return account.startswith(_SOURCE_PREFIXES)


def first_posting_sign(transaction: SynthTransaction) -> int:
    """Return the sign for a transaction's ``ledger_account`` posting.

    Scenario rows can put either the transaction-bearing source account or the
    category/destination account first. Use transaction identity for the
    two-source transfer cases, where account roots alone are ambiguous.
    """
    direction = transaction.direction
    if direction not in {"credit", "debit"}:
        raise ValueError(f"unsupported transaction direction: {direction!r}")
    source_sign = 1 if direction == "credit" else -1

    first = transaction.ledger_account
    counterpart = transaction.ledger_counterpart
    if first is None or counterpart is None:
        raise ValueError("synthetic transaction has no ledger posting accounts")
    first_is_source = _is_source_account(first)
    counterpart_is_source = _is_source_account(counterpart)

    if first_is_source and counterpart_is_source:
        if transaction.category == "self_transfer":
            # Debit rows name the destination first; credit rows name their
            # transaction-bearing destination first.
            return -source_sign if direction == "debit" else source_sign
        if transaction.category == "credit_card_payment":
            # Bank-side rows carry an account mask and name the liability
            # first. Card-side rows have no bank mask and name the card first.
            return -source_sign if transaction.account_mask else source_sign
        raise ValueError("ambiguous two-source synthetic transaction")

    return source_sign if first_is_source else -source_sign
