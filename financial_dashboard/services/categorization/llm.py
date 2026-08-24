"""Provider-neutral LLM contract for categorization.

Holds the prompt, the result type, and the result parser. Both the Gemini
and the OpenAI-compatible provider import from here. Each provider module
only owns its transport: client construction, the call, and its own
structured-output flag.
"""

from collections.abc import Mapping, Sequence
from typing import Any, NamedTuple

from financial_dashboard.services.categorization.fewshot import FewShotExample
from financial_dashboard.services.categorization.normalize import (
    normalize_text,
    redact_names,
    redact_pii,
)

NEEDS_REVIEW = "needs_review"
LLM_TIMEOUT_MS = (
    30_000  # cap per-call latency so a slow provider can't stall the poll loop
)


class LlmResult(NamedTuple):
    slug: str
    confidence: float
    reason: str


# A short, evidenced note per bank on how to read its raw narration codes.
# Only add a bank here once a real sample shows the model needs the help --
# do not guess a format. Keyed by fields["bank"] (the same slug the parsers
# use), so it lines up with whichever bank produced the raw_description.
BANK_NARRATION_HINTS: dict[str, str] = {
    "indusind": (
        "IndusInd UPI descriptions look like "
        "'UPI/<ref>/DR|CR/<name>/<bank-code>/<vpa>'. DR/CR marks debit or "
        "credit here. It is not part of a name, and it does not mean doctor "
        "or healthcare."
    ),
    "idfc": (
        "IDFC UPI descriptions look like 'UPI/DR|CR/<ref>/<name>/<vpa>/"
        "<bank-code>/<note>'. DR/CR marks debit or credit here. It is not "
        "part of a name, and it does not mean doctor or healthcare."
    ),
    "sbi": (
        "SBI UPI descriptions look like "
        "'UPI/DR|CR/<ref>/<name>/<bank-code>/<vpa>/<remark>'. DR/CR marks "
        "debit or credit here. It is not part of a name, and it does not "
        "mean doctor or healthcare."
    ),
    "uboi": (
        "Union Bank UPI descriptions look like "
        "'UPIAB|UPIAR/<ref>/DR|CR/<name>/<bank-code>/<vpa>'. DR/CR marks "
        "debit or credit here. It is not part of a name, and it does not "
        "mean doctor or healthcare."
    ),
}


def _sanitize_counterparty(text: str | None, name_tokens: Sequence[str]) -> str:
    """Counterparty: strip numbers (PII) then mask configured name tokens."""
    return redact_names(redact_pii(text), name_tokens)


def _sanitize_description(text: str | None, name_tokens: Sequence[str]) -> str:
    """Description: strip numbers, mask names, then normalize whitespace/case."""
    return normalize_text(redact_names(redact_pii(text), name_tokens))


def build_prompt(
    *,
    fields: Mapping[str, str | None],
    examples: Sequence[FewShotExample],
    active_slugs: list[str],
    name_tokens: Sequence[str] = (),
) -> str:
    lines = [
        "You are a personal-finance transaction categorizer.",
        "Choose exactly ONE category slug from this list:",
        ", ".join(active_slugs),
        f'If none fit, return "{NEEDS_REVIEW}".',
        "Return JSON: {category, confidence (0..1), reason (one short sentence)}.",
        "",
        "IMPORTANT: 'direction: credit' = money RECEIVED — use an income category "
        "(refund, salary, interest, cashback_rewards, repayment, other_income); "
        "NEVER a spending category.",
        "'direction: debit' = money SPENT — use a spending category.",
        "A credit from an individual paying you back = repayment; "
        "a credit from a merchant = refund.",
        "Do NOT use self_transfer (handled separately). For money moved to/from another "
        "person, use 'repayment' for a credit or 'expense'/the specific spending category "
        "for a debit.",
        "",
    ]
    if examples:
        lines.append("Examples of previously categorized transactions:")
        for ex in examples:
            lines.append(
                f"- [{ex.direction}] {_sanitize_counterparty(ex.counterparty, name_tokens)} "
                f"| {_sanitize_description(ex.raw_description, name_tokens)} -> {ex.category}"
            )
        lines.append("")
    lines.append("Transaction to categorize:")
    lines.append(f"direction: {fields.get('direction')}")
    lines.append(f"amount: {fields.get('amount')} {fields.get('currency')}")
    lines.append(f"channel: {fields.get('channel')}")
    if hint := BANK_NARRATION_HINTS.get(fields.get("bank") or ""):
        lines.append(f"format note: {hint}")
    lines.append(
        f"counterparty: {_sanitize_counterparty(fields.get('counterparty'), name_tokens)}"
    )
    lines.append(
        f"description: {_sanitize_description(fields.get('raw_description'), name_tokens)}"
    )
    return "\n".join(lines)


def parse_result(data: Mapping[str, Any], active_slugs: list[str]) -> LlmResult:
    slug = str(data.get("category", "")).strip()
    try:
        conf = float(data.get("confidence", 0.0))
    except ValueError, TypeError:
        conf = 0.0
    conf = max(0.0, min(1.0, conf))
    reason = str(data.get("reason", ""))[:300]
    if slug != NEEDS_REVIEW and slug not in active_slugs:
        return LlmResult(NEEDS_REVIEW, conf, reason or "model returned unknown slug")
    return LlmResult(slug, conf, reason)
