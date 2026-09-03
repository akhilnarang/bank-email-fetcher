from financial_dashboard.services import settings as settings_mod
from financial_dashboard.services.categorization.rules import (
    default_rule_config,
    load_rule_config,
    match_rules,
)

# Merchant rules are DATA (DB-backed), so these tests use synthetic stand-in
# patterns — one per category the behavioral cases need — rather than real
# merchant names. The engine logic under test is independent of which merchants
# happen to be seeded.
CFG = default_rule_config()._replace(
    self_name_tokens=("alex", "doe"),
    merchant_rules=(
        ("billco", "bill_payment"),
        ("paybills", "bill_payment"),
        ("cardpay", "credit_card_payment"),
        ("paid via cardpay", "credit_card_payment"),
        ("payroll inc", "salary"),
        ("investco", "investment"),
    ),
)


def _f(cp=None, raw=None, channel=None, direction="debit", bank=None):
    return {
        "counterparty": cp,
        "raw_description": raw,
        "channel": channel,
        "direction": direction,
        "bank": bank,
    }


def test_self_transfer_by_name():
    r = match_rules(_f(cp="ALE X QUINN DOE", direction="credit"), CFG)
    assert r is not None and r.slug == "self_transfer"


def test_card_payment_marker():
    r = match_rules(_f(cp="CARDPAY SETTLEMENT"), CFG)
    assert r is not None and r.slug == "credit_card_payment"


def test_cc_spend_is_not_card_payment():
    # "spent on ... Credit Card" is a SPEND, never a card-payment. The cc-spend
    # guard blocks only credit_card_payment rules, so a bill_payment merchant
    # still fires.
    raw = "We inform you that INR 195.00 was spent on your Credit Card"
    r = match_rules(_f(cp="BILLCO LIMITED", raw=raw), CFG)
    assert r is not None and r.slug == "bill_payment"


def test_bill_payment_markers():
    for cp in ("BILLCO LIMITED", "PAYBILLS INC"):
        r = match_rules(_f(cp=cp), CFG)
        assert r is not None and r.slug == "bill_payment", cp


def test_interest_channel():
    r = match_rules(_f(channel="interest", direction="credit"), CFG)
    assert r is not None and r.slug == "interest"


def test_fd_booking_is_investment():
    # The parser labels an FD booking "<Bank> FD" — the row's own bank plus " FD".
    r = match_rules(_f(cp="IDFC FD", direction="debit", bank="idfc"), CFG)
    assert r is not None and r.slug == "investment"


def test_fd_maturity_is_redemption():
    r = match_rules(_f(cp="Slice FD", direction="credit", bank="slice"), CFG)
    assert r is not None and r.slug == "investment_redemption"


def test_fd_maturity_beats_interest_channel():
    # A maturity row carries the "interest" channel, but the whole credit is a
    # redemption. The FD rule must win over the interest shortcut.
    r = match_rules(
        _f(cp="IDFC FD", channel="interest", direction="credit", bank="idfc"), CFG
    )
    assert r is not None and r.slug == "investment_redemption"


def test_fd_label_is_derived_from_the_row_bank():
    # The label must be the row's OWN bank plus " FD". An unrelated beneficiary
    # ending in " FD", or a label naming a different bank, must NOT match.
    assert (
        match_rules(_f(cp="ICICI FD", direction="debit", bank="icici"), CFG).slug
        == "investment"
    )
    assert match_rules(_f(cp="ACME FD", direction="debit", bank="idfc"), CFG) is None
    assert match_rules(_f(cp="IDFC FD", direction="debit", bank="slice"), CFG) is None
    assert match_rules(_f(cp="ACMEFD", direction="debit", bank="idfc"), CFG) is None


def test_email_type_cc_payment_received_with_blank_fields():
    # CC payment/refund SMS alerts have no counterparty/narration; key off email_type.
    fields = {"counterparty": None, "raw_description": None, "direction": "credit"}
    fields["email_type"] = "bank_cc_payment_received_alert"
    assert match_rules(fields, CFG).slug == "credit_card_payment"
    fields["email_type"] = "bank_cc_refund_alert"
    assert match_rules(fields, CFG).slug == "refund"


def test_email_type_cc_payment_forms_are_card_payments():
    # All card-payoff alert forms map to credit_card_payment (neutral, either
    # direction): payment_alert (no "received"), bill_paid, smartpay/BBPS. The
    # bank prefix is irrelevant — only the substring matters.
    def f(et, direction="credit"):
        return {
            "counterparty": None,
            "raw_description": None,
            "email_type": et,
            "direction": direction,
        }

    for et in (
        "banka_cc_payment_alert",
        "bankb_cc_payment_received_alert",
        "bankc_cc_bill_paid_alert",
    ):
        assert match_rules(f(et), CFG).slug == "credit_card_payment", et
    # debit-side payoffs (paying the card from the bank account) still map
    assert (
        match_rules(f("bankd_cc_bill_paid", "debit"), CFG).slug == "credit_card_payment"
    )
    assert match_rules(f("banke_cc_smartpay_bbps_alert", "debit"), CFG).slug == (
        "credit_card_payment"
    )
    # a card SPEND alert must NOT be treated as a payment
    assert match_rules(f("bankf_cc_transaction_alert", "debit"), CFG) is None


def test_salary_marker():
    # marker fires even with trailing narration noise around it
    r = match_rules(_f(cp="PAYROLL INC DBA WAGES 430", direction="credit"), CFG)
    assert r is not None and r.slug == "salary"


def test_investment_marker():
    r = match_rules(_f(cp="INVESTCO BROKING LIMITED"), CFG)
    assert r is not None and r.slug == "investment"


def test_no_match_returns_none():
    assert match_rules(_f(cp="UNLISTED MERCHANT XYZ"), CFG) is None


def test_investment_credit_is_redemption():
    cfg = CFG._replace(merchant_rules=(("investco", "investment"),))
    assert match_rules(_f(cp="INVESTCO", direction="debit"), cfg).slug == "investment"
    assert (
        match_rules(_f(cp="INVESTCO", direction="credit"), cfg).slug
        == "investment_redemption"
    )


def test_merchant_rule_skips_refund_credit():
    # A spend-merchant rule (foodco->dining) must NOT fire on a credit (a refund
    # at that merchant); it falls through so the LLM/fallback can call it a refund.
    cfg = CFG._replace(merchant_rules=(("foodco", "dining"),))
    assert match_rules(_f(cp="WWW FOODCO", direction="debit"), cfg).slug == "dining"
    assert match_rules(_f(cp="WWW FOODCO", direction="credit"), cfg) is None


def test_merchant_type_beats_self_by_counterparty():
    # Precedence: a specific narration signal beats a weak "Self"/own-name
    # counterparty label (banks tag own-account credits as "Self", but the
    # narration — e.g. CASHBACK — is the truth).
    cfg = CFG._replace(
        self_name_tokens=("self",),
        merchant_rules=(("cashback", "cashback_rewards"),),
    )
    r = match_rules(_f(cp="Self", raw="CASHBACK FOR BILLPAY", direction="credit"), cfg)
    assert r is not None and r.slug == "cashback_rewards"


def test_self_transfer_when_no_merchant_match():
    # With no merchant/type signal, a self-identifier counterparty → self_transfer.
    r = match_rules(_f(cp="ALEX QUINN DOE", direction="credit"), CFG)
    assert r is not None and r.slug == "self_transfer"


# --- a card-payment merchant matches across fields and through normalization ---


def test_card_payment_via_description():
    r = match_rules(_f(cp="UPI", raw="Paid Via CardPay", direction="debit"), CFG)
    assert r is not None and r.slug == "credit_card_payment"


def test_card_payment_counterparty():
    r = match_rules(_f(cp="cardpaytech", direction="debit"), CFG)
    assert r is not None and r.slug == "credit_card_payment"


def test_card_payment_upi_handle_normalized():
    # A dotted/suffixed UPI handle normalizes ("cardpay.loans@upi" -> "cardpay
    # loans upi") so the "cardpay" substring still matches.
    r = match_rules(_f(cp="cardpay.loans@upi", direction="debit"), CFG)
    assert r is not None and r.slug == "credit_card_payment"


# --- self_transfer via self_identifiers setting ---


def test_load_rule_config_uses_self_identifiers(monkeypatch):
    monkeypatch.setitem(
        settings_mod._cache, "categorization.self_identifiers", "alex, lee"
    )
    cfg = load_rule_config()
    # 'alex' and 'lee' are >=3 chars and should be in self_name_tokens
    assert "alex" in cfg.self_name_tokens
    assert "lee" in cfg.self_name_tokens
    # base 'self' token is still present
    assert "self" in cfg.self_name_tokens


def test_load_rule_config_self_identifier_triggers_rule(monkeypatch):
    monkeypatch.setitem(settings_mod._cache, "categorization.self_identifiers", "alex")
    cfg = load_rule_config()
    r = match_rules(_f(cp="ALEX SAVINGS ACCOUNT", direction="credit"), cfg)
    assert r is not None and r.slug == "self_transfer"


def test_dividend_credit_is_other_income():
    for raw in (
        "ACME HOTELS COMPANY LTD FINAL DIV 25 26",
        "SOMEFUND FINAN DIV 2025 26",
        "INTERIM DIVIDEND ACME LTD",
    ):
        r = match_rules(_f(raw=raw, direction="credit", channel="bank_statement"), CFG)
        assert r is not None and r.slug == "other_income", raw


def test_dividend_marker_is_whole_token_and_credit_only():
    # "div" inside a word is not a dividend; a debit with the token is not one.
    assert match_rules(_f(raw="INDIVIDUAL STORE", direction="credit"), CFG) is None
    assert match_rules(_f(raw="ACME FINAL DIV 25 26", direction="debit"), CFG) is None


def test_dividend_marker_does_not_fire_on_a_card():
    fields = _f(raw="ACME FINAL DIV 25 26", direction="credit")
    fields["account_type"] = "credit_card"
    r = match_rules(fields, CFG)
    assert r is not None and r.slug == "credit_card_payment"


def test_ach_credit_is_a_dividend():
    # Every ACH credit on this ledger is a share dividend. A listed bank as the
    # payer is the company paying the dividend, not a bank moving money.
    for cp, channel in (
        ("ACH C- SOME BANK LIMITED-9573935", "ach_credit"),
        ("ACH C- SOME BANK LIMITED-9573935", None),
        ("ACH C- ACME MINING LTD-979620", "ach_credit"),
    ):
        r = match_rules(_f(cp=cp, direction="credit", channel=channel), CFG)
        assert r is not None and r.slug == "other_income", (cp, channel)


def test_ach_credit_rule_is_credit_only_and_yields_to_merchant_rules():
    assert match_rules(_f(cp="ACH D- ACME LTD", channel="ach_debit"), CFG) is None
    r = match_rules(
        _f(cp="ACH C- PAYROLL INC-123", direction="credit", channel="ach_credit"), CFG
    )
    assert r is not None and r.slug == "salary"
