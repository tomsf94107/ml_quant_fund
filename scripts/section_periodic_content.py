"""Draft of section_periodic_content() -- 10-K / 10-Q content scan.

DESIGN (pre-registered 2026-08-01, before any code):
  Two extraction targets, structurally DIFFERENT -- that difference is the
  whole difficulty and why one pattern list will not do:

    1. NON-OPERATING INCOME ITEMS (the $3.2B Anthropic-gain class). Live in the
       income statement or an "Other income (expense), net" footnote. Detected
       as DOLLAR MAGNITUDES near investment/equity-method keywords, returned
       with surrounding context.
    2. ACCOUNTING-POLICY CHANGES (15->25yr useful life). Footnote PROSE, often
       with NO dollar figure at all. Detected by keyword, sentence returned
       verbatim.

  DELIBERATELY NOT DOING full HTML financial-table reconstruction: brittle
  across filers, and both acceptance targets are reachable by targeted text
  extraction. If tables are ever genuinely needed, that is a separate build
  with its own justification.

  SCOPE GUARDS: 10-K/10-Q run 5-15MB. Cap to the 2 most recent filings, cap
  matches per category, truncate every context window, never dump the doc.
"""
import re

# --- Category 1: non-operating / one-time income items -----------------------
# Keywords that mark a paragraph as being ABOUT investment gains rather than
# operations. A dollar magnitude alone is meaningless in a 10-K (they are
# everywhere); it is the CO-OCCURRENCE that carries signal.
# REFINED after the MSFT FY26 10-K test (Aug 1 2026). Requiring only an
# investment CONTEXT word surfaced a balance-sheet holding ("equity investments
# ... measured at cost ... were $12.4 billion") as if it were a gain. An income
# item needs INCOME-STATEMENT language, so a gain/loss verb is now mandatory
# and the context words merely narrow it.
_GAIN_KEYS = [
    "net gains", "net losses", "gain of", "loss of", "gains of", "losses of",
    "recognized gain", "recognized loss", "unrealized gain", "unrealized loss",
    "realized gain", "realized loss", "impairment charge", "gain on sale",
    "remeasurement gain", "remeasurement loss", "included $",
]
_BALANCE_SHEET_TELLS = [
    "measured at cost", "carrying value", "as of june", "as of december",
    "as of march", "as of september", "total assets", "were $", "balance of",
]

_INCOME_ITEM_KEYS = [
    "equity method investment", "equity-method investment", "equity investment",
    "unrealized gain", "unrealized loss", "realized gain", "realized loss",
    "fair value adjustment", "remeasurement", "mark-to-market",
    "other income (expense)", "other income, net", "nonoperating",
    "non-operating", "gain on investment", "investment gains", "investment losses",
]

# --- Category 2: accounting-policy / estimate changes ------------------------
# A policy CHANGE needs change language. Matching "useful life" alone returned
# six boilerplate accounting-policy sentences from MSFT's 10-K and zero actual
# changes -- the keyword matched the topic, not the event.
# ROUND 3 (Aug 1 2026), after the MSFT 10-K test. Round 2 swapped one noise
# class for another: "reclassified" matched hedge-accounting TABLE FRAGMENTS and
# "restated" matched an EXHIBIT LIST entry ("Amended and Restated Directors'
# Indemnification Agreement" -- a document title). Both words are used in
# filings overwhelmingly for things that are not accounting changes; dropped.
# What remains requires a company ACTING on an estimate, in the past tense.
_CHANGE_VERBS = [
    "revised the estimated", "revised our estimate", "revised the useful",
    "changed the estimated", "extended the estimated", "shortened the estimated",
    "increased the estimated useful", "decreased the estimated useful",
    "change in accounting estimate", "change in accounting principle",
    "we completed an assessment", "resulted in a change in estimate",
    "accounted for prospectively as a change",
]
_YEAR_RANGE = re.compile(r"from\s+([\w-]+)\s+(?:to|through)\s+([\w-]+)\s+years?", re.I)

_POLICY_KEYS = [
    "useful life", "useful lives", "change in accounting estimate",
    "change in estimate", "revised the estimated", "revised our estimate",
    "extended the estimated", "depreciation period", "amortization period",
    "reclassified", "restated", "change in accounting principle",
    "newly adopted accounting", "adopted asu",
]

# $X.X billion / million / thousand, or bare $X,XXX
_MONEY = re.compile(
    r"\$\s?\d[\d,]*(?:\.\d+)?\s*(?:billion|million|bn|mm|thousand)?", re.I)

MIN_USD = 500_000_000  # only surface items >= $500M; smaller is noise in a 10-K


def _to_usd(tok: str):
    """'$3.2 billion' -> 3.2e9. Returns None if unparseable."""
    m = re.match(r"\$\s?([\d,]+(?:\.\d+)?)\s*(billion|million|bn|mm|thousand)?",
                 tok.strip(), re.I)
    if not m:
        return None
    try:
        v = float(m.group(1).replace(",", ""))
    except ValueError:
        return None
    unit = (m.group(2) or "").lower()
    if unit in ("billion", "bn"):
        v *= 1e9
    elif unit in ("million", "mm"):
        v *= 1e6
    elif unit == "thousand":
        v *= 1e3
    return v


def _sentences(text: str):
    """Split on sentence boundaries; filings use '. ' liberally."""
    return re.split(r"(?<=[.;])\s+(?=[A-Z(])", text)


def scan_periodic_text(cleaned: str, max_income=6, max_policy=6):
    """(income_items, policy_items) from HTML-stripped filing text.

    income_items: [(usd, sentence)] -- dollar magnitude >= MIN_USD appearing in
                  a sentence that also mentions an investment/non-operating key.
    policy_items: [(key, sentence)] -- accounting-estimate language.
    """
    low = cleaned.lower()
    income, policy = [], []
    seen_i, seen_p = set(), set()

    for sent in _sentences(cleaned):
        if len(sent) < 40 or len(sent) > 1200:
            continue
        # PROSE GATE (Aug 1 2026). Stripped HTML tables arrive as fragments
        # ("Amount reclassified from accumulated other comprehensive loss 50")
        # -- no verb, no terminal period, dense with entity refs and numbers.
        # Every category-2 false positive on MSFT was a table fragment or an
        # exhibit-list entry. Require sentence shape before matching anything.
        _st = sent.strip()
        if not _st.endswith((".", ";")):
            continue
        if _st.count("&#") > 2 or _st.count("&nbsp") > 2:
            continue
        _alpha = sum(c.isalpha() for c in _st)
        if _alpha < len(_st) * 0.55:   # tables are number/space heavy
            continue
        s_low = sent.lower()

        if len(income) < max_income:
            _ctx = next((k for k in _INCOME_ITEM_KEYS if k in s_low), None)
            _gain = next((k for k in _GAIN_KEYS if k in s_low), None)
            _bs = sum(1 for k in _BALANCE_SHEET_TELLS if k in s_low)
            # context AND income-statement language, and not obviously a
            # balance-sheet disclosure
            hit_key = _ctx if (_ctx and _gain and _bs < 2) else None
            if hit_key:
                best = None
                for tok in _MONEY.findall(sent):
                    v = _to_usd(tok)
                    if v and v >= MIN_USD and (best is None or v > best):
                        best = v
                if best:
                    sig = (round(best), s_low[:60])
                    if sig not in seen_i:
                        seen_i.add(sig)
                        income.append((best, " ".join(sent.split())))

        if len(policy) < max_policy:
            _pk = next((k for k in _POLICY_KEYS if k in s_low), None)
            _cv = next((v for v in _CHANGE_VERBS if v in s_low), None)
            _yr = _YEAR_RANGE.search(sent)
            # topic AND (change verb OR an explicit "from X to Y years")
            hit = _pk if (_pk and (_cv or _yr)) else None
            if hit:
                sig = (hit, s_low[:60])
                if sig not in seen_p:
                    seen_p.add(sig)
                    policy.append((hit, " ".join(sent.split())))

    income.sort(key=lambda x: -x[0])
    return income, policy


def fmt_usd(v):
    if v >= 1e9:
        return f"${v/1e9:.2f}B"
    if v >= 1e6:
        return f"${v/1e6:.0f}M"
    return f"${v:,.0f}"
