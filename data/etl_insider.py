# data/etl_insider.py

import os
import re
import requests
import pandas as pd
import feedparser
from datetime import datetime
from typing import List

# reuse your SEC headers (must include User-Agent with your contact info)
from sentiment_utils import SEC_EDGAR_HEADERS


def fetch_insider_trades(ticker: str, count: int = 20) -> pd.DataFrame:
    """
    Pull the most recent `count` Form 4 filings for `ticker` from SEC EDGAR.
    Returns a DataFrame with columns:
      - ds            : filing datetime (date)
      - net_shares    : positive for net buys, negative for net sells
      - num_buy_tx    : count of buy transactions
      - num_sell_tx   : count of sell transactions
    """
    # Build the Atom feed URL for Form 4 filings
    url = (
        "https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&"
        f"CIK={ticker}&type=4&owner=include&count={count}&output=atom"
    )

    resp = requests.get(url, headers=SEC_EDGAR_HEADERS, timeout=8)
    resp.raise_for_status()

    feed = feedparser.parse(resp.text)
    records: List[dict] = []

    for entry in feed.entries:
        # parse the filing date
        try:
            ds = datetime.strptime(entry.published, "%Y-%m-%dT%H:%M:%SZ")
        except Exception:
            continue

        # fetch the raw XML version of the filing
        filing_url = entry.link.replace("-index.htm", ".xml")
        xml = requests.get(filing_url, headers=SEC_EDGAR_HEADERS, timeout=8).text

        # sum up all Acquired vs Disposed shares
        buy_shares  = sum(
            int(x.replace(",", ""))
            for x in re.findall(
                r"<transactionAcquiredDisposedCode>\s*Acquired\s*</transactionAcquiredDisposedCode>.*?<transactionShares>\s*([\d,]+)\s*</transactionShares>",
                xml,
                flags=re.S,
            )
        )
        sell_shares = sum(
            int(x.replace(",", ""))
            for x in re.findall(
                r"<transactionAcquiredDisposedCode>\s*Disposed\s*</transactionAcquiredDisposedCode>.*?<transactionShares>\s*([\d,]+)\s*</transactionShares>",
                xml,
                flags=re.S,
            )
        )

        # count number of buy vs sell transactions
        buy_count  = len(re.findall(r"<transactionAcquiredDisposedCode>\s*Acquired\s*</transactionAcquiredDisposedCode>", xml))
        sell_count = len(re.findall(r"<transactionAcquiredDisposedCode>\s*Disposed\s*</transactionAcquiredDisposedCode>", xml))

        records.append({
            "ds": ds.date(),
            "net_shares": buy_shares - sell_shares,
            "num_buy_tx": buy_count,
            "num_sell_tx": sell_count,
        })

    # build and return a date-sorted DataFrame
    df = pd.DataFrame(records)
    if df.empty:
        return df

    return df.sort_values("ds").reset_index(drop=True)
