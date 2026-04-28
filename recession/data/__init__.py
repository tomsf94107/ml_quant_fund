"""
FRED/ALFRED data ingestion for the recession model.

This subpackage handles all external data acquisition:
- FRED API for non-revisable market data (yields, prices, spreads)
- ALFRED API for revisable series with full vintage history
- Local caching to respect FRED's 120 req/min rate limit
- Frequency conversion (daily/weekly/quarterly → monthly)
- Vintage stamping per spec §5

All output writes to recession.db (NEVER accuracy.db).
"""
