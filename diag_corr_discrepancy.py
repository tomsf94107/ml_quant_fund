import sys; sys.path.insert(0,'.')
from pathlib import Path
import numpy as np, pandas as pd

# The combiner uses qv_books.csv directly. Let me read THOSE actual book returns
# and recompute the gp-vs-mom correlation the combiner saw.
books = pd.read_csv('data/qv_books.csv', parse_dates=['date']).set_index('date')
print('qv_books columns:', list(books.columns))
print('qv_books date range:', books.index.min().date(), '->', books.index.max().date(), f'({len(books)} months)')
print()
# the ACTUAL correlation the combiner operates on:
print('=== correlation matrix from qv_books.csv (what combiner uses) ===')
print(books[['mom','gp','op','ep']].corr().round(3))
print()
# is it possibly that the high corr is driven by a COMMON market component?
# check: are these LONG-ONLY books (corr w/ spy high) or long-short (market-neutral)?
if 'spy' in books.columns:
    for c in ['mom','gp','op','ep']:
        print(f'  {c} corr vs spy: {books[c].corr(books["spy"]):+.3f}')
    print('  (if all high +corr to spy -> these are LONG-ONLY books, co-move via market beta)')
