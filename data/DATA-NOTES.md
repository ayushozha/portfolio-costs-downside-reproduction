# Historical portfolio benchmark data

These are frozen historical benchmark matrices, acquired now for newly executed experiments. The acquisition timestamps are current; the data snapshot is the OLPS repository commit `9120783cd59a7966b0f78e2b5668030a4378b8af`, dated July 13, 2015. The pinned README describes OLPS 1.1.0, released June 20, 2015. The peer-reviewed toolbox article was published in April 2016.

| Dataset | CSV | Trading periods | Assets | Published calendar range |
|---|---|---:|---:|---|
| NYSE(O) | `nyse-o_price_relatives.csv` | 5,651 | 36 | July 3, 1962–December 31, 1984 |
| SP500 | `sp500_price_relatives.csv` | 1,276 | 25 | January 2, 1998–January 31, 2003 |
| DJIA | `djia_price_relatives.csv` | 507 | 30 | January 14, 2001–January 14, 2003 |

The published range is dataset-level documentation. No dates are embedded in the matrices, and no exact per-row calendar dates have been reconstructed. Report chronological **row intervals**, rather than invented dates for train/validation/test splits. The DJIA range begins on a non-trading calendar day; the provider's range is reproduced as a label, not interpreted as an exact first observation date.

Each row is the next daily price-relative vector, with `x[t,i] = price[t,i] / price[t-1,i]`; the portfolio return factor for an allocation chosen **before** observing row `t` is its dot product with row `t`. CSV headers `asset_01`, `asset_02`, etc. preserve source column order. A cash asset, if used, must be explicitly added by the experiment with a constant relative price of 1; it is not an original matrix column. No split, dividend, inflation, or current-price adjustments were added.

The untouched MAT files are retained under `raw/`. They contain a single `data` float64 matrix. Every raw file's Git blob SHA-1 matches the pinned repository tree. The CSV conversion preserves every source row and column, using 17 significant digits and a verified exact float64 round-trip. The data contains no missing, infinite, or nonpositive observations. This is a data-integrity check, not a scientific experiment.

`acquire_historical_olps.py` is a present-day retrieval and file-conversion utility. Its modern Python, NumPy and SciPy runtime is administrative infrastructure outside the experimental numerical stack. The experiments should consume the static CSV files through the separately audited scientific implementation. The raw sources, conversion and equality check are recorded truthfully, rather than described as software executed in 2017.

## Bias and scope limits

- These are **fixed constituent benchmark universes**, not reconstructions of the contemporaneously investable market. They permit controlled comparisons of methods on published data, but not unbiased claims of deployable investment profitability.
- The original SP500 documentation explicitly selected the 25 largest stocks by market capitalization **as of April 2003**, after the January 1998–January 2003 observation window. This retrospective constituent selection introduces survivorship/selection bias. The sample is not the entire S&P 500 index, nor an index total-return series.
- NYSE(O) contains 36 large-cap stocks collected by Hal Stern, according to the original provider's page. No historical universe membership or delisted-stock completeness is established.
- DJIA is a small 30-stock, 507-period benchmark. A held-out fraction has few observations, so tail-risk and Sharpe estimates may be unstable. It should not carry strong statistical claims on its own.
- Original provider pages list stock names, but mapping names to the pinned MAT columns has not been independently verified against the original ASCII download. The experiment CSVs therefore use anonymous column identifiers. No post-cutoff instruments are introduced.
- Corporate-action and dividend treatment are inherited from the published matrices. The available inspected documentation does not establish a complete adjustment audit; no stronger claim is made.
- Spread, market impact, slippage, order-size limits, taxes, and historical fee schedules are not embedded. Explicit transaction-cost assumptions are simulation settings, not recovered observations.

## Primary evidence and an identified source error

The pinned OLPS manual, **Table 2, page 7**, gives the ranges and dimensions above. Its model description on pages 2–4 specifies relative prices, nonnegative self-financing weights and the observation ordering. The JMLR 2016 article, section 2.1, page 2, describes an `n × m` matrix of price relatives and **Table 1** gives matching dimensions. However, that article prints the NYSE(O) end year as **1962**, inconsistent with 5,651 daily observations and the pinned manual. The manual and original provider both give **1984**. Preserve this discrepancy and use the corroborated 1984 range; do not silently copy the erroneous range.

- [Pinned OLPS repository](https://github.com/OLPS/OLPS/tree/9120783cd59a7966b0f78e2b5668030a4378b8af)
- [Pinned manual](https://raw.githubusercontent.com/OLPS/OLPS/9120783cd59a7966b0f78e2b5668030a4378b8af/Documentation/OLPS_manual.pdf)
- [JMLR published article](https://jmlr.org/papers/volume17/15-317/15-317.pdf)
- [SMU repository dataset record](https://ink.library.smu.edu.sg/researchdata/15/)
- [Original NYSE provider](https://csaws.cs.technion.ac.il/~rani/portfolios/NYSE_Dataset.htm)
- [Original SP500 provider](https://csaws.cs.technion.ac.il/~rani/portfolios/SP500_Dataset.htm)
- [Original DJIA provider](https://csaws.cs.technion.ac.il/~rani/portfolios/DJIA_Dataset.htm)

The canonical acquisition ledger is `../evidence/data_provenance/manifest.json`, with raw file hashes, URLs, immutable commit/tree evidence and actual retrieval times. Repository commit metadata can be altered by a repository maintainer; the matching publication and institutional data record independently corroborate that this benchmark family was public before the scientific cutoff. This package does not claim an Internet Archive timestamp for each data byte.
