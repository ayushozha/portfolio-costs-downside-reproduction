# Transaction Costs and Downside Risk in Deep Reinforcement Learning for Portfolio Allocation

Code and recorded benchmark results for a comparison of direct deep portfolio policy objectives with exact proportional transaction-cost accounting.

**Author:** Ayush Ojha · Independent Researcher, San Francisco, CA, USA · [ORCID 0009-0001-9404-4930](https://orcid.org/0009-0001-9404-4930)

**Topics:** deep reinforcement learning · portfolio allocation · transaction costs · downside risk · reproducibility

| Status | Evidence |
| --- | --- |
| Recorded experiments | 75 model fits and 336 fee-specific test trajectories |
| Actual execution date | October 4, 2026 |
| Scientific information boundary | Public before July 25, 2017 |
| Numerical runtime | Python 3.6.2 and NumPy 1.13.1 |
| Manuscript status | Research manuscript under preparation; journal submission and peer review are separate |

## Download and inspect

Download `reproduction-package.zip` from the [v1.0.0 release](https://github.com/ayushozha/portfolio-costs-downside-reproduction/releases/tag/v1.0.0). The full archive includes frozen code, protocol, normalization, selected parameters, every validation history and every fee-specific test trajectory. `PACKAGE-MANIFEST.json` records file hashes. `supplementary/PACKAGE-INSTRUCTIONS.md` supplies retrieval and reproduction commands.

This repository also exposes the small source and aggregate-result files for browsing:

- `experiment/`: executed engine, fixed protocol, saved-trajectory analysis and audit code.
- `tools/fetch_reproduction_inputs.py`: administrative retrieval and hash verification of pinned inputs.
- `results/`: primary summaries, additional metrics, checkpoint selections and descriptive initialization comparisons.
- `data/DATA-NOTES.md`: retrospective universe and data limitations.
- `CITATION.cff`: author, ORCID and artifact citation metadata.

## Reproduce a separate run

Extract the full release archive into a fresh directory. Retrieve inputs with an available administrative Python environment containing NumPy and SciPy:

```powershell
py -3.12 tools/fetch_reproduction_inputs.py
```

The utility retrieves three matrices from [OLPS commit 9120783cd59a7966b0f78e2b5668030a4378b8af](https://github.com/OLPS/OLPS/tree/9120783cd59a7966b0f78e2b5668030a4378b8af), verifies original hashes and verifies lossless CSV conversions. Input matrices, third-party papers and runtime binaries are not redistributed.

Provide genuine Python 3.6.2 and NumPy 1.13.1 releases at `runtime/python362/python.exe`; the archive's software-provenance manifest supplies exact sources and checksums. From the extracted archive root:

```powershell
.\runtime\python362\python.exe experiment/portfolio_engine_frozen.py --protocol experiment/protocol.json --output evidence/runs/independent-run
.\runtime\python362\python.exe experiment/analyze_results.py --protocol experiment/protocol.json --results evidence/runs/independent-run --output evidence/analysis/independent-run
```

Use fresh output folders. Modern retrieval, file parsing and document formatting are administrative operations. Scientific training and numerical analysis use the historical versions above. See `experiment/METHOD.md` for accounting, optimizer conventions and evaluation conditions.

## Findings and limits

Cost-aware objectives reduce trading and improve net results relative to the matched return-only setting on these recorded paths. Equal-weight rebalancing exceeds the cost-aware median terminal wealth on all three datasets. Some validation selections retain the random initialization. Downside penalties have mixed wealth effects; a strong penalty increases cash exposure and turnover. These results establish no general trading profitability or unique causal mechanism.

## Verification and privacy

The retained path audit independently recomputes accounting and summary metrics. The package builder checks every archive entry against its manifest. Two JSON projections replace private absolute protocol locations with package-relative paths; the manifest preserves original and projected hashes and identifies the fields changed. Actual timestamps and scientific results remain intact.

## Rights

Copyright 2026 Ayush Ojha. All rights reserved. No open-source license is granted; see `LICENSE-NOTICE.md`. Third-party input rights remain with the original providers. This release contains this study's code and recorded outputs, while third-party inputs are fetched separately.
