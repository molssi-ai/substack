# Articles, whitepapers and technical reports

This repository collects the metadata and artifacts pertinent to MolSSI-AI's
articles, whitepapers and technical reports. Each publication has its own folder
with everything needed to reproduce its results: code, job scripts, environment
files and the raw data behind its figures and tables. The README inside each
folder offers a step-by-step guide to reproducing the publication's results.

## Publications

| date | type | title | folder |
|---|---|---|---|
| 2026-11-25 | article | *Accelerated Machine Learning Interatomic Potentials for Materials Discovery* | [xnn-vs-alchemy-11-25-2026](xnn-vs-alchemy-11-25-2026/) |

## Adding a publication

- Create a folder named `<topic>-<MM-DD-YYYY>`, with the publication date.
- Make it self-contained: its own README, install script and pinned versions, and no absolute paths.
- Do not commit third-party code. Download it in the install script instead.
- Add a row to the table above.
