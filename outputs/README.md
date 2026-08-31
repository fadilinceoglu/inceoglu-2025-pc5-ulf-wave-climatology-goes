# Paper outputs

This directory contains only the named outputs of the published calculation. A
complete reproduction run also creates the checksum manifest shown below:

```text
figures/
  Fig01.jpg
  Fig02.jpg
  Fig03.jpg
  Fig04.jpg
tables/
  correlations_solar_cycles.csv
manifest.json
```

`python scripts/reproduce.py figures` regenerates the complete set from local
event catalogs, the hourly OMNI file, and the annual observation-count summary.
All figures are rendered in a temporary directory and replace the tracked files
only after the complete set succeeds. `python scripts/reproduce.py figure N`
regenerates a single numbered figure.

The tracked `Fig03.jpg` preserves the published figure's \(R^2\) annotation.
When Figure 3 is regenerated, the plotting code labels the calculated
coefficient as `Pearson's R`; the numeric calculation is unchanged.

In `manifest.json`, `inputs` records the public OMNI source URL and SHA-256 hash,
and the singular `parameter_summary` record identifies `configs/paper.toml` by
SHA-256. The `artifacts` records contain the generated figures and correlation
table with their sizes and SHA-256 hashes.

The pre-generated `Fig01.jpg` through `Fig04.jpg` files reproduce article
figures from F. Inceoglu and P. T. M. Loto’aniu, *Scientific Reports* **15**,
36661 (2025),
[https://doi.org/10.1038/s41598-025-20474-z](https://doi.org/10.1038/s41598-025-20474-z),
© The Author(s) 2025. They are distributed under
[CC BY-NC-ND 4.0](https://creativecommons.org/licenses/by-nc-nd/4.0/). The MIT
software license does not relicense the figures, generated research outputs, or
source observations; see [`DATA_NOTICE.md`](../DATA_NOTICE.md).
