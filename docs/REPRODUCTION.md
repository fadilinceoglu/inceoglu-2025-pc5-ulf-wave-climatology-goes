# Reproduction commands

All commands below are run from the repository root. The common entry point is
`scripts/reproduce.py`; the smaller scripts call the same stage implementations.

## Install

The recorded numerical environment used CPython 3.9.6 and the pinned NumPy,
SciPy, pandas, Matplotlib, and joblib versions in `requirements.lock`.
Acquisition-only packages are pinned to maintained releases compatible with
that numerical environment and current supported platforms:

```bash
python3.9 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.lock
python -m pip install -e . --no-deps
```

For checkpoint-based catalog and figure work, the core installation is sufficient:

```bash
python -m pip install -e .
```

For an upstream acquisition run without using `requirements.lock`:

```bash
python -m pip install -e ".[full]"
```

The full dependency group pins SpacePy 0.7.0. Its CTrans path uses the bundled
IGRF13 coefficients in isolated process-local state; user-level SpacePy data
updates are not inputs to this reproduction. If an application has already
imported SpacePy, start the reproduction in a fresh Python process. CTrans
applies IGRF13's endpoint behavior to observations after the model's 2025-01-01
validity endpoint.

Install the test and lint tools with:

```bash
python -m pip install -e ".[dev]"
```

## End-to-end from a clean clone

```bash
python scripts/reproduce.py all --full
```

`--full` explicitly permits the source-data stages. The command downloads hourly
OMNI data, retrieves and prepares the GOES observations, runs Pc5 detection,
constructs the up-to-three-peak event catalogs, and writes all paper outputs. Existing
complete stages are resumed.

Acquisition and detection over the paper interval require substantial compute,
memory, working storage, and network transfer. Run them in a persistent
environment with adequate disk space. The default worker count is bounded at
four; reduce it when memory is the limiting resource:

```bash
python scripts/reproduce.py all --full --workers 2
```

To restart every stage rather than resume it:

```bash
python scripts/reproduce.py all --full --force
```

The combination is intentionally explicit because it replaces expensive local
checkpoints. A plain `all --force` is rejected.

## Inspect local state

```bash
python scripts/reproduce.py status
```

The command reports whether the OMNI file, prepared GOES data, detections,
catalogs, and figures are present. Fields ending in `_artifacts` mean only that
the stage's named files exist. `prepared` additionally requires the annual count
table and a valid `acquisition_complete.json` marker whose hashes match every
prepared output. `detected` verifies a local
`detection_complete.json` marker against the prepared input, detection parameters,
and detection outputs. `cataloged` means the catalog files are usable and, when a
`catalog_complete.json` marker is present, that its output identities match.
If an acquisition completion marker exists but its hashes do not match, normal
resume stops instead of accepting the changed checkpoint; restarting that stage
requires an explicit `--force`.
Supply non-default locations when needed:

```bash
python scripts/reproduce.py status \
  --checkpoint-dir /path/to/checkpoints \
  --omni-file /path/to/omni2_all_years.dat
```

`detection_incomplete.json` and `catalog_incomplete.json` are written when their
stages begin and removed only after successful completion. If either marker
remains, the corresponding stage requires resumption and its outputs are not used
downstream. Optional unmarked checkpoint sets can still be supplied locally;
pickle-based files must come from a source you trust. An unmarked detection set
is accepted only with the repository's default detection parameters. Supplying a
different seed or Monte Carlo sample count requires a matching completed run.

## Optional checkpoint-based run

```bash
python scripts/reproduce.py
```

With no stage argument, `all` is selected. The command resumes from the deepest
complete boundary: existing event catalogs go directly to plotting, while
detection checkpoints first build missing catalogs. The OMNI file is downloaded
when absent. It refuses to launch the resource-intensive acquisition and detection
stages when the required checkpoints are absent; use `all --full` for a clean
clone.

When the complete checkpoint set is stored elsewhere:

```bash
python scripts/reproduce.py all \
  --checkpoint-dir /path/to/checkpoints \
  --omni-file /path/to/omni2_all_years.dat
```

If only the three event catalogs are available, either normal `all` or the
figure-only stage can be used:

```bash
python scripts/reproduce.py figures \
  --checkpoint-dir /path/to/catalogs \
  --omni-file /path/to/omni2_all_years.dat
```

Figure 4 additionally requires annual observation counts. The preferred file is
`observation_counts_by_year.csv` in the checkpoint directory, with columns
`year,observation_count`. If that file is absent, the code can derive it from the
`df_08.pkl` through `df_18.pkl` daily checkpoints by counting every stored daily
table.

### Figure inputs

The optional component catalog checkpoint files are:

- `radial_powers_freq_mlt_date.pkl`;
- `az_powers_freq_mlt_date.pkl`; and
- `par_powers_freq_mlt_date.pkl`.

These files are not committed or downloaded by the repository. Figures 3 and 4
also require OMNI, and Figure 4 requires annual observation
counts. Optional locally supplied event-catalog checkpoints are the practical
starting point for figure-only runs, and figure stages use their supplied rows
without an additional date filter. A source-data run associates each NOAA product
with prepared data by satellite and UTC date. The paper's Monte Carlo seed was
not recorded; the repository default is 2025, so new detections near the 95%
threshold may differ from a locally supplied catalog.

## Individual stages

```bash
# Download OMNI and acquire/prepare GOES data.
python scripts/reproduce.py acquire

# Run CLEAN detection from prepared GOES checkpoints.
python scripts/reproduce.py detect --workers 4

# Retain up to the three highest-amplitude significant peaks per window.
python scripts/reproduce.py catalog

# Regenerate all paper figures and the correlation table.
python scripts/reproduce.py figures

# Regenerate one paper figure.
python scripts/reproduce.py figure 1
python scripts/reproduce.py figure 2
python scripts/reproduce.py figure 3
python scripts/reproduce.py figure 4
```

The equivalent role-specific wrappers are:

```bash
python scripts/prepare_goes_data.py
python scripts/detect_pc5_events.py --workers 4
python scripts/build_event_catalog.py
python scripts/plot_figure_01.py
python scripts/plot_figure_02.py
python scripts/plot_figure_03.py
python scripts/plot_figure_04.py
```

Use `--force` with an individual upstream stage to replace that stage's existing
outputs. Figure commands overwrite their named output by design. A canonical
single-figure command removes the complete-set checksum manifest before the
overwrite; run `python scripts/reproduce.py figures` to recreate the manifest
for a complete output set.

## Shared options

| Option | Meaning |
| --- | --- |
| `--checkpoint-dir PATH` | Prepared, detection, catalog, and observation-count files. |
| `--omni-file PATH` | Hourly `omni2_all_years.dat`. |
| `--figures-dir PATH` | Destination for `Fig01.jpg` through `Fig04.jpg`. |
| `--tables-dir PATH` | Destination for `correlations_solar_cycles.csv`. |
| `--workers N` | Detection worker processes. |
| `--monte-carlo-samples N` | White-noise realizations per CLEAN significance test; paper default 5,000. |
| `--random-seed INTEGER` | Detection seed; default 2025. |
| `--force` | Replace outputs of the selected stage. |
| `--full` | Permit acquisition and detection in an `all` run. |

## Stage contracts

1. `acquire` writes `processed_data.pkl`, one `df_XX.pkl` file per GOES
   satellite, `observation_counts_by_year.csv`, and—only after complete source
   discovery with no unresolved products—`acquisition_complete.json`, which
   records hashes for the prepared outputs.
2. `detect` writes the three `Frequency_Power_*_new_1h.pkl` component files and
   `detection_complete.json` after successful completion. Its resumable daily
   work files are stored under `.cache/detection/`; a remaining
   `detection_incomplete.json` means resumption is required.
3. `catalog` writes `radial_powers_freq_mlt_date.pkl`,
   `az_powers_freq_mlt_date.pkl`, `par_powers_freq_mlt_date.pkl`, and
   `catalog_complete.json`. A remaining `catalog_incomplete.json` means
   resumption is required.
4. `figures` writes `outputs/figures/Fig01.jpg` through `Fig04.jpg`,
   `outputs/tables/correlations_solar_cycles.csv`, and an output checksum
   manifest. All figures are rendered in a staging directory and promoted only
   after the complete set succeeds.

The three completion markers are local operational metadata under the checkpoint
directory and are ignored by Git.

Joblib checkpoints must come from a source you trust because loading pickle-based
files can execute Python code.

## Randomness

The paper calculation used 5,000 Monte Carlo white-noise simulations without a
recorded seed. Omitting `--random-seed` uses the repository seed 2025. Supplying
another integer selects a different repeatable realization; events near the 95%
significance boundary can differ from a locally supplied catalog. Catalog and
figure stages do not add Monte Carlo randomness.

## Tests

The tests use small fixtures and do not run the 30-year acquisition or detection:

```bash
python -m pytest
```
