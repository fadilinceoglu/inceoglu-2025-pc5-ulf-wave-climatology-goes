# Occurrence characteristics and amplitude-frequency relationship of the Pc5 ULF waves from 3 decades of GOES data

This repository contains the calculation and paper outputs for Inceoglu and
Loto’aniu (2025), published in *Scientific Reports*. The analysis identifies Pc5
ultra-low-frequency waves in magnetic-field observations from GOES-8 through
GOES-18, constructs the three-component event catalogs, and reproduces paper
Figures 1–4 and the solar-cycle correlation table.

Paper: [https://doi.org/10.1038/s41598-025-20474-z](https://doi.org/10.1038/s41598-025-20474-z)

## Results in scope

- `outputs/figures/Fig01.jpg`: event amplitude and frequency distributions by
  magnetic local time (MLT).
- `outputs/figures/Fig02.jpg`: amplitude-frequency fits in dawn, day, dusk, and
  night sectors.
- `outputs/figures/Fig03.jpg`: amplitude-frequency fits under strong, moderate,
  and weak solar-wind conditions.
- `outputs/figures/Fig04.jpg`: annual occurrence rates and solar-wind parameters,
  before and after the 1/5 yr⁻¹ high-pass filter.
- `outputs/tables/correlations_solar_cycles.csv`: Pearson correlations for solar
  cycles 23 and 24.

The GOES acquisition interval is 1995-07-01 through 2025-05-10, inclusive. Each
source product is associated with its satellite and UTC date before preparation.
The event catalog retains up to the three highest-amplitude significant peaks in
each 1-hour window with 30-minute overlap. The required hourly OMNI analysis grid
begins on 1995-01-01 so Figure 4 uses a complete first calendar year and runs
through the inclusive study end; later aggregate rows are ignored.

## Environment

The calculation environment used for the study was CPython 3.9.6. Create a
compatible environment with the pinned direct dependency versions:

```bash
python3.9 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.lock
python -m pip install -e . --no-deps
```

The lighter checkpoint-to-figure installation is `python -m pip install -e .`.
The complete upstream path additionally requires the dependencies in the `full`
optional group, all of which are included in `requirements.lock`.
The core numerical-analysis package versions preserve the study environment.
The acquisition-only HTTP client and SpacePy coordinate helper are pinned to
maintained Python-3.9-compatible, wheel-supported releases so a clean install
also works on Apple Silicon.

Geographic-to-solar-magnetic conversion uses SpacePy 0.7.0 with
`use_irbem=False` (CTrans) and the IGRF13 coefficient table bundled with that
release. The converter initializes SpacePy in isolated process-local state so a
user-level SpacePy data update cannot change the calculation. Acquisition must
initialize SpacePy itself and stops with guidance if it was imported earlier.
For observations after IGRF13's
2025-01-01 validity endpoint, CTrans applies the model's endpoint behavior.

Install the test and lint tools, including the Python 3.9 TOML parser, with:

```bash
python -m pip install -e ".[dev]"
```

## Reproduce

Raw observations and checkpoint files are not committed. From a clean clone,
run the complete source-data calculation explicitly:

```bash
python scripts/reproduce.py all --full
```

This command downloads the public OMNI and GOES observations, prepares the GOES
data, detects Pc5 events, builds the catalogs, and regenerates all tracked
outputs. Acquisition and detection across the complete interval are
computationally intensive. Completed checkpoints are resumed; a deliberate
restart requires both flags:

```bash
python scripts/reproduce.py all --full --force
```

Optional detection or event-catalog checkpoints can instead be supplied locally;
the repository does not provide them. With suitable checkpoints, the no-argument
command resumes from the deepest complete stage and downloads the hourly OMNI
file when it is absent:

```bash
python scripts/reproduce.py
```

Without those checkpoints, the no-argument command intentionally refuses to
launch the resource-intensive acquisition and detection stages; `--full` makes
that work explicit. Checkpoint files use pickle-based serialization and must come
from a source you trust because loading them can execute Python code.

Alternative local locations can be supplied explicitly:

```bash
python scripts/reproduce.py all \
  --checkpoint-dir /path/to/checkpoints \
  --omni-file /path/to/omni2_all_years.dat
```

Check the available local stages without changing files:

```bash
python scripts/reproduce.py status
```

Status fields ending in `_artifacts` report only that the named stage files
exist. `prepared`, `detected`, and `cataloged` add the applicable completion and
usability checks. Local completion markers bind successful stages to their named
output hashes. A present acquisition marker with mismatched outputs is an
integrity failure and requires an explicit forced restart; an `*_incomplete.json`
marker means that stage must be resumed before downstream work.

Every stage and every figure can also be run separately. See
[`docs/REPRODUCTION.md`](docs/REPRODUCTION.md) for commands and file contracts.

Optional locally supplied event-catalog checkpoints are the practical starting
point for figure-only runs. A source-data run associates each NOAA product with
prepared data by satellite and UTC date. The paper's Monte Carlo seed was not
recorded; the repository default is 2025, so new detections near the 95%
threshold may differ from a locally supplied catalog.

## Calculation conventions

The published Monte Carlo peak test used 5,000 white-noise simulations, but its
random seed was not recorded. Full source-data runs use the documented repository
seed 2025 by default; `--random-seed INTEGER` selects another repeatable
realization. Detections close to the 95% threshold can differ from locally
supplied catalogs. Figure-only runs from unchanged checkpoints are deterministic
apart from normal rendering differences between platforms.

Two conventions are important when interpreting the outputs:

- Each one-hour component window undergoes at most 50 CLEAN iterations. The
  catalog then retains zero to three highest-amplitude peaks that pass both the
  significance and Pc5-frequency tests; absent ranks remain absent.
- Figure-generating code reports `Pearson's R`, calculated between the fitted
  amplitudes and the binned median amplitudes. The tracked pre-generated Figures
  2 and 3 retain the published figures' \(R^2\) annotations so that the bundled
  images match the version of record.
- For Figure 3, each event is paired with every hourly OMNI record on the same UTC
  calendar date before the solar-wind quantiles and conditioned summaries are
  calculated. This date-level one-to-many expansion is part of the paper
  calculation.

The Figure 4 residuals use a fifth-order Butterworth high-pass filter with cutoff
frequency 1/5 yr⁻¹. Executable parameters are defined in
[`src/pc5_climatology/config.py`](src/pc5_climatology/config.py). The
[`configs/paper.toml`](configs/paper.toml) file is a human-readable summary whose
complete field set is checked against those constants; runtime stages import the
Python constants directly. The full causal path is documented in
[`docs/DATA_PROVENANCE.md`](docs/DATA_PROVENANCE.md).

## Repository contents

- `src/pc5_climatology/`: acquisition, coordinate conversion, detection,
  catalog, statistics, plotting, and orchestration code.
- `scripts/reproduce.py`: common entry point for all stages.
- `scripts/prepare_goes_data.py`, `detect_pc5_events.py`, and
  `build_event_catalog.py`: individual upstream stages.
- `scripts/plot_figure_01.py` through `plot_figure_04.py`: individual paper
  figures.
- `configs/paper.toml`: parity-checked summary of the executable calculation
  parameters and study boundaries.
- `outputs/`: the four paper figures and correlation table.
- `tests/`: focused checks of schemas and scientific transformations.

Run the tests with:

```bash
python -m pytest
```

## Citation and terms

F. Inceoglu and P. T. M. Loto’aniu, “Occurrence characteristics and
amplitude-frequency relationship of the Pc5 ULF waves from 3 decades of GOES
data,” *Scientific Reports* **15**, 36661 (2025).
[https://doi.org/10.1038/s41598-025-20474-z](https://doi.org/10.1038/s41598-025-20474-z)

Machine-readable citation metadata are provided in [`CITATION.cff`](CITATION.cff).
The software and original documentation are released under the
[MIT License](LICENSE). The tracked article figures are distributed under
CC BY-NC-ND 4.0, and GOES and OMNI data remain subject to their providers’
terms; see [`DATA_NOTICE.md`](DATA_NOTICE.md).
