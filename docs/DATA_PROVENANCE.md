# Data provenance and calculation path

The source-data calculation begins with NOAA/NCEI GOES Level-2 magnetometer data
and NASA/SPDF hourly OMNI data. Figure-only runs may begin with optional locally
supplied event-catalog checkpoints. Both paths end with the four paper figures and
the solar-cycle correlation table; raw observations and runtime checkpoints
remain outside Git.

## Causal flow

| Source or stage | Transformation | Output consumed by |
| --- | --- | --- |
| GOES-8 through GOES-18 Level-2 NetCDF | Quality handling, 1-minute means, MLT and mean field-aligned coordinates | Pc5 detection |
| Pc5 detection | Filtering, sliding FFT windows, enhanced CLEAN peak fitting, Monte Carlo significance | Event catalog |
| Event catalog | Select peak ranks 1–3 and normalize component tables | Figures 1–4 |
| Hourly OMNI | Missing-value handling, daily event join or annual averaging | Figures 3–4 and correlation table |
| Prepared GOES daily tables | Count records by year | Figure 4 normalization |

Executable parameter values are defined in
[`src/pc5_climatology/config.py`](../src/pc5_climatology/config.py).
[`configs/paper.toml`](../configs/paper.toml) presents the same values as a
human-readable summary, and a complete-field parity test keeps it synchronized
with the executable constants. Runtime stages import the Python constants.

## Reproduction boundary

Optional locally supplied event-catalog checkpoints are the practical starting
point for figure-only runs, and figure stages use their supplied rows without an
additional date filter. A source-data run associates every product with prepared
data by satellite and UTC date. The paper's Monte Carlo seed was not recorded;
the repository default is 2025, so new detections near the 95% threshold may
differ from a locally supplied catalog.

## Local stage markers

`acquisition_complete.json`, `detection_complete.json`, and
`catalog_complete.json` are operational metadata stored with local checkpoints.
The detection and catalog stages first write `detection_incomplete.json` or
`catalog_incomplete.json`; if an incomplete marker remains, that stage requires
resumption before its outputs can be used downstream. Status fields ending in
`_artifacts` report file presence only, while `prepared`, `detected`, and
`cataloged` apply the completion or usability checks defined for each stage.

## GOES acquisition and preparation

The inclusive study interval is 1995-07-01 through 2025-05-10. The source
satellites are GOES-8, -9, -10, -11, -12, -13, -14, -15, -16, -17, and -18.
Acquisition prepares and stores each discovered archive product version within
this interval. Detection later selects the latest version for each
`(satellite, date)` pair when extracting events.

For each daily NetCDF input, the calculation:

1. converts source time offsets to UTC datetimes;
2. reads the three EPN magnetic-field components and geographic orbit latitude,
   longitude, and radius;
3. converts fill values and magnetic-field values outside ±1024 nT to NaN;
4. rejects orbital samples outside the interval
   \([Q_1-1.5\,IQR, Q_3+1.5\,IQR]\) in any orbital coordinate;
5. calculates 1-minute means;
6. divides geographic radius by 6,371 km, then obtains magnetic local time with
   SpacePy 0.7.0 CTrans (`use_irbem=False`), WGS84, a 30-second transform-reuse
   tolerance, and the release's bundled IGRF13 coefficients;
7. uses clock-aligned 30-minute block means, interpolated to the 1-minute
   samples, to rotate the field into radial, azimuthal, and parallel mean
   field-aligned components; and
8. stores the prepared daily table; each stored table contributes one annual
   denominator record to the `year,observation_count` summary.

The prepared `df_08.pkl` through `df_18.pkl` files contain lists of daily pandas
tables. `processed_data.pkl` records source-product completion identifiers used
for version selection and resumption.

The coordinate converter initializes SpacePy in isolated process-local state,
so user-level SpacePy data cannot replace the bundled model inputs. A process
that imported SpacePy earlier is rejected before conversion. For timestamps
after IGRF13's 2025-01-01 validity endpoint, SpacePy CTrans applies the model's
endpoint behavior.

## Pc5 detection

Each mean field-aligned component is high-pass filtered with a fifth-order
Butterworth filter and 30-minute cutoff period. A complete day supplies 47
sliding windows: 60-minute windows advancing in 30-minute steps. A window is
skipped only when one component is entirely missing; partial NaNs are preserved
through filtering and consequently do not yield finite FFT peaks for that
component.

For every retained window and component, the calculation:

1. applies a Hanning window;
2. zero-pads the signal to 1,440 samples;
3. calculates the real FFT and amplitude as
   \(4|\mathrm{FFT}|/N\), in nT;
4. locates a candidate peak above 1 nT;
5. fits and subtracts a squared-sinc Hanning peak model;
6. compares the candidate with 5,000 white-noise simulations drawn from the
   window's mean and standard deviation;
7. retains peaks with false-alarm probability at most 0.05 in the
   1.6–6.7 mHz band; and
8. repeats for at most 50 CLEAN iterations.

The Monte Carlo seed used for the published event set was not recorded. A full
source-data run uses seed 2025 by default. A supplied `--random-seed` value
selects another repeatable realization of the stated calculation.

A peak persisting through adjacent overlapping windows is counted once in each
window. The catalog stage retains up to the three highest-amplitude significant
peaks per window. The component catalog fields are:

| Field | Meaning | Unit |
| --- | --- | --- |
| `date` | UTC timestamp associated with the window | UTC |
| `t1` | MLT at the first sample | hour |
| `t2` | MLT at the final sample | hour |
| `freq` | fitted peak frequency | Hz |
| `power` | fitted FFT amplitude | nT |

## Figure 1

Figure 1 reads the three component catalogs. Its top row plots
\(10\log_{10}(\mathrm{amplitude})\) against MLT and frequency. Its bottom row
uses 48 MLT bins and 50 frequency bins and displays
\(\log_{10}(\mathrm{count}+1)\).

## Figure 2

The event frequencies are split into 20 equal-width bins. Median amplitude is
calculated in four MLT sectors:

- dawn: 3 ≤ MLT < 9;
- day: 9 ≤ MLT < 15;
- dusk: 15 ≤ MLT < 21; and
- night: 21 ≤ MLT ≤ 24 together with 0 ≤ MLT < 3.

Each component and sector is fitted with
\(A(f)=c f^m\) by nonlinear least squares.

The fit annotation is `Pearson's R`, calculated as
`np.corrcoef(fitted_amplitude, median_amplitude)[0, 1]`.

## Figure 3

The hourly OMNI fields used are UTC year/day/hour, solar-wind speed, signed GSM
\(B_z\), and dynamic pressure. Provider missing-value sentinels are mapped to NaN.

The calculation uses a date-level join: OMNI timestamps are normalized to
UTC calendar dates, and every Pc5 event is paired with every hourly OMNI row on
the same date. Consequently, one event normally expands to 24 event-environment
rows, and days with more events contribute more rows to the quantiles. The first
and third quartiles of solar-wind speed, signed \(B_z\), and dynamic pressure are
then calculated on this expanded table.

Strong conditions require speed at or above its third quartile, \(B_z\) at or
below its first quartile, and pressure at or above its third quartile. Weak
conditions require speed below its first quartile, \(B_z\) above its third
quartile, and pressure below its first quartile. Moderate conditions require all
three variables to lie in their corresponding middle-quartile intervals.

Frequency binning, median amplitudes, power-law fitting, and the Pearson
correlation calculation follow the Figure 2 definitions. Figure 3 retains the
published figure's \(R^2\) annotation.

## Figure 4 and correlation table

Annual occurrence for each component is the number of retained detections in a
year divided by the number of prepared daily GOES tables stored in that year. The
denominator is read from `observation_counts_by_year.csv`, whose columns are
`year,observation_count`. If the CSV is absent, the calculation counts every stored
daily table in `df_08.pkl` through `df_18.pkl`.

Detection selects the latest archive version for each satellite and date. The
occurrence denominator is derived independently from all prepared daily tables;
if a local prepared checkpoint contains multiple archive versions for one
satellite-date, each stored table contributes one denominator record.

The OMNI input contract contains one row for every UTC hour from 1995-01-01
through the inclusive study end, 2025-05-10. This supplies the complete 1995
calendar-year mean used by Figure 4 even though GOES event acquisition begins
later that year. These hourly rows are averaged by year; later rows are excluded
before annual averaging and filtering. Figure 4
uses solar-wind speed, dynamic pressure, and the annual mean of \(|B_z|\). The lower
row applies a fifth-order Butterworth high-pass filter with cutoff frequency 1/5
yr⁻¹ to both occurrence rates and solar-wind series. The same value is summarized
in `configs/paper.toml`.

The correlation table uses `scipy.stats.pearsonr` for raw and high-pass-filtered
series in solar cycle 23 (1996–2008) and solar cycle 24 (2009–2019). The table
reports the coefficient in `Pearson's R` and its two-sided significance in
`P-value`.

## Output identity

`scripts/reproduce.py figures` renders all outputs in a temporary staging
directory, then replaces the public set only after every figure succeeds. The
four canonical names are `Fig01.jpg` through `Fig04.jpg`; the table is
`correlations_solar_cycles.csv`. In `outputs/manifest.json`, the public OMNI
record remains under `inputs`, while the singular `parameter_summary` record
identifies `configs/paper.toml`; generated figures and the table are listed under
`artifacts`.
