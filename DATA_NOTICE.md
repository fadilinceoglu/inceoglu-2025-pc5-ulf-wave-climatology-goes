# Data notice

The repository does not redistribute the complete GOES observations, the hourly
OMNI archive, or the detection and event-catalog checkpoints. The reproduction
code retrieves public source observations or reads locally supplied checkpoints.

The MIT License in [`LICENSE`](LICENSE) applies to the repository software and
documentation. It does not replace the terms, attribution requests, or citation
requirements of the data providers. Users are responsible for consulting the
provider pages below before redistributing source data or derived data products.

## GOES magnetometer data

Provider: US National Centers for Environmental Information (NOAA/NCEI).

Study use: science-quality Level-2 high-resolution magnetic-field and orbit data
from GOES-8 through GOES-18 over 1995-07-01 through 2025-05-10. The acquisition
stage enforces this interval even when the archive contains newer observations.

Product and access pages:

- [GOES 1–15 Space Weather Instruments](https://www.ncei.noaa.gov/products/goes-1-15/space-weather-instruments)
- [GOES-R Magnetometer](https://www.ncei.noaa.gov/products/goes-r-magnetometer)
- [GOES-8 through GOES-15 archive root](https://www.ncei.noaa.gov/data/goes-space-environment-monitor/access/science/mag/)
- [GOES-R archive root](https://data.ngdc.noaa.gov/platforms/solar-space-observing-satellites/goes/)

The [NCEI Archive and Open Data Policy](https://www.ncei.noaa.gov/archive)
states that NOAA-created US government data are in the public domain in the
United States. NCEI holdings supplied by other originators remain subject to
their source licenses, so the metadata for a particular product control where an
exception is stated.

The code reads NetCDF magnetic-field components, timestamps, and geographic orbit
latitude, longitude, and radius. Fill values and magnetic-field magnitudes outside
±1024 nT are treated as missing. Orbit samples outside 1.5 interquartile ranges
from the first or third quartile are removed. Data are reduced to 1-minute means
and transformed to mean field-aligned radial, azimuthal, and parallel components.

## Hourly OMNI data

Provider: NASA Goddard Space Flight Center, Space Physics Data Facility (SPDF).

Study use: hourly solar-wind speed, interplanetary magnetic-field values, and
dynamic pressure for Figures 3 and 4 and the correlation table.

- [OMNI low-resolution archive](https://spdf.gsfc.nasa.gov/pub/data/omni/low_res_omni/)
- [Direct `omni2_all_years.dat` route](https://spdf.gsfc.nasa.gov/pub/data/omni/low_res_omni/omni2_all_years.dat)
- [OMNI documentation](https://omniweb.gsfc.nasa.gov/html/ow_data.html)

The [SPDF data-use policy](https://spdf.gsfc.nasa.gov/data_use_policy.html)
states that SPDF content and data are public domain and made available under CC0
unless otherwise noted. It also asks users to follow scholarly citation norms for
the data package and its source publication.

`omni2_all_years.dat` is a near-current aggregate and can change as SPDF updates
the archive. Preserve the downloaded file and record its checksum when byte-for-
byte reruns matter. The downloaded file must contain every hourly timestamp
from 1995-01-01 through the inclusive study end. Figure 3 joins only dates
present in the event catalog. Figure 4 groups available values by calendar
year through 2025-05-10; later rows are excluded before annual averaging and
high-pass filtering.
Missing-value sentinels are converted to NaN before statistics are calculated.

## Local runtime files

The default local layout is:

```text
data/
  external/
    omni2_all_years.dat
  checkpoints/
    processed_data.pkl
    df_08.pkl ... df_18.pkl
    observation_counts_by_year.csv
    acquisition_complete.json
    Frequency_Power_radial_new_1h.pkl
    Frequency_Power_azimuthal_new_1h.pkl
    Frequency_Power_parallel_new_1h.pkl
    detection_complete.json
    detection_incomplete.json  # present only while resumption is required
    radial_powers_freq_mlt_date.pkl
    az_powers_freq_mlt_date.pkl
    par_powers_freq_mlt_date.pkl
    catalog_complete.json
    catalog_incomplete.json    # present only while resumption is required
```

These directories are ignored by Git. The Joblib files are Python pickle-based
serialization and must be treated as executable input: load only checkpoints you
created or obtained from a source you trust.
`observation_counts_by_year.csv` has
the portable public schema `year,observation_count`.

The three `*_complete.json` files are local operational metadata, not research
data. `acquisition_complete.json` is written only after source discovery and
product processing finish without unresolved items, and it records hashes for
every prepared output. Detection and catalog stages
write their complete markers after successful output promotion; a remaining
`detection_incomplete.json` or `catalog_incomplete.json` means resumption is
required. Status fields ending in `_artifacts` report file presence only, while
`prepared`, `detected`, and `cataloged` apply the relevant completion or usability
checks.

The four generated figures and correlation table are intentionally tracked under
`outputs/`. They are results of the paper calculation, not substitutes for the
machine-readable GOES or OMNI observations.

## Paper-output terms

The pre-generated files `outputs/figures/Fig01.jpg` through `Fig04.jpg` reproduce
figures from the article:

> F. Inceoglu and P. T. M. Loto’aniu, *Scientific Reports* **15**, 36661
> (2025), [https://doi.org/10.1038/s41598-025-20474-z](https://doi.org/10.1038/s41598-025-20474-z).
> © The Author(s) 2025.

The figure files are distributed under the article's
[Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International
License](https://creativecommons.org/licenses/by-nc-nd/4.0/), including its
attribution, non-commercial-use, and no-adaptations conditions. The MIT License
applies to the repository software and original documentation; it does not
relicense these figure files or the generated research outputs.
