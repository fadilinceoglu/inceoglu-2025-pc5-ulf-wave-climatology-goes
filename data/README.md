# Runtime data

No study-scale data are committed in this directory. The acquisition and
detection stages cover three decades of observations and create large local
checkpoints, so Git ignores every runtime data directory except deliberately small
test fixtures.

The default layout created by the pipeline is:

```text
data/
  external/
    omni2_all_years.dat
  checkpoints/
    processed_data.pkl
    faulty_data.pkl
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

Run `python scripts/reproduce.py status` to inspect this contract. Use
`--checkpoint-dir` and `--omni-file` to point at alternative local locations.

The complete source path is:

```bash
python scripts/reproduce.py all --full
```

The command downloads hourly OMNI data from NASA/SPDF and GOES Level-2
magnetometer data from NOAA/NCEI.
See [`../DATA_NOTICE.md`](../DATA_NOTICE.md) for source links, roles, and terms.

Joblib checkpoints use pickle-based serialization. Load only files produced by
this code or obtained from a source you trust. The CSV observation summary uses
the schema:

```text
year,observation_count
```

The three complete markers are local operational metadata. The acquisition marker
is written only after the requested source interval has been discovered and no
product failure remains unresolved, and its hashes bind that claim to every
prepared output. Detection and catalog complete markers are
written after successful stage completion. A remaining detection or catalog
incomplete marker means that stage requires resumption before downstream use.

In `python scripts/reproduce.py status`, fields ending in `_artifacts` report only
that the named files exist. `prepared`, `detected`, and `cataloged` add their
applicable completion or usability checks.
