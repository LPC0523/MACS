# Demo dataset

One 5 T B0 field-map acquisition (dual-echo GRE: Echo1, Echo2 and the B0Map
phase series, 2 mm isotropic, 348 slices) for running the one-click pipeline
end to end:

```bash
python tests/test_pipeline.py            # uses test_dic/ by default
```

De-identified before release: name and ID replaced with `MACS_DEMO`; birth
date, age, sex, weight, institution, station, operator and device serial
removed; all dates set to 2000-01-01; UIDs regenerated; private tags removed.
Acquisition times of day and all imaging parameters are kept, because series
pairing and the frequency baseline use them.
