# `data/` — raw downloads and scratch

Everything in this directory except this README is gitignored. It holds ~4.8 GB
of raw source data and intermediates that are too large to version.

**Derived model inputs do not belong here.** They go in `dgen_os/input_data/`,
which is tracked, so the model's actual inputs stay in the repo. `data/` is only
for raw downloads and working files.

## What lives here, and where it came from

| File | Size | Source |
|---|---|---|
| `tts_state_download_data_2026_release.xlsx` | 1.1 MB | LBNL *Tracking the Sun*, **published aggregated state medians**. This is the source for baseline PV cost. Download from the LBNL Tracking the Sun data page ("state-level data" / viz-tool download). |
| `TTS_LBNL_public_file_28-Jul-2026_all.csv` | 2.3 GB | LBNL *Tracking the Sun* row-level public file (2026 release). |
| `TTS_LBNL_public_file_29-Sep-2025_all.csv` | 1.8 GB | Same, 2025 release. |
| `usurdb.csv` | 183 MB | OpenEI Utility Rate Database full dump. |
| `stanford_files/data_county_to_wholesale.pkl` | 208 MB | County → wholesale hourly price mapping. |
| `Ohm Analytics - Battery Attachment Rates` | — | Vendor data: quarterly storage attachment rates by state. |
| `agent_df_base_res_national_*.pkl` | large | ResStock-derived agent files. Canonical copies live in the `dgen-assets` GCS bucket under `input_agents/`; Batch jobs fetch them from there. |

## Use the aggregated file, not the row-level one, for prices

The row-level public file **cannot reproduce LBNL's published medians** and should
not be used to derive cost anchors. It is a partial subset: NJ is 100%
price-redacted, only 27 states appear, and CT/NY are undercounted (CT 2024 shows
2,742 records against the viz tool's 3,663). Computed national medians came out
~4% below LBNL's published figures. Use
`tts_state_download_data_2026_release.xlsx`.

## Baseline PV cost is built from here

`Notebooks/build_baseline_upfront_cost.ipynb` reads the aggregated xlsx above and
writes `dgen_os/input_data/state_upfront_cost_lbnl_2025.csv` (tracked).
`Notebooks/adjust_pv_batt_price_trajectories.ipynb` then reads that CSV and
uploads the per-state price tables to Cloud SQL.
