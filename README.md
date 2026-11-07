# Trophic Transfer Index (TTI) Calculator

This repository contains a unified, GPU-accelerated pipeline for calculating the Trophic Transfer Index (TTI) across various regional seas (Baltic Sea, Black Sea, Mediterranean Sea, SWES). 

The TTI is defined as the maximum correlation between Primary Production Rate (PPR) and Zooplankton Grazing (GR), evaluated across a sliding temporal window with varying time lags.

To provide flexibility for different scientific workflows, this repository provides mathematically identical implementations in both **R** and **Python**.

## ⚠️ CRITICAL NOTICE: Spatial Grid Orientation & `cdo diff`

If you compare the NetCDF outputs of the R script and the Python script using CDO (e.g., `cdo diff`), **they will appear completely different**, even yielding massive absolute differences. 

**This is a known artifact of how R and Python handle spatial coordinates, not a mathematical error.**
* **Python (`xarray`)**: Strictly preserves the exact spatial orientation (e.g., South-to-North) of the input NetCDF.
* **R (`terra`)**: Automatically inverts South-to-North axes to a standard North-to-South Top-Left origin upon reading, and writes the output in this inverted state. 
* *Note: Viewers like `ncview` often automatically flip axes for display, masking this discrepancy.*

Both outputs are statistically correct. If downstream tools require the original GETM/ERSEM `latc`/`lonc` South-to-North orientation, **the Python output is the faithful representation of the input grid.**

## Repository Structure

```text
tti-framework/
├── input/
│   ├── config_baltic_monthly.yaml
│   ├── config_blacksea_monthly.yaml
│   ├── config_medsea_monthly.yaml
│   ├── config_swes_monthly.yaml
│   └── *.nc (Small 5-year test datasets)
├── output/
│   └── (Generated NetCDFs, CSVs, and PNGs go here)
├── run_tti_master.R         # Unified R pipeline
├── run_tti_master.py        # Unified Python pipeline
├── requirements.txt         # Python dependencies
├── .gitignore
└── README.md