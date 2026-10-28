# AGN Radio Luminosity Estimation

Python script for estimating rest-frame 6 GHz monochromatic luminosities for a fixed list of AGN targets using photometry retrieved from the NASA/IPAC Extragalactic Database (NED).

## Installation and usage

```bash
pip install numpy pandas astropy astroquery openpyxl
python agn_luminosity.py
```

The script produces `AGN_200_Luminosity.xlsx`, containing target names and luminosities in units of \(10^{20}\,\mathrm{erg\,s^{-1}\,Hz^{-1}}\).

## Method

Near-6 GHz measurements are averaged when available; otherwise, a power-law fit estimates the 6 GHz flux density. Spectral slopes are used internally for the K-correction but are not included in the output. The adopted cosmology is flat ΛCDM with H₀ = 70 km s⁻¹ Mpc⁻¹ and Ωₘ = 0.3.

## Scope and limitations

This release preserves the original luminosity-extraction procedure. It assumes that NED `Photometry Measurement` values are in Jy without record-by-record unit verification. Observing epochs, angular resolutions, and upper limits are not explicitly homogenized or handled. An unavailable spectral slope defaults to α = 0 for the luminosity calculation.

Live database queries may differ from historical results. This repository provides the luminosity-extraction script, not the complete statistical analysis of the study.

## Data source

[NASA/IPAC Extragalactic Database (NED)](https://ned.ipac.caltech.edu/)
