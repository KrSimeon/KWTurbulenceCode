This repository contains the code and data analysis for the paper:

Kelvin Wave Induced Shear-Instabilities and Implications for Ozone Across the Tropical Tropopause

Simeon Kraus, Daniel Kunkel, Peter Hoor (2026). , doi:

## Requirements

- Python 3.8 or newer
- See `requirements.txt` for all required packages
  
## Installation

Install the required Python packages using:

```sh
pip install -r requirements.txt
```

## Contents
## Contents

| Path | Purpose |
|---|---|
| `Functions/` | Functions for calculations and data handling |
| `DataProcessing/` | Scripts that calculate all used data |
| `PlottingNotebooks/` | Jupyter Notebooks that contain code for all plots |

## Outputs

| Output | Script |
|---|---|
| Equatorial wave decomposition | `EqWavesMain.py` |
| Occurrence frequencies of turbulence indices | `DailyMapMain.py` |
| Composites at low Ri events | `Composites.py` |
| ERA5 datasets at SHADOZ locations and days | `Era5atShadoz.py` |
| Fig. 1 | `PlottingNotebooks/OccurrenceFrequencyMap.ipynb` |
| Fig. 2 | `PlottingNotebooks/CompositeAnalysis.ipynb` |
| Fig. 3 | `PlottingNotebooks/Shadoz.ipynb` |
| Fig. S3 | `PlottingNotebooks/OccurrenceFrequencyMap.ipynb` |
| Fig. S4 | `PlottingNotebooks/OccurrenceFrequencyMap.ipynb` |
| Fig. S5 | `PlottingNotebooks/CompositeAnalysis.ipynb` |
| Fig. S6 | `PlottingNotebooks/CompositeAnalysis.ipynb` |
| Fig. S7 | `PlottingNotebooks/CompositeAnalysis.ipynb` |

## Citation

If you use this code or data in your research, please cite this repository:

DOI: [10.5281/zenodo.24342019](https://doi.org/10.5281/zenodo.24342019).

**Contact:** [sikraus@uni-mainz.de](mailto:sikraus@uni-mainz.de)
