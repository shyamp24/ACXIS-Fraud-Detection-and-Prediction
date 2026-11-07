# Original experiment notebooks

These are the notebooks used to run the experiments reported in the manuscript, with their saved outputs.
Local data paths were replaced by `PATH/TO/<file>.csv` placeholders; set them to your own copy of the datasets before running.

| Notebook | Dataset |
| :--- | :--- |
| `ACXIS_Cell_wise_corrected.ipynb` | PaySim |
| `ACXIS_IEEE_CIS_cell_wise_fixed_memory_.ipynb` | IEEE-CIS |

The Brier/ECE values in manuscript Section 4.1.1 come from the cells headed "CELL 12" (PaySim) and "CELL 15" (IEEE-CIS).
See the "Calibration Metrics Provenance" section of the top-level README. The scripted pipeline in `scripts/` and the cached
arrays in `cache/` come from a different run, so numbers recomputed from them can differ slightly from the notebook outputs.
