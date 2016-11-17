# jupyter-notebook-template

This repository includes Jupyter notebooks and Python files that serve as templates for engagements at Pivotal with our customers.

## Files
### Notebooks
- MPP Plotting.ipynb: This file gives examples of how to plot on the order of millions and billions of data from HAWQ or GPDB. This uses functions from mpp_plotting_functions.py to summarize the data into manageable pieces. We then use matplotlib to plot these.

- MPP ROC Curve.ipynb: This file shows how to plot an ROC curve from data in HAWQ or GPDB.

- Notebook Template.ipynb: This notebook is a template for any new notebook. It should be copied when creating a new notebook so that all of the libraries are already typed in along with any changes to the default matplotlib settings and magic commands to interact with SQL.

- Plotting Examples.ipynb: This is a reference guide for various matplotlib plots. It also sets up seaborn colours.

### Python FIles
- crednetials.py: This file includes login information into an MPP database. It is important to keep these separate from the notebook so that login information is not present inside of the notebook.

- mpp_plotting_functions.py: This file includes all function definitions for the backend plotting functions.

- sql_functions.py: This file defines utility functions for interacting with the cluster (e.g., getting the table or column names).
