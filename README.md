# jupyter-notebook-template

This repository includes a Jupyter notebook and credentials file that serve as a template for Pivotal engagements with our customers. 

The Notebook Template.ipynb notebook imports relevant libraries for machine learning and connecting to a cluster such as HAWQ GPDB. The accompanying credentials.py includes template code for login. This information is then read in by the notebook to connect to the DCA.

The Plotting Examples.ipynb notebook includes some sample plots and sets up default parameters (e.g., size, line width, font size) to make the plots look nicer. It also includes a couple of multi-plot examples and using seaborn colours.

The MPP Plotting.ipynb gives examples on how to plot when there is a lot of data in HAWQ. This requires summarizing the data into more manageable pieces.