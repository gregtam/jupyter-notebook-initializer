# jupyter-notebook-template

The goal of this repository is to serve as a template for any new projects to avoid the hassle of configuring all the options each time a new project is started. The 'Notebook Template.ipynb' serves as a starting point for any Jupyter notebook by importing useful libraries and setting up default settings (e.g., plotting settings, SQL magic functions).

## Files
### Notebooks
- MPP Plotting.ipynb: This file gives examples of how to plot on the order of millions and billions of data from HAWQ or GPDB. This uses functions from mpp_plotting_functions.py to summarize the data into manageable pieces. We then use matplotlib to plot these.

- MPP ROC Curve.ipynb: This file shows how to plot an ROC curve from data in HAWQ or GPDB.

- Notebook Template.ipynb: This notebook is a template for any new notebook. It should be copied when creating a new notebook so that all of the libraries are already typed in along with any changes to the default matplotlib settings and magic commands to interact with SQL.

- Plotting Examples.ipynb: This is a reference guide for various matplotlib plots. It also sets up seaborn colours.

### Python FIles
- credentials.py: This file includes login information into an MPP database. It is important to keep these separate from the notebook so that login information is not present inside of the notebook.

- mpp_plotting_functions.py: This file includes all function definitions for the backend plotting functions.

- sql_functions.py: This file defines utility functions for interacting with the cluster (e.g., getting the table or column names).

## Examples
### Executing SQL Queries
We can execute SQL commands simply by typing regular SQL code and putting an <code>%%execsql</code> at the very top of the cell.
```
%%execsql
DROP TABLE IF EXISTS example_data_table;
CREATE TABLE example_data_table
   AS SELECT random()
        FROM generate_series(1, 100);
```

### Reading SQL Query Outputs
We can also read SQL query outputs and store them into a pandas DataFrame by putting <code>%%readsql</code> at the top of the cell. This will output the DataFrame below the cell. It also stores it into a variable called <code>_df</code> by default.

```
%%readsql
SELECT *
  FROM example_data_table;
```

If the outputted DataFrame has too many rows, we can also look at the head by typing <code>%%readsql -h #</code> where <code>#</code> is an integer value. This will show the first <code>#</code> rows in the DataFrame, but store the entire DataFrame into <code>_df</code>.

```
%%readsql -h 10
SELECT *
  FROM example_data_table
```

Additionally, if we wanted to store this table to a different variable, we can include this at the end of the top line of the cell.

```
%%readsql -h 10 other_df
SELECT *
  FROM example_data_table;
```

One final possibility is to use string formatting. Suppose we have the name of the table as a variable in Python.

```
table_name = 'example_data_table'
```

We can then select use this variable in our SQL query string.

```
%%readsql -h 10 other_df
SELECT *
  FROM {table_name};
```

### Printing SQL Queries
While writing our SQL queries with string formatting, we may want to view the query beforehand as it there may be errors. We want to this via the <code>%%printsql</code> magic function.

```
%%printsql
SELECT *
  FROM {table_name}
```
