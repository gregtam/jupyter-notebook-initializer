# jupyter-notebook-template

The goal of this repository is to serve as a template for any new projects to avoid the hassle of configuring all the options each time a new project is started. The file Notebook Template.ipynb serves as a starting point for any Jupyter notebook by importing useful libraries and setting up default settings (e.g., plotting settings, SQL magic functions).

## Files
### Notebooks
- Notebook Template.ipynb: This notebook is a template for any new notebook. It should be copied when creating a new notebook so that all of the libraries are already typed in along with any changes to the default matplotlib settings and magic commands to interact with SQL.

### Python Files
- credentials.py: This file includes login information into an MPP database. It is important to keep these separate from the notebook so that login information is not present inside of the notebook.

- mpp_plotting_functions.py: This file includes all function definitions for the backend plotting functions.

- sql_functions.py: This file defines utility functions for interacting with the cluster (e.g., getting the table or column names).

## Examples
We now show some examples of how to use these magic functions.
### Executing SQL Queries
We can execute SQL commands simply by typing regular SQL code and putting an `%%execsql` at the very top of the cell.
```
%%execsql
DROP TABLE IF EXISTS example_data_table;
CREATE TABLE example_data_table
   AS SELECT random()
        FROM generate_series(1, 100);
```

### Reading SQL Query Outputs
We can also read SQL query outputs and store them into a pandas DataFrame by putting `%%readsql` at the top of the cell. This will output the DataFrame below the cell. It also stores it into a pandas DataFrame called `_df` by default.

```
%%readsql
SELECT *
  FROM example_data_table;
```

If the outputted DataFrame has too many rows, we can also look at the head by typing `%%readsql -h #` where `#` is an integer value. This will show the first `#` rows in the DataFrame, but store the entire DataFrame into `_df`.

```
%%readsql -h 10
SELECT *
  FROM example_data_table;
```

Additionally, if we wanted to store this table to a different DataFrame, we can include this at the end of the top line of the cell.

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
While writing our SQL queries with string formatting, we may want to view the query beforehand as it there may be errors. We want to do this via the `%%printsql` magic function.

```
%%printsql
SELECT *
  FROM {table_name};
```

### Autofill Table Names
When tables are created, the `refresh_tables` function is run. This will place the table name in the namespace as a string. This will give us tab completion for our table names.

<img src='autofill.png' width='400'>

### Threading/Background Processes
Threading allows us to run two or more cells concurrently. This is useful if we have a long query running, but want to test other stuff inside of the same notebook. We make use of the `%%background` magic function to execute a cell in the background. The magic function will micro-manage these processes by assigning them with IDs and print and output when the code finishes. If IDs are not descriptive enough, there is also an option to put a comment when running the cell so that comment is shown upon completion.

<img src='threading_example.png'>
