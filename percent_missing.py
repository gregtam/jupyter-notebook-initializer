from textwrap import dedent

import numpy as np
import pandas as pd
import pandas.io.sql as psql
import psycopg2

def get_column_names(schema_name, table_name):
    """
    This function takes a schema name and table name as an input
    and returns the columns as a pandas Series.
    """
    get_col_names_sql = '''
    SELECT column_name
      FROM information_schcema.columns
     WHERE table_schema = '{schema_name}'
       AND table_name = '{table_name}'
     ORDER BY column_name;
    '''.format(schema_name=schema_name, table_name=table_name)

    return psql.read_sql(get_col_names_sql, conn).column_name


def get_percent_missing(schema_name, table_name):
    """
    This function takes a schema name and table name as an input
    and creates a SQL query to determine the number of missing 
    entries for each column. It will also determine the total
    number of rows in the table.

    Returns:
    A pandas DataFrame with a column of the column column names
    in the desired table and a column of the percentage of missing
    values.
    """
    column_names = get_column_names(schema_name, table_name)
    num_missing_sql_list = ['SUM(({name} IS NULL)::integer) AS {name}'.format(name=name) for name in column_names]

    get_missing_count_sql = '''
    SELECT {0},
           COUNT(*) AS total_count
      FROM {schema_name}.{table_name};
    '''.format(',\n           '.join(num_missing_sql_list),
               schema_name=schema_name,
               table_name=table_name
              )

    # Read in the data from the query and transpose it
    pct_df = psql.read_sql(sql, conn).T
    # Rename the column to 'pct_null'
    pct_df.columns = ['pct_null']
    # Get the number of rows of table_name
    total_count = pct_df.ix['total_count', 'pct_null']
    # Remove the total_count from the DataFrame
    pct_df = pct_df[:-1]/total_count
    pct_df.reset_index(inplace=True)
    pct_df.columns = ['column_name', 'pct_null']
    pct_table['table_name'] = table_name

    return pct_df


