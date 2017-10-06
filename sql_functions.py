from __future__ import division
from textwrap import dedent

import pandas as pd
import pandas.io.sql as psql
import psycopg2

def _separate_schema_table(full_table_name, conn):
    """Separates schema name and table name"""
    if '.' in full_table_name:
        return full_table_name.split('.')
    else:
        schema_name = psql.read_sql('SELECT current_schema();', conn).iloc[0, 0]
        table_name = full_table_name
        return schema_name, full_table_name


def clear_schema(schema_name, conn, print_query=False):
    """Remove all tables in a given schema.

    Inputs:
    schema_name - Name of the schema in SQL
    conn - A psycopg2 connection object
    print_query - If True, print the resulting query.
    """

    sql = '''
    SELECT table_name
      FROM information_schema.tables
     WHERE table_schema = '{schema_name}'
    '''.format(**locals())

    if print_query:
        print dedent(sql)

    table_names = psql.read_sql(sql, conn).table_name

    for table_name in table_names:
        del_sql = 'DROP TABLE IF EXISTS {schema_name}.{table_name};'\
            .format(**locals())
        psql.execute(del_sql, conn)


def get_column_names(full_table_name, conn, order_by='ordinal_position',
                     reverse=False, print_query=False):
    """Gets all of the column names of a specific table.

    Inputs:
    conn - A psycopg2 connection object
    full_table_name - Name of the table in SQL. Input can also include
                      have the schema name prepended, with a '.', e.g.
                      'schema_name.table_name'.
    order_by - Specified way to order columns. Can be one of
               ordinal_position, alphabetically. 
               (Default: ordinal_position)
    reverse - If True, then reverse the ordering (Default: False).
    print_query - If True, print the resulting query.
    """

    schema_name, table_name = _separate_schema_table(full_table_name, conn)

    if reverse:
        reverse_key = ' DESC'
    else:
        reverse_key = ''

    sql = '''
    SELECT table_name, column_name, data_type
      FROM information_schema.columns
     WHERE table_schema = '{schema_name}'
       AND table_name = '{table_name}'
     ORDER BY {order_by}{reverse_key};
    '''.format(**locals())

    if print_query:
        print dedent(sql)

    return psql.read_sql(sql, conn)


def get_function_code(function_name, conn, print_query=False):
    """Returns a SQL function's source code."""
    sql = '''
    SELECT pg_get_functiondef(oid)
      FROM pg_proc
     WHERE proname = '{function_name}'
    '''.format(**locals())

    if print_query:
        print dedent(sql)

    return psql.read_sql(sql, conn).iloc[0, 0]


def get_table_names(conn, schema_name=None, print_query=False):
    """ Gets all the table names in the specified database

    Inputs:
    conn - A psycopg2 connection object
    schema_name -  Specify the schema of interest. If left blank, then
                   it will return all tables in the database.
    print_query - If True, print the resulting query.
    """

    if schema_name is None:
        where_clause = ''
    else:
        where_clause = "WHERE table_schema = '{}'".format(schema_name)

    sql = '''
    SELECT table_name
      FROM information_schema.tables
     {}
    '''.format(where_clause)

    if print_query:
        print dedent(sql)

    return psql.read_sql(sql, conn)


def get_percent_missing(full_table_name, conn, print_query=False):
    """This function takes a schema name and table name as an input and
    creates a SQL query to determine the number of missing entries for
    each column. It will also determine the total number of rows in the
    table.

    Inputs:
    full_table_name - Name of the table in SQL. Input can also include
                      have the schema name prepended, with a '.', e.g.,
                      'schema_name.table_name'.
    conn - A psycopg2 connection object
    print_query - If True, print the resulting query.
    """

    column_names = get_column_names(full_table_name, conn).column_name
    schema_name, table_name = _separate_schema_table(full_table_name, conn)

    num_missing_sql_list = ['SUM(({name} IS NULL)::INTEGER) AS {name}'\
                                .format(name=name) for name in column_names]

    num_missing_list_str = ',\n           '.join(num_missing_sql_list)

    sql = '''
    SELECT {num_missing_list_str},
           COUNT(*) AS total_count
      FROM {schema_name}.{table_name};
    '''.format(**locals())

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
    pct_df['table_name'] = table_name

    if print_query:
        print dedent(sql)

    return pct_df