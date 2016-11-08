from textwrap import dedent

import pandas as pd
import pandas.io.sql as psql
import psycopg2

import credentials

def _separate_schema_table(full_table_name, conn):
    """Separates schema name and table name"""
    if '.' in full_table_name:
        return full_table_name.split('.')
    else:
        schema_name = psql.read_sql('SELECT current_schema();', conn).iloc[0, 0]
        table_name = full_table_name
        return schema_name, full_table_name

def get_table_names(conn, schema_name=None, view_query=False):
    """
    Gets all the table names in the specified database

    Inputs:
    conn - A psycopg2 connection object
    schema_name -  Specify the schema of interest. If left blank,
                   then it will return all tables in the database.
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

    if view_query:
        print sql

    return psql.read_sql(sql, conn)

def get_column_names(full_table_name, conn, order_by='ordinal_position', reverse=False, view_query=False):
    """
    Gets all of the column names of a specific table.

    Inputs:
    conn - A psycopg2 connection object
    table_name - Name of the table in SQL. Input can also
                 include have the schema name prepended, with 
                 a '.', e.g. 'schema_name.table_name'.
    order_by - Specified way to order columns. Can be one of
               ordinal_position, alphabetically. 
               (Default: ordinal_position)
    reverse - If True, then reverse the ordering (Default: False).
    view_query - If True, print the resulting query.
    """

    schema_name, table_name = _separate_schema_table(full_table_name, conn)

    if reverse:
        reverse_key = ' DESC'
    else:
        reverse_key = ''

    sql = '''
    SELECT table_name, column_name
      FROM information_schema.columns
     WHERE table_schema = '{schema_name}'
       AND table_name = '{table_name}'
     ORDER BY {ordering}{reverse};
    '''.format(schema_name = schema_name,
               table_name = table_name,
               ordering = order_by,
               reverse = reverse_key
              )

    if view_query:
        print sql

    return psql.read_sql(sql, conn)


