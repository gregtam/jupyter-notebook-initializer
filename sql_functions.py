from __future__ import division
from textwrap import dedent

import pandas as pd
import pandas.io.sql as psql
import psycopg2
from sqlalchemy import Column, Table

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
    print_query - If True, print the resulting query. (Default: False)
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
    print_query - If True, print the resulting query. (Default: False)
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
    """Gets all the table names in the specified database.

    Inputs:
    conn - A psycopg2 connection object
    schema_name -  Specify the schema of interest. If left blank, then
                   it will return all tables in the database.
    print_query - If True, print the resulting query. (Default: False)
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
    print_query - If True, print the resulting query. (Default: False)
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


def get_process_ids(conn, usename=None, print_query=False):
    """Gets the process IDs of current running activity.

    Inputs:
    conn - A psycopg2 connection object
    usename - Username to filter by. If None, then do not filter.
              (Default: None)
    print_query - If True, print the resulting query. (Default: False)

    Returns a Pandas DataFrame
    """

    if usename is None:
        where_clause = ''
    else:
        where_clause = "WHERE usename = '{}'".format(usename)

    sql = '''
    SELECT datname, procpid, usename, current_query, query_start
      FROM pg_stat_activity
     {}
    '''.format(where_clause)

    if print_query:
        print dedent(sql)

    return psql.read_sql(sql, conn)


def kill_process(conn, pid, print_query=False):
    """Kills a specified process.

    Inputs:
    conn - A psycopg2 connection object
    pid - The process ID that we want to kill
    print_query - If True, print the resulting query. (Default: False)
    """

    sql = '''
    SELECT pg_cancel_backend({});
    '''.format(pid)

    psql.execute(sql, conn)


def save_table(selected_table, table_name, metadata, engine,
               distribution_key=None, randomly=False, drop_table=False,
               print_query=False):
    """Saves a SQLAlchemy selectable object to database.
    
    Inputs:
    selected_table - A SQLAlchemy selectable object we wish to save
    table_name - A string indicating what we want to name the table
    metadata - SQLAlchemy MetaData object
    engine - SQLAlchemy engine object
    distribution_key - The specified distribution key, if applicable
                       (Default: None)
    randomly - If True, distribute table randomly (Default: False)
    drop_table - If True, drop the table if it exists before creating
                 new table (Default: False)
    print_query - If True, print the resulting query. (Default: False)
    """

    def _create_empty_table(create_str, columns_str, distribution_str,
                            engine, print_query):
        create_table_str = '{create_str}{columns_str}) {distribution_str};'\
            .format(**locals())
        if print_query:
            print create_table_str
        # Create the table with no rows
        psql.execute(create_table_str, engine)

    def _get_distribution_str(distribution_key, randomly):
        # Set distribution key string
        if distribution_key is None and not randomly:
            return ''
        elif distribution_key is None and randomly:
            return 'DISTRIBUTED RANDOMLY'
        elif distribution_key is not None and not randomly:
            if isinstance(distribution_key, Column):
                return 'DISTRIBUTED BY ({})'.format(distribution_key.name)
            elif isinstance(distribution_key, str):
                return 'DISTRIBUTED BY ({})'.format(distribution_key)
            else:
                raise ValueError('distribution_key must be a string or a Column.')
        else:
            raise ValueError('distribution_key and randomly cannot both be specified.')
        
    if drop_table:
        psql.execute('DROP TABLE IF EXISTS {};'.format(table_name), engine)

    # Set create table string
    create_str = 'CREATE TABLE {} ('.format(table_name)
    # Specify column names and data types
    columns_str = ',\n'.join(['{} {}'.format(s.name, s.type)
                                  for s in selected_table.c])
    # Set distribution key
    distribution_str = _get_distribution_str(distribution_key, randomly)
 
    # Create an empty table with the desired columns
    _create_empty_table(create_str, columns_str, distribution_str, engine,
                        print_query)
 
    created_table = Table(table_name, metadata, autoload=True)
    # Insert rows from selected table into the new table
    insert_sql = created_table\
        .insert()\
        .from_select(selected_table.c,
                     select=selected_table
                    )
    psql.execute(insert_sql, engine)