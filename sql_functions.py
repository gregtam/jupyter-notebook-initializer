from textwrap import dedent

import pandas as pd
import pandas.io.sql as psql
import psycopg2

import credentials

# Connect to database
conn = psycopg2.connect(database=credentials.database,
						user=credentials.user,
						password=credentials.password,
						host=credentials.host
					   )
conn.autocommit = True

def get_table_names(schema_name=None, view_query=False):
	"""
	Gets all the table names in the specified database

	Inputs:
	schema_name: Specify the schema of interest. If left blank,
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

def get_column_names(schema_name, table_name, order_by='ordinal_position', reverse=False, view_query=False):
	"""
	Gets all of the column names of a specific table.

	Inputs:
	schema_name: Name of the schema
	table_name: Name of the table 
	order_by: Specified way to order columns. Can be one of
	          ordinal_position, alphabetically. 
	          (Default: ordinal_position)
	reverse: If True, then reverse the ordering (Default: False)
	"""

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


