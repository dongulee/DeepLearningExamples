import psycopg2
import psycopg2.extras
from tabulate import tabulate
import sys

SHOW_TABLES = """
SELECT *
FROM pg_catalog.pg_tables
WHERE schemaname != 'pg_catalog' AND 
    schemaname != 'information_schema';
"""

INSERT_INTO = """
INSERT INTO {table_name} {columns} VALUES {values}
"""



# data = [(0, 0, 'a'), (0, 1, 'b')]
# psycopg2.extras.execute_values(
#     cur, INSERT_INTO.format(table_name='categories', columns='', values='%s'),
#     data, template=None, page_size=100
# )

# conn.commit()
# conn.close()

class dbcon():
    def __init__(self, dsn_string):
        self.dsn = dsn_string
        self.conn = psycopg2.connect(self.dsn)

    def show_tables(self):
        cursor = self.conn.cursor()
        cursor.execute(SHOW_TABLES)
        rows = cursor.fetchall()
        desc = cursor.description

        print(tabulate(rows, headers=[col[0] for col in desc],tablefmt='github'))
        cursor.close()

    def get_tables(self):
        cursor = self.conn.cursor()
        cursor.execute(SHOW_TABLES)
        rows = cursor.fetchall()
        return rows
    def adhoc_query(self, query):
        cursor = self.conn.cursor()
        cursor.execute(query)
        rows = cursor.fetchall()
        return rows
        
    def insert_rows(self, tname: str, data: list, cols: list=''):
        cursor = self.conn.cursor()
        try:
            psycopg2.extras.execute_values(
                cursor,
                INSERT_INTO.format(table_name=tname, columns=cols, values='%s'),
                data,
                template=None,
                page_size=100
            )
        except:
            print(sys.exc_info()[0])
        self.conn.commit()
        cursor.close()
        print("inserts end")