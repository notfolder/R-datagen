import sqlite3
from contextlib import closing
import pandas as pd
import csv
import time

n_keys = 3
col_keys = [f'key_{i}' for i in range(n_keys)]
files = ['huge_csv_0.csv', 'huge_csv_1.csv', 'huge_csv_2.csv']

start_time = time.time()

with closing(sqlite3.connect('sqlite3.db')) as conn:
    cur = conn.cursor()
    table_name = 'merge_table'

    key_sql = ",".join([f'{x} text' for x in col_keys])
    create_table_sql = f'CREATE TABLE IF NOT EXISTS {table_name} ' \
        f'({key_sql}, col_name text, col_value real, ' \
        f'PRIMARY KEY({",".join(col_keys)}, col_name))'
    cur.execute(create_table_sql)
    conn.commit()

for file in files:

    with open(file, 'r') as f:
        reader = csv.reader(f)
        column_names = next(reader)

    key_index = []
    for key in col_keys:
        key_index.append(column_names.index(key))

    value_column_indexs = [i for i in range(len(column_names))
                           if i not in key_index]

    # カラム分割
    col_delta = 1000
    row_delta = 10000
    start_time = time.time()
    insert_rows = 0
    prev_rows = 0
    for col_current_index in range(0, len(value_column_indexs), col_delta):
        usecols = key_index + value_column_indexs[
            col_current_index:col_delta]
        for df in pd.read_csv(file, chunksize=row_delta, usecols=usecols,
                              low_memory=True):
            with closing(sqlite3.connect(':memory:')) as conn:
                cur = conn.cursor()
                cur.execute(f"ATTACH DATABASE 'sqlite3.db' as sqlite3")
                df.to_sql(f'{table_name}_memory', conn, index=None)
                for i in range(len(usecols)):
                    col_index = usecols[i]
                    select_sql = f'SELECT {",".join(col_keys)}, ' \
                                 f"'{column_names[col_index]}' as col_name, " \
                                 f'{column_names[col_index]} as col_value ' \
                                 f'FROM {table_name}_memory;'
                    insert_sql = f'INSERT INTO sqlite3.{table_name} ' \
                                 f'{select_sql}'
                    cur.execute(insert_sql)

                conn.commit()

                count_sql = f'SELECT count(*) FROM sqlite3.{table_name};'
                cur.execute(count_sql)
                insert_rows = cur.fetchone()[0]

            end_time = time.time()
            print(f'rows: {insert_rows}')
            diff_time = end_time - start_time
            print(f'rows/s: {(insert_rows - prev_rows)/diff_time}')
            prev_rows = insert_rows
            print(f'time: {diff_time}')
            start_time = time.time()
