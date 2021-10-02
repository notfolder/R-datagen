import sqlite3
from contextlib import closing
import csv
import time

n_keys = 3
col_keys = [f'key_{i}' for i in range(n_keys)]
files = ['huge_csv_0.csv', 'huge_csv_1.csv', 'huge_csv_2.csv']

with closing(sqlite3.connect('sqlite3.db')) as conn:
    cur = conn.cursor()
    table_name = 'merge_table'

    key_sql = ",".join([f'{x} text' for x in col_keys])
    create_table_sql = f'CREATE TABLE IF NOT EXISTS {table_name} ' \
        f'({key_sql}, col_name text, col_value real, ' \
        f'PRIMARY KEY({",".join(col_keys)}, col_name))'

    key_place_sql = ",".join([f':{x}' for x in col_keys])
    insert_sql = f'INSERT INTO {table_name} VALUES({key_place_sql}, :col_name, :col_value);'

    start_time = time.time()
    prev_rows = 0
    cur.execute(create_table_sql)
    for file in files:
        with open(file, 'r') as f:
            reader = csv.reader(f)
            column_names = None
            key_index = []
            row_index = - 1
            for row in reader:
                row_index += 1
                if column_names is None:
                    column_names = row
                    for key in col_keys:
                        key_index.append(row.index(key))
                    continue
                insert_dict = {}
                for i in range(len(col_keys)):
                    insert_dict[col_keys[i]] = row[key_index[i]]
                for i in range(len(row)):
                    if i in key_index:
                        continue
                    insert_dict['col_name'] = column_names[i]
                    insert_dict['col_value'] = row[i]
                    cur.execute(insert_sql, insert_dict)
                if row_index % 1000 == 0:
                    conn.commit()
                    end_time = time.time()

                    count_sql = f'SELECT count(*) FROM {table_name};'
                    cur.execute(count_sql)
                    insert_rows = cur.fetchone()[0]

                    print(f'rows: {insert_rows}')
                    diff_time = end_time - start_time
                    print(f'rows/s: {(insert_rows - prev_rows)/diff_time}')
                    prev_rows = insert_rows
                    print(f'time: {diff_time}')
                    start_time = time.time()
