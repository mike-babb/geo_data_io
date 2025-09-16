__author__ = 'babbm'
# import ACS PUMS microdata stored as a csv to a sqlite db

# standard libraries
import os
import sqlite3

# external libraries
import pandas as pd
import numpy as np

def create_table_def_statement(base_sql, col_names):

    for i, tcn in enumerate(col_names):
        base_sql += tcn + ' TEXT, '

    # remove the last comma and space and add the closing parantheses
    base_sql = base_sql[:-2] + ');'

    return base_sql


def import_data_to_sqlite(input_file, db_conn, db_curr, table_name):

    # now, read the lines into sql

    # remove line endings
    line_endings = ['\r', '\r\n', '\n']

    insert_sql = 'insert into ' + table_name + ' values ('

    line_count = 0
    for line in input_file:
        curr_line = line
        # strip line endings
        for i, le in enumerate(line_endings):
            curr_line = curr_line.replace(le, '')

        # split this fucker
        curr_line = curr_line.split(',')
        curr_insert_sql = insert_sql +  '?,'*len(curr_line)
        # remove the last comma
        curr_insert_sql = curr_insert_sql[:-1] + ')'

        #if line_count == 0:
        #    print curr_insert_sql
        db_curr.execute(curr_insert_sql, curr_line)

        line_count += 1
        if line_count % 100000 == 0:
            db_conn.commit()
            print('inserted', '{:,}'.format(line_count), 'records.')

    db_conn.commit()
    print('inserted', '{:,}'.format(line_count), 'records.')

    return None


def pums_csv_to_db(input_file_path, input_file_names, state,
                   dataset_year, drop_db = True):
    """
    Reads in pums data and writes out to a sqlitedb
    :param input_file_path:
    :param input_fileName:
    :return:
    """

    h_recs_f_name, p_recs_f_name = input_file_names

    p_recs_fpn = os.path.join(input_file_path, p_recs_f_name)
    h_recs_fpn = os.path.join(input_file_path, h_recs_f_name)

    p_recs_file = open(p_recs_fpn, 'r')
    h_recs_file = open(h_recs_fpn, 'r')

    # could use either the precs or the hrecs
    db_name = dataset_year + '_' + state + '.db'
    db_path_name = os.path.join(input_file_path, db_name)
    if drop_db:
        if os.path.exists(db_path_name):
            os.remove(db_path_name)

    db_conn = sqlite3.connect(db_path_name)
    db_curr = db_conn.cursor()

    # remove line endings
    line_endings = ['\r', '\r\n', '\n']

    # get the field defintions as text:
    col_names = h_recs_file.readline()
    for i,le in enumerate(line_endings):
        col_names = col_names.replace(le, '')

    col_names = col_names.split(',')

    base_sql = "create table housing ("
    curr_sql = create_table_def_statement(base_sql, col_names )

    db_curr.execute(curr_sql)
    db_conn.commit()

    # insert the housing records
    import_data_to_sqlite(input_file = h_recs_file, db_conn = db_conn,
                          db_curr = db_curr, table_name = 'housing')

    # insert the person records

    # get the field defintions as text:
    col_names = p_recs_file.readline()
    for i,le in enumerate(line_endings):
        col_names = col_names.replace(le, '')

    col_names = col_names.split(',')

    base_sql = "create table person ("
    curr_sql = create_table_def_statement(base_sql, col_names )

    db_curr.execute(curr_sql)
    db_conn.commit()

    # insert the records
    import_data_to_sqlite(input_file = p_recs_file, db_conn = db_conn,
                          db_curr = db_curr, table_name = 'person')

    # close connections
    p_recs_file.close()
    h_recs_file.close()
    db_curr.close()
    db_conn.close()

    return None


if __name__ == '__main__':

    ifp = 'H:/data/acs/acs_pums_2017_1_year'
    ifn = ['psam_h24.csv', 'psam_p24.csv']
    state = 'md'
    dsy = 'pums_2017_1yr'

    pums_csv_to_db(input_file_path = ifp, input_file_names = ifn,
                   state = state, dataset_year = dsy)