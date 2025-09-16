__author__ = 'babbm'

# utility functions for help with importing data to SQLite.

# standard libraries
import os
import sqlite3

# external libraries
import pandas as pd


def delete_db(db_path=None, db_name=None, db_pn=None):
    """
    Delete a sqlite db. Really, this will delete any file.
    :param db_path:
    :param db_name:
    :return:
    """
    if db_pn:
        pass
    else:        
        db_pn = os.path.join(db_path, db_name)

    if os.path.exists(db_pn):
        print('...Deleting:', os.path.normpath(db_pn))
        os.remove(db_pn)
    else:
        print('FILE NOT FOUND')

    return None


def create_table_def_statement(table_name, col_names, data_types=None):
    """
    Programmatically create a table definition SQL statement
    :param table_name:
    :param col_names:
    :param data_types:
    :return:
    """

    if data_types:
        pass
    else:
        data_types = ['TEXT'] * len(col_names)

    base_sql = "create table " + table_name + " ("

    bad_chars = '`~@#$%^&*()=+[]\\{}|;\':\",./<>? '
    for i, tcn in enumerate(col_names):
        # clean up the column names by removing spaces and funny characters
        for bc in bad_chars:
            tcn = tcn.replace(bc, '')

        base_sql += tcn + ' ' + data_types[i] + ', '

    # remove the last comma and space
    base_sql = base_sql[:-2]

    # add the info for the line number and whether the information is bad
    # add the closing parantheses
    base_sql += ', LineNumber INTEGER, ImportSuccess INTEGER);'

    return base_sql


def import_data_to_sqlite(input_file, db_conn, db_curr, table_name, field_delim,
                          n_columns, show_first_line=True, test_lines=False):
    """
    Read a delimited text file and write the values to sqlite.
    :param input_file:
    :param db_conn:
    :param db_curr:
    :param table_name:
    :param field_delim:
    :param n_columns:
    :param show_first_line:
    :param test_lines:
    :return:
    """   

    # remove line endings
    line_endings = ['\r', '\r\n', '\n']

    insert_sql = 'insert into ' + table_name + ' values ('

    line_count = 0
    import_success = 0
    for line in input_file:
        curr_line = line
        # strip line endings
        for i_le, le in enumerate(line_endings):
            curr_line = curr_line.replace(le, '')

        # split this fucker
        curr_line = curr_line.split(field_delim)

        # trim the values by removing spaces at the end and the beginning
        curr_line = [x.strip() for x in curr_line]


        if len(curr_line) == n_columns:
            curr_line.append(line_count)
            curr_line.append(1)
            import_success += 1
        else:
            curr_line = [None] * n_columns
            curr_line.append(line_count)
            curr_line.append(0)

        curr_insert_sql = insert_sql + '?,'*len(curr_line)
        # remove the last comma
        curr_insert_sql = curr_insert_sql[:-1] + ')'

        if show_first_line and line_count == 0:
            print(curr_insert_sql)
            print(curr_line)

        db_curr.execute(curr_insert_sql, curr_line)

        line_count += 1
        if line_count % 100000 == 0:
            db_conn.commit()
            print('...inserted', '{:,}'.format(line_count), 'records to', table_name)

            # call the whole thing off if we're just testing!
            if test_lines:
                break

    # final commit of records.
    db_conn.commit()
    # how did we do?
    print('...inserted', '{:,}'.format(line_count), 'total records to', table_name)

    # summary statistics
    bad_line_count = line_count - import_success
    percent_import_success = import_success / line_count
    summary_stat_list = [table_name, line_count, import_success,
                         bad_line_count, percent_import_success]

    return summary_stat_list


def write_summary_stats(summary_stat_list, db_conn, db_curr):
    """
    Collect the summary stats and write to a database.
    :param summary_stat_list:
    :param db_conn:
    :param db_curr:
    :return:
    """

    # Delete the record for the summary stat table if it already exists.
    table_name = summary_stat_list[0]

    sql = 'select tbl_name from sqlite_master where tbl_name = \'SummaryStats\';'
    tbl_name_df = pd.read_sql(sql=sql, con=db_conn)
    if len(tbl_name_df) > 0:
        # delete the record that matches the table_name
        sql = 'delete from SummaryStats where TableName = \'' + table_name + '\';'
        db_curr.execute(sql)
        db_conn.commit()

    # create a df featuring the import statistics
    # then write them to sqlite
    col_names = ['TableName', 'nRecords', 'ImportSuccess', 'ImportFail', 'PerImportSuccess']
    summary_stat_df = pd.DataFrame(data=[summary_stat_list], columns=col_names)  
    summary_stat_df.to_sql(name='SummaryStats', con=db_conn, if_exists='append', index=False)

    del tbl_name_df
    del summary_stat_df

    return None


def csv_to_db(input_file_path, input_file_name, db_path, db_name,
              table_name, col_names=None, data_types=None,
              field_delim='\t', drop_db=False, show_first_line=False,
              test_lines=False):
    """

    :param input_file_path:
    :param input_file_name:
    :param db_path:
    :param db_name:
    :param table_name:
    :param col_names:
    :param data_types:
    :param field_delim:
    :param drop_db:
    :param show_first_line:
    :return:
    """
    import sqlite3
    print('Now processing', input_file_name, '...')

    input_fpn = os.path.join(input_file_path, input_file_name)

    input_file = open(input_fpn, 'r')

    db_path_name = os.path.join(db_path, db_name)
    if drop_db:
        if os.path.exists(db_path_name):
            os.remove(db_path_name)

    db_conn = sqlite3.connect(db_path_name)
    db_curr = db_conn.cursor()

    if col_names:
        pass
    else:
        # remove line endings
        line_endings = ['\r', '\r\n', '\n']
        # get the field definitions as text:
        col_names = input_file.readline()
        for i_le, le in enumerate(line_endings):
            col_names = col_names.replace(le, '')

        col_names = col_names.split(field_delim)

    drop_sql = 'drop table if exists ' + table_name
    db_curr.execute(drop_sql)
    db_conn.commit()

    # determine the number of columns
    n_columns = len(col_names)

    # build the CREAT TABLE SQL statement
    sql = create_table_def_statement(table_name, col_names, data_types)

    db_curr.execute(sql)
    db_conn.commit()

    # insert the records
    summary_stat_list = import_data_to_sqlite(input_file=input_file, db_conn=db_conn,
                                              db_curr=db_curr, table_name=table_name,
                                              field_delim=field_delim, n_columns=n_columns,
                                              show_first_line=show_first_line,
                                              test_lines = test_lines)

    # write the summary results
    write_summary_stats(summary_stat_list=summary_stat_list, db_conn=db_conn,
                        db_curr=db_curr)

    # close connections
    input_file.close()
    db_curr.close()
    db_conn.close()

    return None


def dict_to_sqlite_table(table_name, values_dict, db_path, db_name,
                         create_sql=None):
    """
    Enumerate keys and values in a dictionary, send to SQLite.
    :param table_name:
    :param values_dict:
    :param db_path:
    :param db_name:
    :param create_sql:
    :return:
    """

    import sqlite3

    db_path_name = os.path.join(db_path, db_name)
    db_conn = sqlite3.connect(db_path_name)
    db_cur = db_conn.cursor()

    # drop table if it exists
    if create_sql:
        drop_sql = 'drop table if exists ' + table_name
        db_cur.execute(drop_sql)
        db_conn.commit()

        # insert statement to create a table
        db_cur.execute(create_sql)
        db_conn.commit()
        # insert values

    # sorted list of keys
    id_values = sorted(values_dict.keys())

    # determine how many values to insert?
    n_values = len(values_dict[id_values[0]])

    # place holders
    ph_values = '?,'*n_values
    # remove the last comma
    ph_values = '?,' + ph_values[:-1]

    base_sql = 'insert into ' + table_name + ' values (' + ph_values + ')'
    for id_value, insert_values in values_dict.items():
        # look up in the dictionary, insert into sql
        # it's the form of key: value
        insert_values.insert(0, id_value)
        insert_values = tuple(insert_values)

        print(base_sql, insert_values)

        db_cur.execute(base_sql, insert_values)

    db_conn.commit()

    db_cur.close()
    db_conn.close()

    return None


def copy_table(i_db_path, i_db_name, o_db_path, o_db_name, table_names):
    """
    Copy table(s) from one sqlitedb to another.

    :param i_db_path:
    :param i_db_name:
    :param o_db_path:
    :param o_db_name:
    :param table_names:
    :return:
    """
    
    import sqlite3

    i_db_pn = os.path.join(i_db_path, i_db_name)
    o_db_pn = os.path.join(o_db_path, o_db_name)

    i_db_conn = sqlite3.connect(i_db_pn)
    i_db_conn.text_factory = str

    o_db_conn = sqlite3.connect(o_db_pn)
    o_db_conn.text_factory = str

    if table_names is None:

        # assume that we want all tables to be copied
        sql = 'SELECT name FROM sqlite_master WHERE type=\'table\';'
        df = pd.read_sql(sql=sql, con=i_db_conn)
        table_names = df['name'].tolist()

    if isinstance(table_names, str):
        table_names = [table_names]

    for i_tcn, tcn in enumerate(table_names):
        print(tcn)
        sql = 'select * from ' + tcn
        df = pd.read_sql(sql=sql, con=i_db_conn)
        df.to_sql(name=tcn, con=o_db_conn, if_exists='replace', index=False)

    i_db_conn.close()
    o_db_conn.close()

    return None


def add_table_indices(db_path, db_name, index_list):
    """
    :param db_path:
    :param db_name:
    :param index_list:
    :return:
    """
    execute_sql_statement(db_path=db_path, db_name=db_name, sql=index_list)

    return None
    

def execute_sql_statement(db_path, db_name, sql):
    """ EXECUTE AN ARBITRARY SQL STATEMENT. Usually a delete statement
    """

    import sqlite3

    # set up the db connection
    db_path_name = os.path.join(db_path, db_name)
    db_conn = sqlite3.connect(db_path_name)

    db_cursor = db_conn.cursor()

    if isinstance(sql, str):
        sql = [sql]

    for i_sql, str_sql in enumerate(sql):
        print('...executing:', str_sql)

        db_cursor.execute(str_sql)

    print('...saving...')
    db_conn.commit()

    db_cursor.close()
    db_conn.close()

    print('...finished.')

    return None


if __name__ == '__main__':

    ifp = '/Census2010/geography/pums'
    ifn = 'PUMA10_PUMA00_cw.txt'
    dbp = '/Census2010/geography/pums'
    dbn = 'puma10_puma00_crosswalk.db'
    tn = 'puma10_puma00_crosswalk'
    col_names_ = None
    data_types_ = ['TEXT', 'TEXT', 'INTEGER', 'INTEGER', 'REAL']

    print(os.path.basename(__file__))

    # csv_to_db(input_file_path=ifp, input_file_name=ifn,
    #           db_path=dbp, db_name=dbn,
    #           table_name=tn, col_names=col_names_,
    #           data_types=data_types_, field_delim='\t',
    #           drop_db=True)


