# creates a dictionary of dictionaries from pandas dataframe.
# each row/field is a key value pair in the inner dictionary.

# standard libraries
import inspect
import os
import sqlite3
from time import perf_counter_ns

# external libraries
import numpy as np
import pandas as pd


# custom
from base_functions import get_date_time_stamp


def build_data_log_table_entry(f_path, f_name, table_name, data_type, description,
                               n_records, n_columns, script_name):
    # get the time stamp - this will happen after the table is written
    last_update = get_date_time_stamp()
    
    source_platform = 'python'

    # create our dataframe
    lt_data = [[f_path, f_name, table_name, data_type, description, n_records,
                n_columns, script_name, source_platform, last_update]]
    col_names = ['file_path', 'file_name', 'table_name', 'data_type', 'description',
                 'n_records', 'n_columns', 'source_script_name', 'source_platform', 'last_update']
    lt_data = pd.DataFrame(data=lt_data, columns=col_names)

    return(lt_data)


def write_data(obj_to_store, file_path, file_name, table_name='',
               description='', data_export_log_db_path='H:/project/data_export_log',
               data_export_log_db_name='data_export_log.db', if_exists_option =  'replace'):
    # THIS IS A WRAPPER FUNCTION THAT WRITES DATAFRAMES TO DISK.
    # UPON WRITING, A LOG ENTRY IS MADE LISTING WHEN THE DATA/PLOT WAS MADE, WHERE IT WAS STORED,
    # AND THE SCRIPT THAT GENERATED THE DATA

    # get the full file path
    file_path = os.path.normpath(os.path.abspath(file_path))

    # extract the file extension
    file_info = os.path.splitext(file_name)
    data_type = file_info[1]

    # get the name of the script that created the data
    #script_name = os.path.basename(__file__)
    script_name = os.path.realpath(inspect.stack()[-1][1])
    
    # build the fully qualified file path for everything but the database
    if data_type != '.db':
        file_path_name = os.path.join(file_path, file_name)

    n_records = len(obj_to_store)
    n_columns = len(obj_to_store.columns.tolist())

    lt_data = build_data_log_table_entry(f_path=file_path,
                                         f_name=file_name,
                                         table_name=table_name,
                                         data_type=data_type,
                                         description=description,
                                         n_records=n_records,
                                         n_columns=n_columns,
                                         script_name=script_name)                                             

    # add the last update field
    obj_to_store = obj_to_store.copy()
    obj_to_store['last_update'] = lt_data['last_update'].iloc[0]        

    if data_type == '.db':
        # 1. sqlite database table
        # this writes the table of interest to sqlite        
        write_data_to_sqlite(df=obj_to_store, table_name=table_name,
                             db_path=file_path, db_name=file_name,
                             if_exists_option=if_exists_option)       

    if data_type in ('.xls', '.xlsx'):
        # 2. excel
        obj_to_store.to_excel(excel_writer=file_path_name, index=False)

    if data_type in ('.txt', '.csv'):
        # 3. csv/txt
        obj_to_store.to_csv(path_or_buf=file_path_name, sep='\t', index=False)    

    # write to the db of interest if were storing a database
    if data_type == '.db':        
        write_data_to_sqlite(df=lt_data, table_name='_log_table',
                         db_path=file_path,
                         db_name=file_name,
                         verbose=False,
                         if_exists_option='append')


    # write to the "global" data export log
    write_data_to_sqlite(df=lt_data, table_name='data_export_log',
                         db_path=data_export_log_db_path,
                         db_name=data_export_log_db_name,
                         verbose=False,
                         if_exists_option='append')
    
    return None


def write_data_to_sqlite(df, table_name, db_path=None, db_name=None,
                         db_conn=None, if_exists_option='replace',
                         index_option=False, verbose=True,
                         make_col_name_lcase=True):
    """
    A wrapper function to help with writing a pandas df to SQLite
    :param df:
    :param table_name:
    :param db_path:
    :param db_name:
    :param db_conn:
    :param if_exists_option:
    :param index_option:
    :return:
    """
    import sqlite3

    # check to see if a live connection was passed.
    if db_conn:
        close_connection = False
    else:
        close_connection = True
        db_pn = os.path.join(db_path, db_name)
        db_conn = sqlite3.connect(db_pn)

    if verbose:
        print('...now writing:', table_name)

    if make_col_name_lcase:
        df = change_column_name_case(df=df)

    # write the table of interest
    df.to_sql(name=table_name, con=db_conn, if_exists=if_exists_option,
              index=index_option)

    if close_connection:
        db_conn.close()

    return None


def load_data_from_sqlite(sql, db_path=None, db_name=None,
                          db_conn=None, verbose=True,
                          make_col_name_lcase=True):
    """
    Loads data from a sqlite. Can either build the database connection
    or accept a live connection.
    :param sql:
    :param db_path:
    :param db_name:
    :param db_conn:
    :return:
    """
    time_start = perf_counter_ns()

    if verbose:
        print('...EXECUTING:', sql)

    # check to see if a live connection was passed.
    if db_conn:
        close_connection = False
    else:
        close_connection = True
        db_pn = os.path.join(db_path, db_name)
        db_conn = sqlite3.connect(db_pn)

    df = pd.read_sql(sql=sql, con=db_conn)

    if close_connection:
        db_conn.close()

    if make_col_name_lcase:
        df = change_column_name_case(df=df)

    if 'last_update' in df.columns.tolist():
        df = df.drop(labels = 'last_update', axis = 1)

    time_end = perf_counter_ns()
    time_proc = (time_end - time_start) / 1e9    
    if verbose:        
        df_rows, df_cols = df.shape
        display_string = f'...LOADED | {df_rows:,} rows | {df_cols:,} columns | in {round(time_proc,4)} seconds...'          
        print(display_string)

    return df


def into_dict(df, key_fn, value_fns=None, with_value_field_names=True):
    """
    return a dictionary of dictionaries from a pandas data frame
    :param df:
    :param key_fn:
    :param value_fns:
    :param with_value_field_names:
    :return:
    """

    temp_dict = {}

    if type(value_fns) is str:
        value_fns = [value_fns]
    else:
        value_fns = df.columns.tolist()
        value_fns.remove(key_fn)

    def put_it(row):
        curr_dict = {}

        key = row[key_fn]

        if with_value_field_names:
            for ivfn, vfn in enumerate(value_fns):
                curr_dict[vfn] = row[vfn]

            temp_dict[key] = curr_dict
        else:
            temp_dict[key] = row[value_fns[0]]

        return None

    # put the items into a dictionary
    df.apply(put_it, 1)

    return temp_dict


def extract_year(x):
    return x.year


def extract_month(x):
    return x.month


def extract_day(x):
    return x.day


def combine_year_month(row, year_col_name, month_col_name):
    curr_year = row[year_col_name]
    curr_year = str(curr_year)
    curr_month = row[month_col_name]
    curr_month = str(curr_month).zfill(2)

    curr_year_month = curr_year + '_' + curr_month

    return curr_year_month


def get_date_parts(df, date_col_name, get_year=True, get_month=True,
                   get_day=False, year_col_name='ext_year',
                   month_col_name='ext_month', day_col_name='ext_day',
                   year_month_col_name='year_month'):
    """
    Extract parts of the date into separate columns
    :param df:
    :param date_col_name:
    :param get_year:
    :param get_month:
    :param get_day:
    :param year_col_name:
    :param month_col_name:
    :param day_col_name:
    :param year_month_col_name:
    :return:
    """

    # ensure that we are working with a date

    if get_year:
        df[year_col_name] = df[date_col_name].map(extract_year)

    if get_month:
        df[month_col_name] = df[date_col_name].map(extract_month)

    if get_day:
        df[day_col_name] = df[date_col_name].map(extract_day)

    if year_month_col_name != '':
        df[year_month_col_name] = df.apply(func=combine_year_month, axis=1,
                                           args=(year_col_name,
                                                 month_col_name))

    return df


def cut_series(df, field_name, cut_type, cut_labels=None):
    """
    Cut a field in a dataframe
    :param df:
    :param cut_field_name:
    :param cut_type:
    :return:
    """

    if cut_type == 'quartile':
        cut_breaks = [0, .25, .5, .75, 1]
        cut_labels = ['1', '2', '3', '4']
    elif cut_type == 'quintile':
        cut_breaks = [0, .20, .4, .6, .8, 1]
        cut_labels = ['1', '2', '3', '4', '5']
    elif cut_type == 'decile':
        cut_breaks = [0, .1, .20, .3, .4, .5, .6, .7, .8, .9, 1]
        cut_labels = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10']
    else:
        cut_breaks = [0.5]

    quantiles = df[field_name].quantile(cut_breaks).tolist()

    cut_series = pd.cut(x=df[field_name], bins=quantiles, right=True,
                        labels=cut_labels).astype(str)

    new_field_name = field_name + '_cut'
    df[new_field_name] = cut_series

    return df


def change_column_name_case(df, case_option='lower'):
    """
    Simple function to make column names lower case. Or upper case.
    :param df:
    :param case_option:
    :return:
    """
    col_names = df.columns.tolist()

    if case_option == 'upper':
        col_names = [str(i).upper() for i in col_names]
    else:
        col_names = [str(i).lower() for i in col_names]

    df.columns = col_names

    return df


def set_comparison(a_keys, b_keys, return_option=None, verbose=True,
                   output_file_path_name=None, preamble=None, write_option='a'):
    
    

    if output_file_path_name:
        output_file = open(output_file_path_name, write_option)

    if output_file_path_name and preamble:
        output_file.write(preamble + '\n')

    len_a_keys = str(len(a_keys))
    len_b_keys = str(len(b_keys))

    if verbose:
        print('There are', len_a_keys, 'keys in set/dataframe A.')
        print('There are', len_b_keys, 'keys in set/dataframe B.')

    if output_file_path_name:
        write_line = 'There are ' + len_a_keys + ' keys in set/dataframe A.'
        output_file.write(write_line + '\n')
        write_line = 'There are ' + len_b_keys + ' keys in set/dataframe B.'
        output_file.write(write_line + '\n')

    int_keys = a_keys.intersection(b_keys)
    len_int_keys = str(len(int_keys))

    if verbose:
        print('There are', len_int_keys,
              'shared keys between the two sets/dataframes.')

    if output_file_path_name:
        write_line = 'There are ' + len_int_keys + \
            ' shared keys between the two sets/dataframes.'
        output_file.write(write_line + '\n')

    a_diff_b = a_keys.difference(b_keys)
    len_a_diff_b = str(len(a_diff_b))

    if verbose:
        print('There are', len_a_diff_b,
              'keys in set/dataframe A not in set/dataframe B.')

    if output_file_path_name:
        write_line = 'There are ' + len_a_diff_b + \
            ' keys in set/dataframe A not in set/dataframe B.'
        output_file.write(write_line + '\n')

    b_diff_a = b_keys.difference(a_keys)
    len_b_diff_a = str(len(b_diff_a))

    if verbose:
        print('There are', len_b_diff_a,
              'keys in set/dataframe B not in set/dataframe A.')

    if output_file_path_name:
        write_line = 'There are ' + len_b_diff_a + \
            ' keys in set/dataframe B not in set/dataframe A.'
        output_file.write(write_line + '\n')

    if return_option:
        output = [a_keys, b_keys, int_keys, a_diff_b, b_diff_a]
    else:
        output = None

    if output_file_path_name:
        output_file.close()

    return output


def compare_df_keys(a_df, b_df, a_key, b_key, return_option=None,
                    verbose=True, output_file_path_name=None, preamble=None,
                    write_option='a'):
    """
    Function describing the commonality of the keys between two dataframes.
    There is an option to return the outcomes of the membership checks.
    """

    if verbose:
        print('Dataframe A has', len(a_df), 'records.')
        print('Dataframe B has', len(b_df), 'records.')

    if output_file_path_name and preamble:
        output_file = open(output_file_path_name, write_option)
        output_file.write(preamble + '\n')
        output_file.close()
        write_option = 'a'

    if type(a_key) is not list:
        a_key = [a_key]
    if type(b_key) is not list:
        b_key = [b_key]

    for ak, bk in zip(a_key, b_key):

        if output_file_path_name:
            output_file = open(output_file_path_name, write_option)
            write_line = 'a key: ' + ak
            output_file.write(write_line + '\n')
            write_line = 'b key: ' + bk
            output_file.write(write_line + '\n')
            output_file.close()
            preamble = None

        a_keys = set(a_df[ak].unique().tolist())
        b_keys = set(b_df[bk].unique().tolist())

        if verbose:
            print('current keys:', ak, bk)
        output = set_comparison(a_keys=a_keys, b_keys=b_keys,
                                return_option=return_option,
                                verbose=verbose,
                                output_file_path_name=output_file_path_name,
                                preamble=preamble,
                                write_option=write_option)

    return output


def compare_df_columns_names(a_df, b_df, return_option=True):
    """ Compare two dataframe to see if they have the same columns    """

    a_col_names = set(a_df.columns.tolist())
    b_col_names = set(b_df.columns.tolist())

    output = set_comparison(a_keys=a_col_names, b_keys=b_col_names,
                            return_option=True, verbose=True)

    return output


def aggregate_df(df, col_names, num_agg_fields=-1, with_reset=True):

    df_agg = df[col_names].groupby(col_names[:num_agg_fields]).agg("sum")

    if with_reset:
        df_agg = df_agg.reset_index()

    return df_agg


def export_sql_to_excel(sql, db_path, db_name, sheet_name, output_path,
                        output_name):
    """ EXPORT A SQL STATEMENT TO AN EXCEL FILE
    """

    df = load_data_from_sqlite(sql=sql, db_path=db_path, db_name=db_name)

    ofpn = os.path.join(output_path, output_name)
    df.to_excel(excel_writer=ofpn, sheet_name=sheet_name, index=False)

    return None

# calculate an r-squared value
def calc_r_squared(observed, estimated):
    import numpy as np
    r = np.corrcoef(observed, estimated)[0][1]
    r2 = r ** 2
    return r2

# calculate the root mean square error
def calc_RMSE(observed, estimated):
    import numpy as np
    res = (observed - estimated) ** 2
    rmse = round(np.sqrt(np.mean(res)), 3)

    return rmse


if __name__ == '__main__':

    db_path = 'h:/temp'
    db_name = 'testo.db'
    testo = {'a': np.random.randint(0, 100, 100),
             'b': np.random.randint(0, 1000, 100)}
    testo = pd.DataFrame(data=testo)
    write_data_to_sqlite(df=testo, table_name='testo',
                         db_path=db_path, db_name=db_name)
