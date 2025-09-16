# import the ACS data to sqlite. Again. Because that's all that we do.

# standard libraries
import os
import sqlite3

# external libraries
import pandas as pd
import numpy as np

# custom
from df_operations import load_data_from_sqlite, write_data_to_sqlite
from fc_df_spatial import *
from sqlite_operations import delete_db

# so, here's what we need to
# read in all of the table shells.
# read in the estimate data
# read in the MOE data
# read in the geo data - this is the hard one. Because it's fixed width.


def load_data(f_name, seq, ts_path, data_path, geo_df):

    table_name = 't_' + str(seq).zfill(5) + f_name[0]

    seq = 'Seq' + str(seq) + '.xls'
    print('...Now loading:', seq, f_name)

    # get the columns
    seq_fpn = os.path.join(ts_path, seq)
    if os.path.exists(seq_fpn):
        pass
    else:        
        seq_fpn += 'x'
        

    col_name_df = pd.read_excel(io=seq_fpn)

    col_names = col_name_df.columns.tolist()

    # the name of the file
    data_fpn = os.path.join(data_path, f_name)
    data_df = pd.read_csv(filepath_or_buffer=data_fpn,sep=',',header=None,
                          names=col_names,dtype=str)
    #print(data_df.head()) 
    #print(geo_df.head())

    # add the geographic information
    data_df = pd.merge(left=data_df,right=geo_df,how='left')

    # update the col_name_df to include the est or moe
    variable_descriptions = col_name_df.iloc[:1].as_matrix()[0]
    #print(variable_descriptions)
    col_name_df = pd.DataFrame(data=col_names, columns = ['Short_Name'])
    col_name_df['Long_Name'] = variable_descriptions

    col_name_df['record_type'] = f_name[0]

    # remove the index stuff
    remove_values = ['FILEID','FILETYPE','STUSAB','CHARITER',
                     'SEQUENCE','LOGRECNO']
    col_name_df = col_name_df[-col_name_df['Short_Name'].isin(remove_values)]

    col_name_df['table_name'] = table_name

    return [col_name_df, data_df, table_name]


def load_geo_data(geo_path, geo_name):

    geo_pn = os.path.join(geo_path, geo_name)

    geo_df = pd.read_excel(io=geo_pn,dtype=str)

    geo_df['SUM_LEVEL'] = geo_df['GEOID'].str[:3]
    geo_df = geo_df.drop('GEOGRAPHY NAME', 1)

    return geo_df


def call_load_data(data_df, geo_df, db_path, db_name):
    """

    :param data_df:
    :param geo_df:
    :param db_path:
    :param db_name:
    :return:
    """

    db_pn = os.path.join(db_path, db_name)
    db_conn = sqlite3.connect(db_pn)

    f_name_list = data_df['f_name'].tolist()
    seq_list = data_df['seq'].tolist()

    # let's store the record counts
    record_count_list = []

    for i_fn, fn in enumerate(f_name_list):
        seq = seq_list[i_fn]

        col_name_df, data_df, table_name = load_data(f_name=fn, seq=seq,
                                                     ts_path=ts_path,
                                                     data_path=data_path,
                                                     geo_df=geo_df)

        # write col_name_df to sqlite
        write_data_to_sqlite(df=col_name_df,table_name='metadata',
                             db_conn=db_conn,if_exists_option='append',
                             index_option=False,
                             make_col_name_lcase=False)

        write_data_to_sqlite(df=data_df, table_name=table_name,
                             db_conn=db_conn, if_exists_option='replace',
                             index_option=False,
                             make_col_name_lcase=False)

        # something to store the record counts
        curr_list = [table_name, len(data_df)]
        record_count_list.append(curr_list)

    col_names = ['table_name', 'record_count']
    rc_df = pd.DataFrame(data=record_count_list, columns=col_names)
    write_data_to_sqlite(df=rc_df, table_name='record_count',
                         db_conn=db_conn, if_exists_option='replace',
                         index_option=False,
                         make_col_name_lcase=False)

    db_conn.close()
    return None


def gather_file_specs(data_path, ts_path, geo_path, geo_name):
    print('...Gathering file specs')

    data_files = os.listdir(data_path)
    data_files = [i for i in data_files if i[0] == 'e']
    ts_files = os.listdir(ts_path)

    # let's use a pandas df to help with sorting and processing
    data_df = pd.DataFrame(data=data_files, columns=['f_name'])
    ts_df = pd.DataFrame(data=ts_files, columns=['seq_name'])

    # extract the sequence
    data_df['seq'] = data_df['f_name'].str[8:-7].astype(int32)

    # load the geo data
    geo_df = load_geo_data(geo_path, geo_name)

    return data_df, geo_df


def rename_seq_files(file_path):

    seq_files = os.listdir(file_path)
    seq_files = [i for i in seq_files if i[:3] == 'seq']

    for i_sf, sf in enumerate(seq_files):
        seq = sf[3:-5]
        seq = int(seq)
        new_fn = 'Seq' + str(seq) + '.xlsx'
        old_fpn = os.path.join(file_path, sf)
        new_fpn = os.path.join(file_path, new_fn)

        print(old_fpn)
        print(new_fpn)

        os.rename(src=old_fpn, dst=new_fpn)


if __name__ == '__main__':

    #file_path = 'H:/data/acs/acs_sf_2017_5_year/tableshells'
    #rename_seq_files(file_path)


    base_path = 'H:/data/acs/acs_sf_?_5_year'
    db_name = 'acs_sf_?_5_year.db'
    for dy in range(2017, 2018):
        dataset_year = str(dy)
        curr_base_path = base_path.replace('?', dataset_year)
        curr_db_name = db_name.replace('?', dataset_year)

        delete_db(db_path=curr_base_path,db_name=curr_db_name)

        data_path = os.path.join(curr_base_path, 'data')
        ts_path =  os.path.join(curr_base_path, 'tableshells')

        # rename_seq_files(file_path = ts_path)

        db_path = curr_base_path

        geo_path = os.path.join(curr_base_path, 'geo')
        geo_name = 'wa.xls'
        geo_pn = os.path.join(geo_path, geo_name)
        if os.path.exists(geo_pn):
            pass
        else:
            geo_name = 'wa.xlsx'

        data_df, geo_df = gather_file_specs(data_path=data_path, ts_path=ts_path,
                                            geo_path=geo_path, geo_name=geo_name)

        call_load_data(data_df, geo_df, db_path, curr_db_name)