# standard libraries
import os
import sqlite3

# external libraries
import pandas as pd
import numpy as np

# custom
from df_operations import load_data_from_sqlite


def load_acs_data(short_name_list, new_col_names=None, county_fips=None,
                  state_fips=None, dataset_year=None,
                  acs_db_path='H:/data/acs', acs_db_name='acs_sf.db'):
    """
    Query the ACS database to get some data.
    :param acs_db_path:
    :param acs_db_name:
    :param short_name_list:
    :param new_col_names:
    :return:
    """

    acs_db_pn = os.path.join(acs_db_path, acs_db_name)
    acs_db_conn = sqlite3.connect(acs_db_pn)

    # sql to use to gather data
    base_sql = '''select
    DATASET_YEAR,
    TRACT_FIPS,
    SHORT_NAME,
    VALUE
    from acs_est
    where SHORT_NAME = \'?\'
    '''
    # filter by year
    if dataset_year:
        base_sql += ' and DATASET_YEAR = \'' + dataset_year + '\''

    # filter by state
    if state_fips:
        base_sql += ' and STATE_FIPS = \'' + state_fips + '\''

    # filter by county
    if county_fips:
        base_sql += ' and COUNTY_FIPS = \'' + county_fips + '\''

    # add the end SQL statement character
    base_sql += ';'

    # to hold the output
    df_list = []
    for cn in short_name_list:
        curr_sql = base_sql.replace('?', cn)

        # load the data into memory
        df = load_data_from_sqlite(
            sql=curr_sql, db_conn=acs_db_conn, verbose=False,
            make_col_name_lcase=False)
        # create the geography and year identifier
        df['T_Y'] = df['TRACT_FIPS'] + '_' + df['DATASET_YEAR']

        df_list.append(df)

    # create the output df
    df = pd.concat(df_list)
    del df_list

    # pivot
    df = df.pivot(index='T_Y', columns='SHORT_NAME', values='VALUE')

    # apply the new column names
    if new_col_names:
        df.columns = new_col_names
    # convert to numeric values
    df = df.astype(int32)

    # extract the data column names
    data_col_names = df.columns.tolist()

    # re-index
    df = df.reset_index()

    # split out the tract fips and the year
    df['TRACT_FIPS'] = df['T_Y'].str[:11]
    df['DATASET_YEAR'] = df['T_Y'].str[12:]

    # re-order the columns
    col_names = ['DATASET_YEAR', 'TRACT_FIPS']
    col_names.extend(data_col_names)
    df = df[col_names]

    # convert to lowercase
    col_names = df.columns.tolist()
    col_names = [i.lower() for i in col_names]
    df.columns = col_names

    # return the df
    return df


def call_load_ACS_data_by_group(short_name_list, new_col_names,
                                group_dict, acs_db_path='H:/data/acs',
                                acs_db_name='acs_sf.db'):
    """
    Wrapper function to load many acs fields.
    For use when looking at different groups
    :param acs_db_path:
    :param acs_db_name:
    :param short_name_list:
    :param new_col_names:
    :param group_dict:
    :return:
    """

    output_df_list = []
    for group_key, group_val in group_dict.items():
        print(group_key, group_val)

        curr_short_name_list = [i.replace('?', group_key)
                                for i in short_name_list]

        df = load_acs_data(acs_db_path=acs_db_path, acs_db_name=acs_db_name,
                           short_name_list=curr_short_name_list,
                           new_col_names=new_col_names)

        # reorganize the field names
        df['group_val'] = group_val

        col_names = df.columns.tolist()
        id_col_names = col_names[:2]
        data_col_names = col_names[2:-1]
        id_col_names.append('group_val')

        id_col_names.extend(data_col_names)

        df = df[id_col_names]

        output_df_list.append(df)

    df = pd.concat(output_df_list)
    del output_df_list

    return df


if __name__ == '__main__':

    acs_db_path = 'H:/data/acs'
    acs_db_name = 'acs_sf.db'

    short_name_list = ['b07003_001', 'b07003_004', 'b07003_007', 'b07003_010',
                       'b07003_013', 'b07003_016']

    df = load_acs_data(acs_db_path=acs_db_path, acs_db_name=acs_db_name,
                       short_name_list=short_name_list)

    print(df.head())
