# import the 2000 sf3 data

# standard libraries
import os
import sqlite3

# external
import numpy as np
import pandas as pd
import pyodbc

# custom
from sqlite_operations import load_data_from_sqlite, write_data_to_sqlite
from read_fixed_width_files import read_fixed_width_file
from ms_access_operations import load_data_from_ms_access
from sqlite_operations import execute_sql


def read_summary_file(f_path, f_name, col_names):

    fpn = os.path.join(f_path, f_name)
    print('...now reading in:', fpn)

    df = pd.read_csv(filepath_or_buffer=fpn,sep=',',header=None,names=col_names,
                     dtype=str)

    return df


def gather_column_names(access_db_path, access_db_name):

    base_sql = 'select * from ?;'

    output_dict = {}
    for i in range(1, 77):
        tn = 'SF3' + str(i).zfill(4)
        sql = base_sql.replace('?', tn)
        print('...now executing:', sql)

        df = load_data_from_ms_access(sql = sql, db_path=access_db_path,
                                      db_name=access_db_name)
        col_names = df.columns.tolist()

        output_dict[tn] = col_names

    return output_dict


def call_read_summary_file(access_db_path, access_db_name,
                           input_file_path, geo_file_path,
                           geo_header_file_name,
                           output_db_path, output_db_name):

    # gather the column names
    col_name_dict = gather_column_names(access_db_path=access_db_path,
                                        access_db_name=access_db_name)

    # read in the geo file=
    geo_file_list =  os.listdir(geo_file_path)

    geo_file_list = [i for i in geo_file_list if i != geo_header_file_name]

    geo_header_fpn = os.path.join(geo_file_path, geo_header_file_name)
    geo_header_df = pd.read_csv(filepath_or_buffer=geo_header_fpn, sep='\t', header=None,
                                names=['col_name', 'col_width'])

    geo_header_df['col_width'] = geo_header_df['col_width'] - 1
    geo_col_names = geo_header_df['col_name'].tolist()
    geo_col_width = geo_header_df['col_width'].tolist()

    # load the geo files
    # this is very lazy - but just get it done now.
    geo_file_name = geo_file_list[0]
    geo_df = read_fixed_width_file(input_file_path=geo_file_path,input_file_name=geo_file_name,
                                   column_widths=geo_col_width, new_column_names=geo_col_names,
                                   data_line_check_characters=None)

    geo_df = geo_df.loc[geo_df['SUMLEV'] == '140', :]
    geo_df['TRACT_FIPS'] = geo_df['STATE'] + geo_df['COUNTY'] + geo_df['TRACT']
    col_names = ['LOGRECNO', 'TRACT_FIPS']
    geo_df = geo_df[col_names]


    # load the geo_data
    input_file_list = os.listdir(input_file_path)
    sf_file_list = [i for i in input_file_list if i[:2] == 'wa']


    sf_file_df = pd.DataFrame(data=sf_file_list,columns=['file_name'])
    sf_file_df['seq'] = sf_file_df['file_name'].str[3:7]
    sf_file_df['table_name'] = 'SF3' + sf_file_df['seq']

    sf_file_list = sf_file_df['file_name'].tolist()
    table_name_list = sf_file_df['table_name'].tolist()

    for i_sf, sf in enumerate(sf_file_list):

        tn = table_name_list[i_sf]
        print(sf, tn)
        col_names = col_name_dict[tn]
        sf_df = read_summary_file(f_path=input_file_path, f_name=sf,
                                  col_names=col_names)

        # join the geo data file
        sf_df = pd.merge(left=sf_df, right=geo_df)

        write_data_to_sqlite(df=sf_df,table_name=tn,db_path=output_db_path,
                             db_name=output_db_name,if_exists_option='replace',
                             index_option=False)


def normailze_2000_data(input_db_path, input_db_name,
                        output_db_path, output_db_name):

    input_db_pn = os.path.join(input_db_path, input_db_name)
    input_db_con = sqlite3.connect(input_db_pn)

    output_db_pn = os.path.join(output_db_path, output_db_name)
    output_db_con = sqlite3.connect(output_db_pn)

    sql = 'select tbl_name from sqlite_master'
    table_list_df = load_data_from_sqlite(sql=sql,db_conn=input_db_con)

    print(table_list_df.head())

    drop_column_names = ['FILEID', 'STUSAB', 'CHARITER', 'CIFSN', 'LOGRECNO']

    table_list = table_list_df['tbl_name'].tolist()

    base_sql = 'select * from ?;'
    for i_tn, tn in enumerate(table_list):

        print(tn)

        curr_sql = base_sql.replace('?', tn)
        curr_df = load_data_from_sqlite(sql=curr_sql,db_conn=input_db_con)

        curr_df = curr_df.drop(drop_column_names, 1)

        value_vars = curr_df.columns.tolist()

        value_vars.remove('TRACT_FIPS')

        # melt
        output_df = pd.melt(frame=curr_df,id_vars=['TRACT_FIPS'],
                            value_vars=value_vars,var_name='SHORT_NAME',
                            value_name='VALUE')

        output_df['STATE_FIPS'] = output_df['TRACT_FIPS'].str[:2]
        output_df['COUNTY_FIPS'] = output_df['TRACT_FIPS'].str[:5]

        if i_tn == 0:
            replace_option = 'replace'
        else:
            replace_option = 'append'
        write_data_to_sqlite(df=output_df,table_name='census_2000',
                             db_conn=output_db_con,if_exists_option=replace_option,
                             index_option=False)

    sql = 'CREATE INDEX idx_census_2000_short_name ON census_2000 (SHORT_NAME);'
    execute_sql(sql)

    return None


if __name__ == '__main__':

    # access_db_path = 'H:/data/census_data/census2000'
    # access_db_name = 'SF3.accdb'
    # input_file_path = 'H:/data/census_data/census2000/Washington/sf'
    # geo_file_path = 'H:/data/census_data/census2000/Washington/geo'
    # geo_header_file_name = 'geo_header.txt'
    # output_db_path = 'H:/data/census_data/census2000/Washington'
    # output_db_name = 'sf3.db'
    #
    # call_read_summary_file(access_db_path=access_db_path,
    #                        access_db_name=access_db_name,
    #                        input_file_path=input_file_path,
    #                        geo_file_path=geo_file_path,
    #                        geo_header_file_name=geo_header_file_name,
    #                        output_db_path=output_db_path,
    #                        output_db_name=output_db_name)

    input_db_path = 'H:/data/census_data/census2000/Washington'
    input_db_name = 'sf3.db'
    output_db_path = 'H:/data/census_data/census2000'
    output_db_name = 'sf3.db'

    normailze_2000_data(input_db_path=input_db_path, input_db_name=input_db_name,
                        output_db_path=output_db_path, output_db_name=output_db_name)