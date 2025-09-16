# mike babb

# normalize the acs data.
# is this really worth while?
# how does one stop shopping work?
# how should it work?

# standard libraries
import os
import sqlite3

# external
import pandas

# custom
from fc_df_spatial import *
from sqlite_operations import delete_db, add_table_indices
from df_operations import load_data_from_sqlite, write_data_to_sqlite


def normalize_09_11_acs(out_db_path, out_db_name,
                        db_path, db_name, data_year,
                        test_buildout=False):
    """
    Convert the wide ACS tables to long format.
    :param out_db_path:
    :param out_db_name:
    :param db_path:
    :param db_name:
    :return:
    """

    # the destination database
    out_db_pn = os.path.join(out_db_path, out_db_name)
    out_db_conn = sqlite3.connect(out_db_pn)

    # the input database
    db_pn = os.path.join(db_path, db_name)
    db_conn = sqlite3.connect(db_pn)

    # gather the list of table names to import
    sql = 'select table_name from record_count where record_count > 0'
    df = load_data_from_sqlite(
        sql=sql, db_conn=db_conn, make_col_name_lcase=False)
    table_name_list = df['table_name'].tolist()

    # id_columns
    drop_columns = ['FILEID', 'FILETYPE', 'STUSAB', 'CHARITER', 'SEQUENCE', 'LOGRECNO',
                    'STATE', 'SUM_LEVEL', 'GEOID']

    # metadata columns - so we can go back to the original data
    metadata_col_names = ['FILEID', 'FILETYPE', 'SEQUENCE']

    # output variable order
    output_col_names = ['dataset_year', 'state_fips',
                        'county_fips', 'tract_fips', 'short_name', 'value']

    # base sql
    base_sql = 'select * from ? where SUM_LEVEL = \'140\';'

    # list to hold all of the metadata
    metadata_df_list = []

    # enumerate the tables
    for i_tn, tn in enumerate(table_name_list):
        # are we working with est or moes?
        if tn[-1] == 'e':
            output_table_name = 'acs_est'
        else:
            output_table_name = 'acs_moe'

        # extract the acs data from a previously
        # created acs database.
        sql = base_sql.replace('?', tn)
        df = load_data_from_sqlite(sql=sql, db_conn=db_conn,
                                   make_col_name_lcase=False)

        # check to see if there are variables just for Puerto Rico - drop those.
        col_names = df.columns.tolist()
        drop_col_names = [i for i in col_names if 'PR' in i]
        if drop_col_names:
            # print(drop_col_names)
            df = df.drop(drop_col_names, 1)

        # extract the metadata for this table
        # the file identification and the sequences.
        # print(metadata_col_names)
        # print(df.head())
        metadata_df = df[metadata_col_names].drop_duplicates()

        # extract the tract fips
        df['tract_fips'] = df['GEOID'].str[-11:]
        # drop extraneous columns
        df = df.drop(drop_columns, 1)

        # Melt the dataframe from wide to long
        value_col_name = df.columns.tolist()
        value_col_name.remove('tract_fips')
        df = pd.melt(frame=df, id_vars=['tract_fips'], value_vars=value_col_name,
                     var_name='short_name', value_name='value')

        # only keep the first 1000 rows if testing
        if test_buildout:
            df = df.iloc[:100]

        # add the state and county fips
        df['county_fips'] = df['tract_fips'].str[:5]
        df['state_fips'] = df['tract_fips'].str[:2]

        # add the dataset year
        df['dataset_year'] = str(data_year)

        # rearrange the columns
        df = df[output_col_names]
        # convert all column names to uppercase
        col_names = df.columns.tolist()
        col_names = [str(i).upper() for i in col_names]
        df.columns = col_names

        # write the data df to sqlite
        write_data_to_sqlite(df=df, table_name=output_table_name,
                             db_conn=out_db_conn, if_exists_option='append',
                             index_option=False)

        # build out the metadata df
        col_name_df = pd.DataFrame(data=value_col_name, columns=['short_name'])
        col_name_df = pd.concat([col_name_df, metadata_df], axis=1)
        col_name_df = col_name_df.fillna(method='pad')
        col_name_df['table_name'] = tn
        metadata_df_list.append(col_name_df)

        del col_name_df
        del df

    # write the metadata
    metadata_df = pd.concat(metadata_df_list)
    del metadata_df_list

    col_names = metadata_df.columns.tolist()
    col_names = [str(i).upper() for i in col_names]
    metadata_df.columns = col_names

    # read in the long metadata
    sql = 'select * from metadata'
    long_md_df = load_data_from_sqlite(sql=sql, db_conn=db_conn)
    # make names upper case
    col_names = long_md_df.columns.tolist()
    col_names = [str(i).upper() for i in col_names]
    long_md_df.columns = col_names

    metadata_df = pd.merge(left=metadata_df, right=long_md_df)
    del long_md_df

    # rearrange the columns
    col_names = ['FILEID', 'FILETYPE', 'SEQUENCE', 'SHORT_NAME', 'LONG_NAME']
    metadata_df = metadata_df[col_names]

    write_data_to_sqlite(df=metadata_df, table_name='metadata',
                         db_conn=out_db_conn, if_exists_option='append',
                         index_option=False)

    # close the connection
    out_db_conn.close()

    return None


def normalize_column_names(cn):

    if cn != 'tract_fips':

        if 'e' in cn:
            new_cn = cn.replace('e', '_')
        if 'm' in cn:
            new_cn = cn.replace('m', '_')

        # underscore position
        us_pos = new_cn.find('_') + 1
        # split
        new_cn = new_cn[:us_pos] + new_cn[us_pos:].zfill(3)

    else:
        new_cn = cn

    return new_cn


def uppercase_list(curr_list):

    new_list = [str(i).upper() for i in curr_list]

    return new_list


def normalize_12_16_acs(out_db_path, out_db_name,
                        file_path, data_year,
                        test_buildout=False):
    """
    Convert the wide ACS tables to long format.
    :param out_db_path:
    :param out_db_name:
    :param db_path:
    :param db_name:
    :return:
    """

    # TODO: Figure out if this is working with an export of data from a personal geodatabase

    data_year = str(data_year)

    # the destination database
    out_db_pn = os.path.join(out_db_path, out_db_name)
    out_db_conn = sqlite3.connect(out_db_pn)

    # the input folder
    table_name_list = os.listdir(file_path)
    table_name_list = [i for i in table_name_list if i[0] == 'X']
    table_name_list = ['X07_MIGRATION.txt']

    # metdata file
    metadata_file_name = 'TRACT_METADATA_' + data_year + '.txt'
    metadata_fpn = os.path.join(file_path, metadata_file_name)
    l_md_df = pd.read_csv(filepath_or_buffer=metadata_fpn,
                          sep='\t', dtype=str)
    # rename the columns
    l_md_df.columns = ['long_name', 'short_name']

    # gather the list of table names to import
    for i_tn, tn in enumerate(table_name_list):

        print('...now reading:', tn, data_year)
        tn_pn = os.path.join(file_path, tn)

        # read in the data
        df = pd.read_csv(filepath_or_buffer=tn_pn, sep='\t',
                         header=0, dtype=str)

        # drop the columns that pertain to Puerto Rico.
        col_names = df.columns.tolist()

        # create the tract_fips
        df['tract_fips'] = df['GEOID'].str[-11:]
        df = df.drop('GEOID', 1)

        drop_col_names = [i for i in col_names if 'PR' in i]
        if drop_col_names:
            print(drop_col_names)
            df = df.drop(drop_col_names, 1)

        # the est and moe columns
        col_names = df.columns.tolist()
        est_col_names = [i for i in col_names if 'e' in i or i == 'tract_fips']
        moe_col_names = [i for i in col_names if 'm' in i or i == 'tract_fips']

        est_df = df.loc[:, est_col_names]
        moe_df = df.loc[:, moe_col_names]

        del df

        est_col_names.remove('tract_fips')
        moe_col_names.remove('tract_fips')

        # melt
        est_df = pd.melt(frame=est_df, id_vars=['tract_fips'], value_vars=est_col_names,
                         var_name='short_name', value_name='value')

        moe_df = pd.melt(frame=moe_df, id_vars=['tract_fips'], value_vars=moe_col_names,
                         var_name='short_name', value_name='value')

        est_df['county_fips'] = est_df['tract_fips'].str[:5]
        est_df['state_fips'] = est_df['tract_fips'].str[:2]

        moe_df['county_fips'] = moe_df['tract_fips'].str[:5]
        moe_df['state_fips'] = moe_df['tract_fips'].str[:2]

        est_df['dataset_year'] = data_year
        moe_df['dataset_year'] = data_year

        # output variable order
        output_col_names = ['dataset_year', 'state_fips',
                            'county_fips', 'tract_fips', 'short_name', 'value']
        est_df = est_df[output_col_names]
        moe_df = moe_df[output_col_names]

        # metadata columns - so we can go back to the original data
        metadata_col_names = ['FILEID', 'FILETYPE', 'SEQUENCE']

        metadata_est_df = pd.DataFrame(data=est_df['short_name'].unique().tolist(),
                                       columns=['short_name'])
        metadata_moe_df = pd.DataFrame(data=moe_df['short_name'].unique().tolist(),
                                       columns=['short_name'])

        if test_buildout:
            est_df = est_df.loc[:1000, :]
            moe_df = moe_df.loc[:1000, :]

        metadata_est_df['FILEID'] = 'ACSSF'
        metadata_moe_df['FILEID'] = 'ACSSF'

        metadata_est_df['FILETYPE'] = data_year + 'e5'
        metadata_moe_df['FILETYPE'] = data_year + 'm5'

        metadata_est_df['SEQUENCE'] = tn
        metadata_moe_df['SEQUENCE'] = tn

        df_list = [metadata_est_df, metadata_moe_df]
        s_md_df = pd.concat(df_list)
        del metadata_est_df
        del metadata_moe_df
        del df_list

        # join the metadata together
        md_df = pd.merge(left=s_md_df, right=l_md_df)
        del s_md_df

        # convert all column names to uppercase
        col_names = est_df.columns.tolist()
        col_names = uppercase_list(col_names)
        est_df.columns = col_names

        # write the estimate data df to sqlite
        est_df.columns = col_names
        est_df['SHORT_NAME'] = est_df['SHORT_NAME'].map(normalize_column_names)
        write_data_to_sqlite(df=est_df, table_name='acs_est',
                             db_conn=out_db_conn, if_exists_option='append',
                             index_option=False)

        col_names = moe_df.columns.tolist()
        col_names = uppercase_list(col_names)
        moe_df.columns = col_names
        moe_df['SHORT_NAME'] = moe_df['SHORT_NAME'].map(normalize_column_names)
        # write_data_to_sqlite(df=moe_df, table_name='acs_moe',
        #                      db_conn=out_db_conn, if_exists_option='append',
        #                      index_option=False)

        col_names = md_df.columns.tolist()
        col_names = uppercase_list(col_names)
        md_df.columns = col_names

        # rearrange the columns
        col_names = ['FILEID', 'FILETYPE',
                     'SEQUENCE', 'SHORT_NAME', 'LONG_NAME']
        md_df = md_df[col_names]

        md_df['SHORT_NAME'] = md_df['SHORT_NAME'].map(normalize_column_names)

        write_data_to_sqlite(df=md_df, table_name='metadata',
                             db_conn=out_db_conn, if_exists_option='append',
                             index_option=False)

    # close the connection
    out_db_conn.close()


def create_indices(db_path, db_name):
    # add the table indices on the following fields:
    # dataset_year, state_fips, county_fips, short_name

    index_list = [
        'CREATE INDEX IF NOT EXISTS idx_acs_est_dataset_year ON acs_est (DATASET_YEAR);',
        'CREATE INDEX IF NOT EXISTS idx_acs_est_state_fips ON acs_est (STATE_FIPS);',
        'CREATE INDEX IF NOT EXISTS idx_acs_est_county_fips ON acs_est (COUNTY_FIPS);',
        'CREATE INDEX IF NOT EXISTS idx_acs_est_short_name ON acs_est (SHORT_NAME);'
    ]

    add_table_indices(db_path=db_path, db_name=db_name, index_list=index_list)

    return None


if __name__ == '__main__':
    out_db_path = 'H:/data/acs'
    out_db_name = 'acs_sf.db'

    # delete that bad boy.
    # delete_db(db_path=out_db_path, db_name=out_db_name)

    ####
    # 2009 - 2017
    ####
    db_path = 'H:/data/acs/acs_sf_?_5_year'
    db_name = 'acs_sf_?_5_year.db'

    for dy in range(2017, 2018):

        data_year = str(dy)
        curr_db_path = db_path.replace('?', data_year)
        curr_db_name = db_name.replace('?', data_year)

        normalize_09_11_acs(out_db_path=out_db_path,
                            out_db_name=out_db_name,
                            db_path=curr_db_path, db_name=curr_db_name,
                            data_year=data_year,
                            test_buildout=False)

    #
    create_indices(db_path=out_db_path, db_name=out_db_name)

    # ####
    # # 2012 - 2016
    # ####
    # file_path = 'H:/data/acs/acs_sf_?_5_year'
    #
    # for dy in range(2012, 2017):
    #     data_year = str(dy)
    #     curr_file_path = file_path.replace('?', data_year)
    #
    #     normalize_12_16_acs(out_db_path=out_db_path,
    #                         out_db_name=out_db_name,
    #                         file_path=curr_file_path,
    #                         data_year=data_year,
    #                         test_buildout=False)
