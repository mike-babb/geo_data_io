####
# CREATE A SQLITEDB WITH FROM ANY TIME OF GEOGRAPHY
####

# standard libraries
import os
import sqlite3
from multiprocessing import Pool

# external libraries
import pandas as pd
import geopandas as gpd

# custom libraries
from base_functions import *
from fc_df_spatial import unpack_bbox_values, get_envelope_area, get_wkt, load_geo_data_from_sqlite, create_centroids
from df_operations import change_column_name_case, write_data_to_sqlite

def load_blocks(f_values):
    """
    Loads the blocks into memory
    :param f_values:
    :return:
    """

    s_file_path_name, convert_cs = f_values

    # read in the file as a geodataframe
    gdf = gpd.read_file(s_file_path_name)
    n_features = '{:,}'.format(len(gdf))
    print('...loaded', n_features, 'features for',
          os.path.normpath(s_file_path_name))

    # let's convert FROM NAD 1983 TO WGS 84
    if convert_cs:
        wgs84_crs = {'init': 'epsg:4326'}
        gdf = gdf.to_crs(wgs84_crs)
        print('...finished re-projecting.')
    
    # calculate the area of the geometry    
    gdf['xy_area'] = gdf['geometry'].map(get_envelope_area)

    # calculate the wkt        
    gdf['wkt'] = gdf['geometry'].map(get_wkt)

    gdf = unpack_bbox_values(gdf=gdf)

    # drop the geometry
    gdf = gdf.drop('geometry', 1)

    # set column names to uppercase
    # maybe do lowercase in the future?
    gdf = change_column_name_case(df=gdf, case_option='lower')
    
    return gdf


def add_county_fips(gdf, state_county_fn):

    # get the county fips
    state_fn, county_fn = state_county_fn
    gdf['countyfips'] = gdf[state_fn] + gdf[county_fn]

    return gdf


def add_cbsa_fips(gdf, cbsa_db_path_name):

    cbsa_db_conn = sqlite3.connect(cbsa_db_path_name)

    # load the county to cbsa cross walk
    sql = 'select countyfips, cbsa_code from county_cbsa_crosswalk'
    county_cbsa_df = pd.read_sql(sql=sql, con=cbsa_db_conn)
    col_names = ['countyfips', 'cbsa_geoid']
    county_cbsa_df.columns = col_names

    # join in the cbsa df
    gdf = pd.merge(left=gdf, right=county_cbsa_df, how='left')

    return gdf


def add_table_indices(db_path, db_name, index_list):

    # set up the db connection
    db_path_name = os.path.join(db_path, db_name)
    db_conn = sqlite3.connect(db_path_name)

    db_cursor = db_conn.cursor()

    for i_sql, sql in enumerate(index_list):
        print('...executing:', sql)

        db_cursor.execute(sql)

    print('saving indices')
    db_conn.commit()

    db_cursor.close()
    db_conn.close()

    print('finished.')


def enumerate_s_file_directory(s_file_path, db_path, db_name,
                               table_name, unique_id_fn=None,
                               state_county_fn=None, cbsa_db_path_name=None,
                               index_list=None):
    """
    Enumerate a directory with shapefiles.
    Import the shapefiles a few batches at a time to a sqlite db
    :param s_file_path:
    :param output_path:
    :param output_db_name:
    :param output_table_name:
    :return:
    """

    # working path
    s_file_names = os.listdir(s_file_path)
    # only get shape files.
    s_file_names = [i for i in s_file_names if i[-3:] == 'shp']
    # sort by file size so that each core gets a similar sized file to work with.
    size_list = []
    for i_sfn, sfn in enumerate(s_file_names):
        i_fpn = os.path.join(s_file_path, sfn)  # full path
        f_stats = os.stat(i_fpn)  # get file stats

        size_list.append(f_stats.st_size)  # file size

    # create a dataframe and then sort by file size
    df = {'f_name': s_file_names, 'f_size': size_list}
    df = pd.DataFrame(data=df)
    df = df.sort_values(by='f_size')

    s_file_names = df['f_name'].tolist()
    # s_file_names = s_file_names[:6]

    n_features = len(s_file_names)

    db_path_name = os.path.join(db_path, db_name)
    db_conn = sqlite3.connect(db_path_name)

    process_list = []
    write_count = 0
    for i_sfn, s_file_name in enumerate(s_file_names):
        print('Reading in:', s_file_name)

        s_file_path_name = os.path.join(s_file_path, s_file_name)

        process_list.append([s_file_path_name, True])

        if check_for_processing(n_features=n_features, i_id=i_sfn,
                                per_core=1):

            with Pool(processes=6) as p:
                df_list = p.map(func=load_blocks, iterable=process_list)

            # df_list = map(load_blocks,process_list)

            # create the output dataframe
            output_df = pd.concat(df_list)
            del df_list


            # add the state county information
            if state_county_fn:
                output_df = add_county_fips(gdf=output_df,
                                            state_county_fn=state_county_fn)

            if cbsa_db_path_name:
                output_df = add_cbsa_fips(gdf=output_df,
                                          cbsa_db_path_name=cbsa_db_path_name)

            process_list = []

            # replace option
            if write_count == 0:
                replace_option = 'replace'
            else:
                replace_option = 'append'

            print('...writing data to db')
            output_df.to_sql(name=table_name, con=db_conn, if_exists=replace_option,
                             index=False)

            write_count += 1

    db_conn.close()

    if index_list:
        add_table_indices(db_path=db_path, db_name=db_name,
                          index_list=index_list)


def main():

    print('blarh')


if __name__ == '__main__':
    s_file_path = 'H:/data/census_geography/tracts/tracts2017/shapefiles'
    db_path = 'H:/data/census_geography/tracts/tracts2017'
    db_name = 'tl_2017_us_tracts.db'

    unique_id_fn = ['geoid']

    state_county_fn = ['statefp', 'countyfp']

    cbsa_db_path = 'H:/data/census_geography/cbsa'
    cbsa_db_name = 'cbsa.db'
    cbsa_db_path_name = os.path.join(cbsa_db_path, cbsa_db_name)

    index_list = ['CREATE UNIQUE INDEX tl_2017_us_tracts_geoid_uindex ON tl_2017_us_tracts (geoid);',
                  'CREATE INDEX tl_2017_us_tracts_bbox_index ON tl_2017_us_tracts (x_min, y_min, x_max, y_max);',
                  'CREATE INDEX tl_2017_us_tracts_countyfips_index ON tl_2017_us_tracts (countyfips);',
                  'CREATE INDEX tl_2017_us_tracts_cbsa_index ON tl_2017_us_tracts (cbsa_geoid);']

    # enumerate_s_file_directory(s_file_path=s_file_path, db_path=db_path,
    #                            db_name=db_name, table_name='tl_2017_us_tracts',
    #                            state_county_fn=state_county_fn,
    #                            cbsa_db_path_name=cbsa_db_path_name,
    #                            index_list=index_list)

    # create centroids
    
    sql_ = 'select statefp, cbsa_geoid, countyfips, geoid, wkt from tl_2017_us_tracts'
    gdf = load_geo_data_from_sqlite(db_path=db_path, db_name=db_name, sql=sql_)

    gdf = create_centroids(gdf)
    gdf['wkt'] = gdf['geometry'].map(get_wkt)
    gdf = gdf.drop('geometry', 1)
    print(gdf.head())
    print(gdf.crs)

    write_data_to_sqlite(df=gdf, table_name='tl_2017_us_tracts_centroids',
                            db_path=db_path, db_name=db_name)