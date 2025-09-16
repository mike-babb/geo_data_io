####
# CREATE A SQLITEDB WITH CENSUS BLOCKS
####

# standard libraries
import os
import sqlite3
from multiprocessing import Pool

# external libraries
import pandas as pd
import geopandas as gpd

def load_blocks(f_values):
    s_file_path_name, county_cbsa_df = f_values


    gdf = gpd.read_file(s_file_path_name)
    n_features = '{:,}'.format(len(gdf))
    print('...loaded', n_features, 'features.')

    wgs84_crs = {'init': 'epsg:4326'}

    # let's convert to WGS 84
    gdf = gdf.to_crs(wgs84_crs)
    print('...finished re-projecting.')

    # extract the bounding box values
    temp_lambda = lambda x: x.bounds

    bbox = gdf['geometry'].map(temp_lambda)

    # get the county fips
    gdf['COUNTYFIPS'] = gdf['STATEFP10'] + gdf['COUNTYFP10']

    # Returns minimum bounding region (minx, miny, maxx, maxy)

    col_names = ['x_min', 'y_min', 'x_max', 'y_max']
    for i_cn, cn in enumerate(col_names):
        temp_lambda = lambda x: x[i_cn]
        gdf[cn] = bbox.map(temp_lambda)

    # calculate the x_dff, y_dff and xy_area
    gdf['x_diff'] = gdf['x_max'] - gdf['x_min']
    gdf['y_diff'] = gdf['y_max'] - gdf['y_min']
    gdf['xy_area'] = gdf['x_diff'].abs() * gdf['y_diff'].abs()

    # get the wkt
    temp_lambda = lambda x: x.get_wkt()
    gdf['wkt'] = gdf['geometry'].map(temp_lambda)

    # to sql
    gdf = gdf.drop('geometry', 1)

    # join in the cbsa df
    print(len(gdf))
    gdf = pd.merge(left=gdf, right=county_cbsa_df, how='left')
    print(len(gdf))

    col_names = gdf.columns.tolist()
    col_names = [str(i).upper() for i in col_names]
    gdf.columns = col_names

    return gdf


def enumerate_s_file_directory(s_file_path):

    # working path
    s_file_names = os.listdir(s_file_path)
    s_file_names = [i for i in s_file_names if i[-3:] == 'shp']
    size_list = []
    for sfn in s_file_names:
        i_fpn = os.path.join(s_file_path, sfn)
        f_stats = os.stat(i_fpn)

        size_list.append(f_stats.st_size)

    df = {'f_name': s_file_names, 'f_size': size_list}

    df = pd.DataFrame(data=df)
    df = df.sort_values(by='f_size')

    print(df.head())

    s_file_names = df['f_name'].tolist()
    # s_file_names = s_file_names[:6]

    db_path = s_file_path
    db_name = 'blocks2010.db'
    db_path_name = os.path.join(db_path, db_name)
    db_conn = sqlite3.connect(db_path_name)

    # load the county to cbsa cross walk
    sql = 'select * from county_cbsa'
    county_cbsa_df = pd.read_sql(sql = sql, con=db_conn)
    col_names = ['COUNTYFIPS', 'CBSA_GEOID']
    county_cbsa_df.columns = col_names


    process_list = []
    write_count = 0
    for i_sfn, s_file_name in enumerate(s_file_names):
        print('Reading in:', s_file_name)

        s_file_path_name = os.path.join(s_file_path, s_file_name)

        process_list.append([s_file_path_name, county_cbsa_df])

        if i_sfn > 0 and (len(process_list) % 6 == 0 or i_sfn == len(s_file_names) - 1):
            with Pool(processes=6) as p:
                df_list = p.map(func=load_blocks,iterable=process_list)

            output_df = pd.concat(df_list)
            df_list = None

            process_list = []

            # replace option
            if write_count == 0:
                replace_option = 'replace'
            else:
                replace_option = 'append'

            print('...writing data to db')
            output_df.to_sql(name = 'blocks2010', con=db_conn, if_exists = replace_option, index = False)

            write_count += 1

    # generate indices
    db_cursor = db_conn.cursor()

    print('adding block index')
    sql = 'CREATE UNIQUE INDEX blocks2010_BLOCKID10_uindex ON blocks2010 (BLOCKID10);'
    db_cursor.execute(sql)

    print('adding bbox index')
    sql = 'CREATE INDEX blocks2010_bbox_index ON blocks2010 (x_min, y_min, x_max, y_max);'
    db_cursor.execute(sql)

    print('adding county fips index')
    sql = 'CREATE INDEX blocks2010_countyfips_index ON blocks2010 (COUNTYFIPS);'
    db_cursor.execute(sql)

    print('adding cbsa fips index')
    sql = 'CREATE INDEX blocks2010_cbsa_index ON blocks2010 (cbsa_geoid);'
    db_cursor.execute(sql)

    print('saving indices')
    db_conn.commit()

    db_cursor.close()
    db_conn.close()

    print('finished.')

if __name__ == '__main__':
    s_file_path = '/Users/babbm/data/census_geography/blocks'
    enumerate_s_file_directory(s_file_path)