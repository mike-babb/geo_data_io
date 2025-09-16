####
# CREATE A SQLITEDB WITH CENSUS BLOCKS
####

# standard libraries
import os
import sqlite3

# external libraries
import pandas as pd
import geopandas as gpd
import numpy as np

# custom
from df_operations import write_data_to_sqlite
from fc_df_spatial import get_wkt, unpack_bbox_values


def import_cbsa_pop_data(pop_file_path, pop_file_name):

    # import the population data
    pop_fpn = os.path.join(pop_file_path, pop_file_name)
    pop_df = pd.read_csv(filepath_or_buffer=pop_fpn,
                         sep=',', header=0, encoding='cp1252')
    drop_column_names = ['GEO.id', 'GEO.display-label', 'HD02_VD01']
    pop_df = pop_df.drop(drop_column_names, 1)
    # remove the second row
    pop_df = pop_df.iloc[1:]

    # set column names
    col_names = ['GEOID', 'ACS_16_5YR_POP']
    pop_df.columns = col_names

    # change types
    pop_df['ACS_16_5YR_POP'] = pop_df['ACS_16_5YR_POP'].astype(int32)

    return(pop_df)


def import_cbsa_sqlite():
    import sqlite3

    # working path
    s_file_path = 'H:/data/census_geography/cbsa'
    s_file_names = os.listdir(s_file_path)
    s_file_names = [i for i in s_file_names if i[-3:] == 'shp']

    db_path = s_file_path
    db_name = 'tl_2018_us_cbsa.db'
    db_path_name = os.path.join(db_path, db_name)
    db_conn = sqlite3.connect(db_path_name)

    # read in the population information
    pop_file_name = 'ACS_16_5YR_B01003_with_ann.csv'
    pop_df = import_cbsa_pop_data(pop_file_path=s_file_path,
                                  pop_file_name=pop_file_name)

    # wgs 84 coordinates
    wgs84_crs = {'init': 'epsg:4326'}

    for i_sfn, s_file_name in enumerate(s_file_names):
        print('Reading in:', s_file_name)

        s_file_path_name = os.path.join(s_file_path, s_file_name)
        gdf = gpd.read_file(s_file_path_name)
        n_features = '{:,}'.format(len(gdf))
        print('...loaded', n_features, 'features.')

        # let's convert to WGS 84
        gdf = gdf.to_crs(wgs84_crs)
        print('...finished re-projecting.')

        # extract the bounding box values
        gdf = unpack_bbox_values(gdf=gdf)

        # get the wkt
        gdf['wkt'] = gdf['geometry'].map(get_wkt)

        # to sql
        gdf = gdf.drop('geometry', 1)
        gdf = pd.merge(left=gdf, right=pop_df)

        # replace option
        if i_sfn == 0:
            replace_option = 'replace'
        else:
            replace_option = 'append'

        table_name = s_file_name[:-4]
        gdf.to_sql(name=table_name, con=db_conn,
                   if_exists=replace_option, index=False)

    return None


def import_county_cbsa_info():
    # working path
    cc_file_path = 'H:/data/census_geography/cbsa'

    # county to cbsa crosswalk
    cc_file_name = 'list1.xlsx'
    cc_fpn = os.path.join(cc_file_path, cc_file_name)

    dtype_dict = {'CBSA Code': str, 'Metropolitan Division Code': str,
                  'CSA Code': str, 'CBSA Title': str,
                  'Metropolitan/Micropolitan Statistical Area': str,
                  'Metropolitan Division Title': str, 'CSA Title': str,
                  'County/County Equivalent': str, 'State Name': str,
                  'FIPS State Code': str, 'FIPS County Code': str,
                  'Central/Outlying County': str}

    cc_df = pd.read_excel(io=cc_fpn, skiprows=[0, 1], dtype=dtype_dict)
    cc_df = cc_df.iloc[:1899]

    col_names = ['CBSA_CODE', 'MET_DIV_CODE', 'CSA_CODE', 'CBSA_TITLE', 'AREA_TYPE',
                 'MET_DIV_TITLE', 'CSA_TITLE', 'COUNTY', 'ST_NAME', 'STATEFIPS',
                 'CNTYFIPS', 'COUNTYSTATUS']

    cc_df.columns = col_names
    cc_df = cc_df.fillna('')
    cc_df = cc_df.replace('nan', '')

    cc_df['COUNTYFIPS'] = cc_df['STATEFIPS'] + cc_df['CNTYFIPS']

    print(cc_df.head())

    # add a field delineating metro/micro and inner/outlying
    print(cc_df['AREA_TYPE'].unique().tolist())
    print(cc_df['COUNTYSTATUS'].unique().tolist())
    col_names = ['micro', 'metro', 'central', 'outlying']
    for cn in col_names:
        cc_df[cn] = 0

    msa_values = ['Micropolitan Statistical Area',
                  'Metropolitan Statistical Area']
    col_names = ['micro', 'metro']
    for i_msa, msa in enumerate(msa_values):
        cn = col_names[i_msa]
        cc_df.loc[cc_df['AREA_TYPE'] == msa, cn] = 1

    centrality_values = ['Central', 'Outlying']
    col_names = ['central', 'outlying', ]
    for i_cv, cv in enumerate(centrality_values):
        cn = col_names[i_cv]
        cc_df.loc[cc_df['COUNTYSTATUS'] == cv, cn] = 1

    def metro_status(row):
        micro = row['micro']
        metro = row['metro']
        central = row['central']
        outlying = row['outlying']

        if micro + central == 2:
            outcome = 'central micro'
        elif micro + outlying == 2:
            outcome = 'outlying micro'
        elif metro + central == 2:
            outcome = 'central metro'
        elif metro + outlying == 2:
            outcome = 'outlying metro'
        else:
            outcome = 'non-metro'

        return outcome

    cc_df['metro_status'] = cc_df.apply(metro_status, 1)

    # turn these into betas
    col_names = ['central_micro', 'outlying_micro',
                 'central_metro', 'outlying_metro']
    for cn in col_names:
        w_cn = cn.replace('_', ' ')
        cc_df[cn] = 0
        cc_df.loc[cc_df['metro_status'] == w_cn, cn] = 1

    db_path = cc_file_path
    db_name = 'tl_2018_us_cbsa.db'

    write_data_to_sqlite(df=cc_df, table_name='county_cbsa_crosswalk',
                         db_path=cc_file_path, db_name=db_name)


def import_cbsa_mapper():
    import sqlite3
    cbsa_db_path = 'H:/data/census_geography/cbsa'
    cbsa_db_name = 'tl_2018_us_cbsa.db'
    cbsa_db_pn = os.path.join(cbsa_db_path, cbsa_db_name)

    tract_db_path = 'H:/data/census_geography/tracts/tracts2017'
    tract_db_name = 'tracts2017.db'
    tract_db_pn = os.path.join(tract_db_path, tract_db_name)

    block_db_path = 'H:/data/census_geography/blocks'
    block_db_name = 'blocks2010.db'
    block_db_pn = os.path.join(block_db_path, block_db_name)

    county_db_path = 'H:/data/census_geography/county'
    county_db_name = 'tl_2018_us_county.db'
    county_db_pn = os.path.join(county_db_path, county_db_name)

    # connections
    cbsa_db_con = sqlite3.connect(cbsa_db_pn)
    tract_db_con = sqlite3.connect(tract_db_pn)
    block_db_con = sqlite3.connect(block_db_pn)
    county_db_con = sqlite3.connect(county_db_pn)

    # get the county to cbsa mapper
    sql = 'select * from county_cbsa_crosswalk'
    cc_df = pd.read_sql(sql=sql, con=cbsa_db_con)
    col_names = ['countyfips', 'cbsa_code', 'cbsa_title', 'micro', 'metro', 'central', 'outlying',
                 'central_micro', 'outlying_micro', 'central_metro', 'outlying_metro',
                 'metro_status']

    cc_df = cc_df[col_names]

    # push to my cbsa db
    print('writing county_cbsa to cbsa')
    write_data_to_sqlite(
        df=cc_df, table_name='county_cbsa', db_conn=cbsa_db_con)

    # push to my tract db
    print('writing county_cbsa to tracts')
    write_data_to_sqlite(
        df=cc_df, table_name='county_cbsa', db_conn=tract_db_con)

    # push to my block db
    print('writing county_cbsa to blocks')
    write_data_to_sqlite(
        df=cc_df, table_name='county_cbsa', db_conn=block_db_con)

    print('writing county_cbsa to counties')
    write_data_to_sqlite(
        df=cc_df, table_name='county_cbsa', db_conn=county_db_con)

    block_db_con.close()
    cbsa_db_con.close()
    tract_db_con.close()
    county_db_con.close()

    return None


if __name__ == '__main__':

    import_cbsa_sqlite()
    import_county_cbsa_info()
    import_cbsa_mapper()
