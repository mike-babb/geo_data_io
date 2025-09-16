# a scratch file to import spatial data. A template
# standard
import os
import sqlite3

# external
import pandas as pd
import geopandas as gpd

# custom
from fc_df_spatial import 

sf_path = 'H:/data/census_geography/county'
sf_name = 'tl_2010_us_county00.shp'
sf_pn = os.path.join(sf_path, sf_name)
print(sf_pn)



# read in as a gdf
gdf = gpd.read_file(filename=sf_pn)


# # to sqlite
# db_path = sf_path
# db_name = 'tl_2018_us_state.db'
# write_geo_data_to_sqlite(gdf=gdf,table_name='tl_2018_us_state',
# db_path=db_path,db_name=db_name)



