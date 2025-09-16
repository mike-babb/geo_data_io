# import county boundaries
# standard libraries
import os
import sqlite3

# external
import pandas as pd
import geopandas as gpd
import numpy as np

# custom
from df_operations import write_data_to_sqlite, change_column_name_case
from sqlite_operations import delete_db, add_table_indices
from fc_df_spatial import load_geospatial_data, load_geo_data_from_sqlite, create_centroids, get_wkt

# delete the database
db_path_ = 'H:/data/census_geography/county'
db_name_ = 'tl_2017_us_county.db'
delete_db(db_path=db_path_, db_name=db_name_)

# read in the data from the shapefile
gs_path_ = 'H:/data/census_geography/county'
gs_name_ = 'tl_2017_us_county.shp'
gdf_ = load_geospatial_data(gs_path=gs_path_, gs_name=gs_name_)

# write the county data to disk
write_data_to_sqlite(df=gdf_, table_name='tl_2017_us_county',
                     db_path=db_path_, db_name=db_name_)

# create centroids
sql_ = 'select GEOID, STATEFP, wkt from tl_2017_us_county'
gdf_ = load_geo_data_from_sqlite(db_path=db_path_, db_name=db_name_, sql=sql_)

gdf_ = create_centroids(gdf_)
gdf_['wkt'] = gdf_['geometry'].map(get_wkt)
gdf_ = gdf_.drop('geometry', 1)
print(gdf_.head())
print(gdf_.crs)

write_data_to_sqlite(df=gdf_, table_name='tl_2017_us_county_centroids',
                     db_path=db_path_, db_name=db_name_)

# add indices
idx_list = [
    'CREATE UNIQUE INDEX uidx_tl_2017_us_county_geoid ON tl_2017_us_county (GEOID);',
    'CREATE INDEX idx_tl_2017_us_county_statefp ON tl_2017_us_county (STATEFP);',
    'CREATE UNIQUE INDEX uidx_tl_2017_us_county_centroids_geoid ON tl_2017_us_county_centroids (GEOID);',
    'CREATE INDEX idx_tl_2017_us_county_centroids_statefp ON tl_2017_us_county_centroids (STATEFP);']


add_table_indices(db_path=db_path_, db_name=db_name_, index_list=idx_list)
