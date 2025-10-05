# mike mf babb
# convert a featureclass to a sqlitedb table
# uses ogr.

# standard libraries
import os
import sqlite3

# external
import geopandas as gpd
import numpy as np
from osgeo import ogr
import pandas as pd
import shapely.geometry
from shapely.wkt import loads
from shapely.geometry import Polygon
from shapely.ops import linemerge


# custom
from df_operations import \
    into_dict, load_data_from_sqlite, write_data_to_sqlite
from sqlite_operations import add_table_indices, delete_db

# extract the bounding box values
def get_bbox(x):
    return x.bounds


def get_area(x):
    return x.area


def get_envelope_area(x):
    return x.envelope.area


def get_wkt(x):
    return x.wkt


def get_x_coord(x):
    return x.x


def get_y_coord(x):
    return x.y


def unpack_bbox_values(gdf):
    # extract the bounding box values
    bbox = gdf['geometry'].map(get_bbox)

    # Returns minimum bounding region (minx, miny, maxx, maxy)
    col_names = ['x_min', 'y_min', 'x_max', 'y_max']
    for i_cn, cn in enumerate(col_names):
        def calc_fn(x):
            return x[i_cn]
        gdf[cn] = bbox.map(calc_fn)

    return gdf


def create_geometry(gdf, wkt_fn='wkt', crs=None):

    # create the geometry
    print('...creating geometry')
    geos = gdf[wkt_fn].map(loads)

    # create the coordinate system
    if crs:
        pass
    else:
        crs = {'init': 'epsg:4326'}

    # create a geoseries and a geodataframe
    geos = gpd.GeoSeries(geos, crs=crs)
    gdf = gpd.GeoDataFrame(data=gdf, geometry=geos)
    # drop the field names
    gdf = gdf.drop([wkt_fn], 1)

    return gdf


def load_geo_data_from_sqlite(db_path, db_name, sql, wkt_fn='wkt', crs=None):
    """

    :param db_path:
    :param db_name:
    :param sql:
    :param wkt_fn:
    :param crs:
    :return:
    """

    # load data from sqlite
    gdf = load_data_from_sqlite(db_path=db_path, db_name=db_name,
                                sql=sql)

    # create geometry
    gdf = create_geometry(gdf=gdf, wkt_fn=wkt_fn, crs=crs)

    return gdf


def write_geo_data_to_sqlite(gdf, table_name, db_path=None, db_name=None,
                             db_conn=None, if_exists_option='replace',
                             index_option=False, convert_to_wgs84=True,
                             change_case='lower'):
    """
    Write a GeoPandas GeoDataFrame to SQLite.
    :param df:
    :param table_name:
    :param db_path:
    :param db_name:
    :param db_conn:
    :param if_exists_option:
    :param index_option:
    :return:
    """

    col_names = gdf.columns.tolist()
    if 'geometry' in col_names:

        if convert_to_wgs84:
            print('...reprojecting to WGS 84...')
            crs = {'init': 'epsg:4326'}
            gdf = gdf.to_crs(crs)

        if 'wkt' in col_names:
            pass
        else:
            print('...creating wkt...')
            gdf['wkt'] = gdf['geometry'].map(get_wkt)

        gdf = gdf.drop('geometry', 1)

    if change_case in ('lower', 'upper'):
        col_names = gdf.columns.tolist()
        if change_case == 'lower':
            col_names = [x.lower() for x in col_names]
        if change_case == 'upper':
            col_names = [x.upper() for x in col_names]

        gdf.columns = col_names

    write_data_to_sqlite(df=gdf, table_name=table_name, db_path=db_path,
                         db_name=db_name, db_conn=db_conn,
                         if_exists_option=if_exists_option,
                         index_option=index_option)

    return None


def write_geo_data(gdf, file_path, file_name, driver='GeoJSON'):
    """

    :param gdf:
    :param file_path:
    :param file_name:
    :param driver:
    :return:
    """

    if driver == 'GeoJSON':
        f_pn = os.path.join(file_path, file_name)
        if os.path.exists(f_pn):
            os.remove(f_pn)

        gdf.to_file(filename=f_pn, driver=driver)

    return None


def make_point_wkt(df, wkt_fn, lng_fn, lat_fn):
    """
    Create the wkt for point geometry.
    :param df: Pandas dataframe.
    :param wkt_fn: string. Name of the field to hold the wkt .
    :param lng_fn: string. Name of the field with the lng/x coordinate.
    :param lat_fn: string. Name of the field with the lat/y coordinate.
    :return: Pandas dataframe.
    """

    df[wkt_fn] = 'POINT (' + df[lng_fn].astype(str) + ' ' + \
                 df[lat_fn].astype(str) + ')'

    return df


def make_bbox_wkt(df, wkt_fn, x_min_fn, y_min_fn, x_max_fn, y_max_fn):
    """
    Create a bounding box wkt.
    :param df:
    :param wkt_fn: string. Name of the field to hold the wkt .
    :param x_min: string. Name of the field with the
    :param y_min:
    :param x_max:
    :param y_max:
    :return: Pandas dataframe
    """

    def make_wkt(row, x_min_fn, y_min_fn, x_max_fn, y_max_fn):

        # get the points
        x_min = row[x_min_fn]
        x_max = row[x_max_fn]
        y_min = row[y_min_fn]
        y_max = row[y_max_fn]

        # build the point tuples.
        ll = (x_min, y_min)
        lr = (x_max, y_min)
        ur = (x_max, y_max)
        ul = (x_min, y_max)

        # shapely polygon
        geo = Polygon([ll, lr, ur, ul])
        geo_wkt = geo.wkt
        return geo_wkt

    df[wkt_fn] = df.map(func=make_wkt, axis=1,
                        args=(x_min_fn, y_min_fn, x_max_fn, y_max_fn))

    return df


def add_bounding_box_values(gdf, geo_fn='geometry'):
    """
    Create and store the bounding box values.
    DOES NOT CREATE THE BOUNDING BOX.
    SEE make_bbox_wkt()
    :param df:
    :param id_fn:
    :param geo_fn:
    :return:
    """

    bbox = gdf[geo_fn].map(get_bbox)
    # Returns minimum bounding region (minx, miny, maxx, maxy)
    col_names = ['x_min', 'y_min', 'x_max', 'y_max']
    for i_cn, cn in enumerate(col_names):
        def calc_fn(x):
            return x[i_cn]

        gdf[cn] = bbox.map(calc_fn)

    return gdf


def esri_fc_to_pandas_df(fc_path, fc_name, return_spatial_ref=False,
                         driver='ESRI Shapefile'):
    """
    Right now, this function essentially replicates reading in an ESRI FC
    to a pandas df using nothing but OGR. It's cool, but let's replace this
    with GeoPandas.
    Reads an ESRI featureclass (either a file geodatabase featureclasss or
    a shapefile) and returns a pandas dataframe with the data.
    :param fc_path: string. Path to the feature class.
    :param fc_name: string. Name of the feature class.
    :param return_geometry: Boolean. Option to include the geometry
    with the tabular data.
    :param return_spatial_ref: Boolean. Option to return the feature classes'
    spatial reference.
    :param fc_type: string. Indicates the type of input featureclass.
    shp for shapefile and fc for file geodatabase featureclass.
    :return: 
    """

    # path and name of the featureclass
    print('Now reading ' + fc_name)

    if driver == 'OpenFileGDB':
        gdf = gpd.read_file(filename=fc_path, driver='OpenFileGDB',
                            layer=fc_name)
    else:
        fc_pn = os.path.join(fc_path, fc_name)
        gdf = gpd.read_file(filename=fc_pn)

    # get the spatial reference
    if return_spatial_ref:
        crs = gdf.crs
    else:
        crs = None

    # count number of features
    feature_count = len(gdf)

    # delete the shape area and shape length fields
    column_names = gdf.columns.tolist()
    if 'Shape_Length' in column_names:
        gdf = gdf.drop('Shape_Length', 1)

    if 'Shape_Area' in column_names:
        gdf = gdf.drop('Shape_Area', 1)

    print('...DONE...')

    if return_spatial_ref:
        return gdf, crs
    else:
        return gdf


def pandas_df_to_shapefile(df, shapefile_path, shapefile_name,
                           geo_fn='wktGeo', create_geo_field=True,
                           crs={'init': 'epsg:4269'}):
    """
    Same thing as above - writing a shapefile using nothing but OGR.
    Pretty cool - but GeoPandas does this as well.
    :param df:
    :param shapefile_path: 
    :param shapefile_name: 
    :param geo_fn: 
    :param create_geometry: 
    :param crs:
    :return: 
    """

    # a dictionary that translates between the pandas/numpy data type
    # and the ogr datatype

    # create a path to the data source / shapefile
    if shapefile_name[-4:] == '.shp':
        output_sfile_name = shapefile_name[:-4]
    else:
        output_sfile_name = shapefile_name

    print(output_sfile_name)
    output_sfile_p_n = os.path.join(shapefile_path, output_sfile_name)

    # create geometry, if needed
    if create_geo_field:
        df = create_geometry(gdf=df, wkt_fn=geo_fn, crs=crs)

    df.to_file(filename=output_sfile_p_n, driver='ESRI Shapefile')

    return None


def read_features_into_dictionary(file_path, file_name,
                                  ref_field_name='GEOID'):
    """

    :param file_path: 
    :param file_name: 
    :param ref_field_name: 
    :return: 
    """
    # TODO: fix the way a geodf loads

    file_pn = os.path.join(file_path, file_name)

    gdf = gpd.read_file(filename=file_pn)

    feature_dict = into_dict(df=gdf, key_fn=ref_field_name)

    return feature_dict


def export_gdb(gdb_path, gdb_name, export_type='db', export_path=None):
    """
    Export an ESRI db to a sqlite db, text files, or a dictionary of pandas 
    dataframes keyed by featureclass name. Uses OGR and pandas.
    :param gdb_path: string. path to the ESRI file geodatabase.
    :param gdb_name: string. name of the ESRI file geodatabase.
    :param export_type: string. a value in ('db', 'text', 'dict').
    Default is a sqlite db.  
    :param export_path: string. Alternate destination path. Default is the
    same as the input gdb_path.
    :return: dict (optional). Dictionary of pandas dataframes
    """

    if export_type == 'db':
        # sqlite
        if export_path:
            db_path = export_path
        else:
            db_path = gdb_path
        db_name = gdb_name[:-3] + '.db'
        db_path_name = os.path.join(db_path, db_name)
        db_conn = sqlite3.connect(db_path_name)
        db_conn.text_factory = str

    elif export_type == 'txt':
        # text file
        if export_path:
            if os.path.exists(export_path):
                pass
            else:
                os.mkdir(export_path)

            file_path = export_path
        else:
            file_path = gdb_path
    else:
        # dictionary of data frames
        df_dict = {}

    # use OGR specific exceptions
    ogr.UseExceptions()

    # create the open file gdb driver
    driver = ogr.GetDriverByName("OpenFileGDB")

    gdb_path_name = os.path.join(gdb_path, gdb_name)

    # open the FileGDB as a vector data store
    vector_ds = driver.Open(gdb_path_name, 0)

    # list to store layers' names
    f_class_list = []

    # parsing layers by index to get a list of layers
    for f_class_idx in range(vector_ds.GetLayerCount()):
        f_class = vector_ds.GetLayerByIndex(f_class_idx)
        f_class_list.append(f_class.GetName())

    # close the connection
    vector_ds.Destroy()
    del driver

    # sort the list
    f_class_list.sort()

    for fc_name in f_class_list:
        print(fc_name)
        df = gpd.read_file(filename=gdb_path_name,
                           layer=fc_name, driver='OpenFileGDB')

        # if there is valid geometry - convert to wkt
        geo_check = df['geometry'].is_valid.unique()
        if len(geo_check) == 1 and geo_check[0] is False:
            # drop the geometry field
            pass
        else:
            df['wkt'] = df['geometry'].map(get_wkt)

        df = df.drop('geometry', 1)

        if export_type == 'db':
            write_data_to_sqlite(df=df, table_name=fc_name, db_conn=db_conn,
                                 if_exists_option='replace',
                                 index_option=False)

        elif export_type == 'txt':
            file_name = fc_name + '.txt'
            file_path_name = os.path.join(file_path, file_name)
            df.to_csv(path_or_buf=file_path_name, sep='\t', index=False)

        else:
            df_dict[fc_name] = df

    if export_type == 'db':
        db_conn.close()

    if export_type == 'dict':
        return df_dict
    else:
        return None


def create_centroids(gdf):
    """
    Create the centroids.
    :param gdf:
    :param output_name:
    :return:
    """

    centroid_geo = gdf.centroid
    gdf = gdf.set_geometry(centroid_geo)

    return gdf


def load_geospatial_data(gs_path, gs_name, reproject=True):
    """
    Read in geospatial data stored in a text file.
    :param gs_path:
    :param gs_name:
    :param db_path:
    :param db_name:
    :return:
    """
    gs_pn = os.path.join(gs_path, gs_name)

    # read in the file as a geodataframe
    gdf = gpd.read_file(gs_pn)
    n_features = '{:,}'.format(len(gdf))
    print('...loaded', n_features, 'features for', os.path.normpath(gs_pn))

    # get the coordinate system
    curr_crs = gdf.crs
    if reproject:
        if 'init' in curr_crs:
            if curr_crs['init'] == 'epsg:4326':
                pass
        else:
            # reproject that bad boy
            # let's convert FROM NAD 1983 TO WGS 84
            wgs84_crs = {'init': 'epsg:4326'}
            gdf = gdf.to_crs(wgs84_crs)
            print('...finished re-projecting.')

    # add the bounding box values
    gdf = add_bounding_box_values(gdf=gdf)

    # calculate the area of the geometry
    gdf['xy_area'] = gdf['geometry'].map(get_area)

    gdf['wkt'] = gdf['geometry'].map(get_wkt)

    # drop the geometry
    gdf = gdf.drop('geometry', 1)

    # set column names to uppercase
    # maybe do lowercase in the future?
    col_names = gdf.columns.tolist()
    col_names = [str(i).upper() for i in col_names]
    gdf.columns = col_names

    return gdf


def create_bounding_box_geometry(gdf):

    # extract the bounding box geometry
    bbox_geo = gdf['geometry'].envelope
    gdf['geometry'] = bbox_geo

    return gdf


def extract_spatial_reference(s_file_path, s_file_name):
    print('blarh')


def delete_shapefile(s_file_path, base_s_file_name):
    s_file_extensions = ['.cpg', '.dbf', '.prj', '.shp', '.shx', '.shp.xml']

    for sfe in s_file_extensions:
        s_file_name_part = base_s_file_name + sfe
        sfpn = os.path.join(s_file_path, s_file_name_part)
        if os.path.exists(sfpn):
            print('...deleting', os.path.normpath(sfpn))
            os.remove(sfpn)

    return None


def rename_shapefile(s_file_path, base_s_file_name, new_s_file_path,
                     new_base_s_file_name):
    s_file_extensions = ['.cpg', '.dbf', '.prj', '.qix', '.qpj', '.shp',
                         '.shp.ea.iso.xml', '.shp.iso.xml',
                         '.shp.xml', '.shx']

    for sfe in s_file_extensions:
        s_file_name_part = base_s_file_name + sfe
        sfpn = os.path.join(s_file_path, s_file_name_part)
        if os.path.exists(sfpn):
            print('...renaming', os.path.normpath(sfpn))
            new_s_file_name_part = new_base_s_file_name + sfe
            new_sfpn = os.path.join(new_s_file_path, new_s_file_name_part)

            os.rename(sfpn, new_sfpn)

    return None


def write_gdf(gdf: gpd.GeoDataFrame, output_file_path:str, output_file_name:str):
    
    ofpn = os.path.join(output_file_path, output_file_name)

    if 'coords' in gdf.columns:
        output_gdf = gdf.drop(labels = ['coords'], axis = 1)
        output_gdf.to_file(filename = ofpn, driver = 'GPKG', index = False)
    else:
        gdf.to_file(filename = ofpn, driver = 'GPKG', index = False)

    return None


def check_MultiLineStrings(geom:shapely.geometry):
    # does every MultiLineString need to be a MultilineString?
    output = geom
    if geom.geom_type == 'MultiLineString':
        new_geom = linemerge(lines = geom)
        if new_geom.geom_type == 'LineString':
            output = new_geom
    return output


def keep_largest_geometry(gdf:gpd.GeoDataFrame,group_col_names:list = None):

    # explode MultiPolygon Geometries and keep only the largest.
    # this removes slivers and splinters.
        
    # explode
    gdf = gdf.explode()
    keep_col_names = gdf.columns
    # select only Polygon or MultiPolygon geometries
    # some geospatial operations produce errant points and LineStrings
    gdf = gdf.loc[gdf['geometry'].geom_type.isin(['Polygon', 'MultiPolygon']), :]

    # keep the biggest piece / remove slivers.
    # Compute the area to accomplish this
    gdf['geom_area'] = gdf['geometry'].area
    
    if group_col_names is None:
        gdf['area_rank'] = gdf['geom_area'].rank(method = 'dense', ascending=False)
    else:
        gdf['area_rank'] = gdf.groupby(group_col_names)['geom_area'].rank(method = 'dense', ascending=False)
    
    # keep the largest, drop the area_rank column
    gdf = gdf.loc[gdf['area_rank'] == 1, keep_col_names]

    return gdf

def build_gdf_from_geom(geom:shapely.geometry, remove_slivers:bool=True, 
                        return_geom:bool=False,
                        crs:int=4326):
    # given a single shapely geometry, create a dataframe from it.
    # optionally remove the slivers 
    gdf = gpd.GeoDataFrame(data = {'id':[0]}, geometry=[geom], crs = crs)
    if remove_slivers:
        gdf = keep_largest_geometry(gdf = gdf)
    
    # optionally, return only the geometry
    if return_geom:
        output = gdf['geometry'].iloc[0]
    else:
        output = gdf

    return output


# let's do some fun inner ring buffering
def inner_ring_buffer(gdf:gpd.GeoDataFrame, dist_start:int, dist_end:int, buff_dist:int):

    # output lists
    # polygon
    output_data_list = []
    output_geom_list = []

    # lines
    output_line_data_list = []
    output_line_geom_list = []
    col_names = gdf.columns.tolist()
    col_names.remove('geometry')

    for ir, row in gdf.iterrows():
        
        # the focal geometry
        geom = row['geometry']
        # the perimeter
        perim = geom.boundary
        # a dictionary to store the previously created buffer
        # important for creating rings
        previous_buff_dict = {}
        # buffer out 10 units at a time. The units are the same as the units of the
        # geometry's coordinate system.
        for i_dist in range(dist_start, dist_end + 1, buff_dist):
            # buffer the perimeter. This creates geometry that is both on the inside
            # and outside of the input focal geometry
            my_buff = perim.buffer(distance= i_dist)
            
            # perform an intersection to get only the stuff on the inside.
            my_buff = my_buff.intersection(geom)
            
            # remove slivers and splinters
            my_buff = build_gdf_from_geom(geom = my_buff,return_geom=True, crs = gdf.crs)        
            
            # add this cleaned geometry to the previous buffer dictionary
            previous_buff_dict[i_dist] = my_buff

            # now, clip it to the previous buffer
            if i_dist > 10:
                previous_buff = previous_buff_dict[i_dist - 10] 
                # the difference is the part that doesn't overlap - this is the 
                # next ring in the series. 
                my_buff = my_buff.difference(previous_buff)
                
                my_buff = build_gdf_from_geom(geom = my_buff, return_geom=True, crs = gdf.crs)        
            
            # this is for the polygon output                
            temp_list = [row[cn] for cn in col_names]
            temp_list.append(i_dist)
            output_data_list.append(temp_list)
            output_geom_list.append(my_buff)

            # extract the lines for these inner ring buffers. 
            # one-stop shopping
            line_index = 0
            geom_boundary = my_buff.boundary
            if geom_boundary.geom_type == 'MultiLineString':            
                geom_list = geom_boundary.geoms
            else:
                geom_list = [geom_boundary]

            for line_geom in geom_list:
                curr_list = temp_list[:]
                curr_list.append(line_index)
                output_line_data_list.append(curr_list)
                output_line_geom_list.append(line_geom)
                line_index += 1


        print('Processing row:', f"{ir:,}")

    # create the polygon output gdf
    output_col_names = col_names[:]
    output_col_names.append('distance')
    output_gdf = gpd.GeoDataFrame(data = output_data_list, geometry = output_geom_list,
                                crs = gdf.crs, columns = output_col_names)
    
        # create the line output gdf
    output_col_names = col_names[:]
    output_col_names.append('distance')
    output_col_names.append('line_index')
    output_line_gdf = gpd.GeoDataFrame(data = output_line_data_list, geometry = output_line_geom_list,
                                crs = gdf.crs, columns = output_col_names)
    
    return (output_gdf, output_line_gdf)




if __name__ == '__main__':
    # delete the database
    db_path_ = 'H:/data/census_geography/county'
    db_name_ = 'tl_2018_us_county.db'
    delete_db(db_path=db_path_, db_name=db_name_)

    # read in the data from the shapefile
    gs_path_ = 'H:/data/census_geography/county'
    gs_name_ = 'tl_2018_us_county.shp'
    gdf_ = load_geospatial_data(gs_path=gs_path_, gs_name=gs_name_)

    # write the county data to disk
    write_data_to_sqlite(df=gdf_, table_name='tl_2018_us_county',
                         db_path=db_path_, db_name=db_name_)

    gdf_ = create_geometry(gdf=gdf_)

    gdf_ = create_centroids(gdf_)
    gdf_['wkt'] = gdf_['geometry'].map(get_wkt)
    gdf_ = gdf_.drop('geometry', 1)
    print(gdf_.head())
    print(gdf_.crs)

    write_data_to_sqlite(df=gdf_, table_name='tl_2018_us_county_centroids',
                         db_path=db_path_, db_name=db_name_)

    # # add indices
    # idx_list = [
    # 'CREATE UNIQUE INDEX uidx_tl_2017_us_county_geoid ON tl_2017_us_county (GEOID);',
    # 'CREATE INDEX idx_tl_2017_us_county_statefp ON tl_2017_us_county (STATEFP);',
    # 'CREATE UNIQUE INDEX uidx_tl_2017_us_county_centroids_geoid ON tl_2017_us_county_centroids (GEOID);',
    # 'CREATE INDEX idx_tl_2017_us_county_centroids_statefp ON tl_2017_us_county_centroids (STATEFP);']

    # add_table_indices(db_path=db_path_,db_name=db_name_,index_list=idx_list)
