###############################################################################
# Generate a grid of geohash cells that covers any polygons.
#

# standard libaries
import pickle
import itertools
import os
from multiprocessing import Pool, cpu_count
import math
import time
import sqlite3

# external libraries
import geopandas as gpd
import pandas as pd
import geohash
import numpy as np
from shapely.geometry import Point, Polygon
from shapely.wkt import loads
from shapely import speedups

# custom
from base_functions import *
from fc_df_spatial import *

# use python-Geohash to decode the cell values
# from this guy: https://github.com/hkwi/python-geohash/wiki/GeohashReference

def make_polygon_from_bounds(bounds):
    """
    Create a shapely polygon from bounds as stored in a dictionary
    :param bounds: dictionary. Keyed values of the bounds of a cell.
    :return: shapely polygon.
    """

    if type(bounds) is str:
        bounds = geohash.bbox(bounds)

    # extract the bounds based on dictionary keys
    x_min = bounds['w']
    y_min = bounds['s']
    x_max = bounds['e']
    y_max = bounds['n']

    # build the point tuples.
    ll = (x_min, y_min)
    lr = (x_max, y_min)
    ur = (x_max, y_max)
    ul = (x_min, y_max)

    # shapely polygon
    geo = Polygon([ll, lr, ur, ul])

    return geo


def get_delta_values(focal_geo, geohash_level):
    """
    Extract the delta values (width and height) of the centroid for use
    in determining the extent of the points to drop.
    :param focal_geo: shapely geometry.
    :param geohash_level: int. Precision of the geohash
    :return:
    """

    # get the centroid of the focal geometry
    geo_centroid = focal_geo.centroid
    gc_x = geo_centroid.x
    gc_y = geo_centroid.y

    # get the geohash of the centroid
    geo_centroid_gh = geohash.encode(latitude=gc_y, longitude=gc_x, precision=geohash_level)
    # delta values of the geohash at the centroid
    geo_centroid_delta = geohash.decode_exactly(geo_centroid_gh)

    # extract the width and height and place into a list.
    gh_width = geo_centroid_delta[2]
    gh_height = geo_centroid_delta[3]

    return [gh_width, gh_height]


def export_x_y_point_lists(point_list, intermediate_fpn):
    """
    Function used in testing only. Writes the generated list of points to disk.
    Can be used to help diagnose gaps in coverage.
    :param point_list: List. x and y coordinate pairs.
    :param intermediate_fpn: string. Path to and name of output file.
    :return:
    """

    # create a list of shapely geometry
    point_geo_list = [Point(x[0], x[1]) for x in point_list]

    # create a list of geohashes
    gh_list = [geohash.encode(latitude=x[1], longitude=x[0]) for x in point_list]

    # create a geodf
    point_gdf = gpd.GeoDataFrame(data=gh_list, geometry=gpd.GeoSeries(point_geo_list))

    col_names = ['gh', 'geometry']
    point_gdf.columns = col_names

    # write to disk
    geometry_to_disk(gh_gdf=point_gdf, output_fpn=intermediate_fpn)

    return None


def get_x_y_point_lists(focal_geo, gh_width, gh_height, geohash_level,
                        intermediate_fpn=None):
    """
    Create two lists: x, y, that when combined form the points
    to extract geohashes for
    :param focal_geo: shapely polygon.
    :param gh_width: float. width of the geohash cells.
    :param gh_height: float. height of the geohash cells.
    :param geohash_level: precision of the geohash
    :param intermediate_fpn: string. Optional file path and name of intermediate data.
    :return: geodataframe featuring geohashes and the
    corresponding polygons.
    """

    # bounds of the focal geometry
    # min_x, min_y, max_x, max_y
    x_min, y_min, x_max, y_max = focal_geo.bounds

    # calculate the number of values in the interval
    # the difference in the longitudes and latitudes
    # divided by the cell width.
    # here is where things get kinda tricky.
    # computing the x-difference and y-difference
    # and then dividing by the width and height of the geohash doesn't
    # guarantee cell-coverage over the entire geometry.
    # what's needed is to decrease the spacing between the interval
    # of the points by decreasing the width and height of the geohash cells
    # smaller width and height means more points

    x_diff = abs(x_max - x_min) / (gh_width / 2)
    y_diff = abs(y_max - y_min) / (gh_height / 2)

    # round up to the largest integer.
    x_diff = int(math.ceil(x_diff))
    y_diff = int(math.ceil(y_diff))

    # generate the x and y extents of the bounding boxes
    # using the np.linspace command
    # this basically chops an interval into a set number of pieces
    x_point_list = np.linspace(start=x_min, stop=x_max, num=x_diff)
    y_point_list = np.linspace(start=y_min, stop=y_max, num=y_diff)

    # use itertools to efficiently cross the point lists
    point_list = list(itertools.product(x_point_list, y_point_list))
    del x_point_list
    del y_point_list

    if intermediate_fpn:
        export_x_y_point_lists(point_list, intermediate_fpn)

    # use list comprehension to convert these lng/lat pairs into geohashes
    point_list = [geohash.encode(latitude=x[1], longitude=x[0],
                                 precision=geohash_level) for x in point_list]
    # convert the list to a set to find unique values.
    # And then back to a list.
    gh_list = list(set(point_list))

    # set objects to None to clean up memory.
    # useful to do in a multi-processing framework.
    del point_list

    # use a dictionary to hold the output
    gh_decode_dict = {}

    for i_gh, gh in enumerate(gh_list):

        # decode the cell bounding box values
        # into a polygon bounding box
        curr_decode = geohash.bbox(gh)
        poly = make_polygon_from_bounds(curr_decode)

        gh_decode_dict[gh] = poly

    # let's turn the gh_decode_dict into a dataframe
    gh_df = pd.DataFrame.from_dict(data=gh_decode_dict, orient='index')
    gh_df = gh_df.reset_index()
    col_names = ['gh', 'geometry']
    gh_df.columns = col_names

    # housekeeping
    del gh_decode_dict

    return gh_df


def get_geohash_centroid_point(gh):

    gh_decode = geohash.decode_exactly(gh)
    gh_centroid = Point(gh_decode[0], gh_decode[1])

    return gh_centroid


def perform_intersect(geo_series, focal_geo):
    # now, let's do some intersects

    res_intersections = geo_series.intersects(focal_geo)

    return res_intersections


def perform_intersection(geo_series, focal_geo):
    # this is the part that takes the longest time.
    # could we make it go faster?
    # is vectorization even a possibility here?

    res_intersections = geo_series.intersection(focal_geo)

    return res_intersections


def calc_overlap_percent(geo_series, focal_geo_area):
    # calculate the area of the intersection / overlap

    overlap_percent  = geo_series.area / focal_geo_area

    return overlap_percent


def perform_within_intersection(gh_gdf, focal_geo):
    # this returns a boolean - whether or not the
    # geohash cells are (completely) within the
    # focal geometry.
    res_within = gh_gdf['geometry'].within(focal_geo)

    focal_geo_area = focal_geo.area

    # determine what we have
    # A list of [True, False], [True], or [False]
    res_within_unique = res_within.unique().tolist()

    # this will tell us if the cells are all within or just partially within.
    if len(res_within_unique) == 1:
            if res_within_unique[0] is True:
                # all cells are within
                # no need to do an intersect
                # this will be rare. Very rare.
                # we can just calculate the areal overlap.
                gh_gdf['overlap_percent'] = calc_overlap_percent(
                    geo_series=gh_gdf['geometry'], focal_geo_area=focal_geo_area)
            else:
                # all cells overlap the focal geometry, but none are completely
                # within and so must compute the intersection
                res_intersections = perform_intersection(geo_series=gh_gdf['geometry'],
                                                         focal_geo=focal_geo)
                # calculate the area overlap
                gh_gdf['overlap_percent'] = calc_overlap_percent(
                    geo_series=res_intersections, focal_geo_area=focal_geo_area)
    else:
        # some of the cells are completely within, some are not.
        # this is the most likely scenario

        # geohashes completely within
        c_within = gh_gdf.loc[res_within, :].copy()
        c_within['overlap_percent'] = calc_overlap_percent(
            geo_series=c_within['geometry'], focal_geo_area=focal_geo_area)

        # geohashes partially within
        p_within = gh_gdf.loc[-res_within, :].copy()
        res_intersections = perform_intersection(geo_series=p_within['geometry'],
                                                 focal_geo=focal_geo)
        # calculate the area overlap
        p_within['overlap_percent'] = calc_overlap_percent(
            geo_series=res_intersections, focal_geo_area=focal_geo_area)

        # concatenate the two together
        gh_gdf = pd.concat([c_within, p_within])

    return gh_gdf


def geometry_to_disk(gh_gdf, output_fpn, o_file_encoding='GeoJSON'):
    # write the geohash cells to disk.

    output_file_path, output_file_name = os.path.split(output_fpn)

    if o_file_encoding == 'GeoJSON':

        if output_fpn[-4:] != 'json':
            output_fpn += 'json'

        if os.path.exists(output_fpn):
            os.remove(output_fpn)
    else:
        # lovely shapefile stuff
        sfile_endings = ['.cpg','.dbf','.prj','.shp','.shp.xml','.shx']
        for i_sfn, sf in enumerate(sfile_endings):
            curr_fpn = os.path.join(output_file_path,
                                    (output_file_name + sf) )
            if os.path.exists(curr_fpn):
                os.remove(curr_fpn)

    gh_gdf.to_file(filename=output_fpn, driver=o_file_encoding)


def allocate_geo(geoid, geo_id_fn, focal_geo, gh_df, intermediate_fpn):
    """
    Use areal weighting to determine how much of the focal geometry
    is covered by each geohash
    :param geoid: str. Identification of the geometry
    :param geo_id_fn: str. Name of the field with the identification.
    :param focal_geo: shapely geometry.
    :param gh_df: geopandas dataframe. Features geohash geometry.
    :param intermediate_fpn: str. If specified,
    directs intermediate output to be written
    :return: pandas dataframe featuring the percent areal overlap of the units.
    """

    # let's create the polygon gh gdf
    gh_gdf = gpd.GeoDataFrame(data=gh_df['gh'], geometry=gh_df['geometry'])

    # let's write the cells to disk - for testing. At this point, they
    # will be the extent of the bounding box.
    if intermediate_fpn:
        geometry_to_disk(gh_gdf=gh_gdf, output_fpn=intermediate_fpn)

    # This is a very quick way to determine if a geohash cell intersects with
    # the focal geometry
    res_intersection = perform_intersect(geo_series=gh_gdf['geometry'],
                                         focal_geo=focal_geo)
    n_gh = len(gh_gdf)

    # remove polygons that do not intersect:
    # the first stage of excess polygon removal
    gh_gdf = gh_gdf[res_intersection].copy()
    n_intersect_gh = len(gh_gdf)

    # print('...Found', n_intersect_gh, 'out of', n_gh,
    #       'geohashes to test for intersection for', geoid)

    # compute the actual intersection
    # this is the part that takes the longest.
    # how could this be done more quickly?
    gh_gdf = perform_within_intersection(gh_gdf=gh_gdf, focal_geo=focal_geo)

    gh_gdf[geo_id_fn] = geoid
    col_names = ['gh', geo_id_fn, 'overlap_percent']
    df = gh_gdf[col_names]

    del gh_df
    del gh_gdf

    return df


def mem_usage(pandas_obj):
    # courtesy of: https://www.dataquest.io/blog/pandas-big-data/
    if isinstance(pandas_obj,pd.DataFrame):
        usage_b = pandas_obj.memory_usage(deep=True).sum()
    else: # we assume if not a df it's a series
        usage_b = pandas_obj.memory_usage(deep=True)
    usage_mb = usage_b / 1024 # convert bytes to kilobytes
    return "{:03.2f} KB".format(usage_mb)


def shrink_df(df):

    old_mem = mem_usage(df)

    # shrink numeric columns
    df_int = df.select_dtypes(include=['int']).copy()
    converted_int = df_int.apply(pd.to_numeric, downcast='unsigned')

    df_float = df.select_dtypes(include=['float']).copy()
    converted_float = df_float.apply(pd.to_numeric, downcast='float')

    df_object = df.select_dtypes(include=['object']).copy()

    num_total_values = len(df_object)
    for i_col, col in enumerate(df_object.columns.tolist()):
        num_unique_values = len(df_object[col].unique())
        if num_unique_values / num_total_values <= 0.5 or \
            num_unique_values <= 10:
            df_object.loc[:, col] = df_object.loc[:, col].astype('category')

    # assemble our df
    df = pd.concat([df_object, converted_int, converted_float], axis = 1)

    new_mem = mem_usage(df)

    # print(old_mem, new_mem)

    del df_object
    del converted_int
    del converted_float

    return df


def get_coverage(f_values):
    """
    This function is called from a multiprocessor program. It finds the overlaps.
    :param f_values:
    :return:
    """

    geoid, geo_id_fn, geo, geohash_level, intermediate_path = f_values

    # create the shapely geometry from the wkt, if it's not already a wkt
    if isinstance(geo, str):
        geo = loads(geo)

    # get the delta values - width and height of the geohash cell
    xy_delta = get_delta_values(focal_geo=geo, geohash_level=geohash_level)

    # set up output paths and names
    if intermediate_path:
        intermediate_name = str(geoid) + '_points.json'
        intermediate_points_fpn = os.path.join(intermediate_path, intermediate_name)

        intermediate_name = str(geoid) + '_cells.json'
        intermediate_cells_fpn = os.path.join(intermediate_path, intermediate_name)
    else:
        intermediate_points_fpn = None
        intermediate_cells_fpn = None

    # get the xy lists - points!
    gh_df = get_x_y_point_lists(focal_geo=geo, gh_width=xy_delta[0],
                                gh_height=xy_delta[1], geohash_level=geohash_level,
                                intermediate_fpn=intermediate_points_fpn)

    # now enumerate to allocate
    geo_smash_list = allocate_geo(geoid=geoid, geo_id_fn=geo_id_fn, focal_geo=geo,
                                  gh_df=gh_df, intermediate_fpn=intermediate_cells_fpn)

    # here I could pickle the df?
    # downcast the type of the columns?

    # geo_smash_list = shrink_df(geo_smash_list)

    return geo_smash_list


def enumerate_geometry(geo_df, geo_id_fn, area_fn, geo_fn,
                       output_path, geo_id_desc, geohash_level=6,
                       intermediate_path=None, pool_option='multi'):
    """
    Enumerate a geodataframe. Create a mesh of geohashes that cover
    each geometry.
    :param geo_df: pandas dataframe. Features focal geometry stored as a WKT.
    :param geo_id_fn: string. Name of the field that identifies the geometry.
    :param area_fn: string. Name of the field that features the geometry's area.
    :param geo_fn: string. Name of the field that features the geometry.
    :param output_path: string. Full directory path where output data will be written.
    :param geo_id_desc: string. Textual description of the output.
    :param geohash_level: int. 1 - 12. The precision of the geohash.
    :param intermediate_path: string. Default is None.
    Path to place intermediate output data.
    :param pool_option: string. Default is 'multi' for parallel processing.
    If something else, items are produced sequentially. Can be useful for testing.
    :return: None
    """

    # first, sort by area so as to ensure even processing across the cores
    # we want to distribute to each core geometry of similar size.
    geo_df = geo_df.sort_values(by=area_fn, ascending=False)
    # get the number of features to send around in batches of ten
    n_features = len(geo_df)
    per_core = calc_tenths(n_items=n_features)

    # extract a list of ids. To be used for enumeration.
    id_list = geo_df[geo_id_fn].tolist()

    # the geometry that will be covered with geohashes
    geo_list = geo_df[geo_fn].tolist()

    # this will hold the lists to do multiprocessing in batches.
    process_list = []
    write_count = 0
    processor_count = cpu_count()
    for i_id, geoid in enumerate(id_list):

        geo = geo_list[i_id]  # geometry. In the form of a WKT.

        # put values in the list that will be unpacked during processing
        curr_list = [geoid, geo_id_fn, geo, geohash_level, intermediate_path]

        process_list.append(curr_list)

        if check_for_processing(n_features=n_features, i_id=i_id,
                                per_core=per_core):

            curr_progress = (i_id + 1) / n_features
            curr_progress = '{:.2%}'.format(curr_progress)
            print('...', curr_progress, 'complete.')

            # send to the multi-processor function.
            # or process in sequence.
            if pool_option == 'multi':
                print('...beginning multi-processing...')
                with Pool(processes=6) as p:
                    write_list = p.map(get_coverage, process_list)
                    # write_list = [p.map_async(get_coverage, process_list).get()[0]]
                    # write_list = [x.get()[0] for x in results]
                    # print(write_list)
            else:
                write_list = map(get_coverage, process_list)

            # concatenate it together
            output_df = pd.concat(write_list)
            del write_list

            # write the apportioned files to a text file.
            # but the proper write mode must be specified.
            output_file_name = geo_id_desc + '_gh.txt'
            output_fpn = os.path.join(output_path, output_file_name)

            if write_count == 0:
                # on the first write - create the text file.
                header_option = True
                write_option = 'w'
            else:
                # append on any other write
                header_option = False
                write_option = 'a'

            # sort - for testing
            col_names = output_df.columns.tolist()
            col_names = col_names[:-1]
            output_df = output_df.sort_values(by=col_names)

            output_df.to_csv(path_or_buf=output_fpn, sep='\t',
                             header=header_option, mode=write_option,
                             index=False)

            write_count += 1

            # remove the df from memory
            del output_df

            # reset the process list
            process_list = []

    del process_list

    return None


def get_process_list(db_path, db_name, geo_option='cbsa'):
    # do this for all blocks - by county. or by cbsa

    db_path_name = os.path.join(db_path, db_name)
    db_conn = sqlite3.connect(db_path_name)

    if geo_option == 'county':
        sql = 'select countyfips from countyfips;'
    else:
        sql = 'select cbsa_geoid from countyfips where cbsa_geoid is not null group by cbsa_geoid'

    geo_id_df = pd.read_sql(sql=sql, con=db_conn)
    col_names = geo_id_df.columns.tolist()

    process_list = geo_id_df.loc[:, col_names]

    del geo_id_df

    return process_list


def call_make_geohash_coverage(i_file_path, i_file_name, o_file_path, o_file_name,
                      dtype_dict, o_file_encoding='GeoJSON'):

    i_f_pn = os.path.join(i_file_path, i_file_name)
    df = pd.read_csv(filepath_or_buffer=i_f_pn, sep='\t', header=0,
                     dtype=dtype_dict)

    make_geohash_coverage(df, o_file_path, o_file_name,
                          o_file_encoding=o_file_encoding)

    return None


def make_geohash_coverage(df, o_file_path, o_file_name,
                          o_file_encoding='GeoJSON'):
    """
    Let's visualize the degree of areal weighting overlaps.
    :param i_file_path: string. Path to input file.
    :param i_file_name: string. Name of input file.
    :param o_file_path: string. Path to output file.
    :param o_file_name: string. Name of output file.
    :param dtype_dict: dict. Specifies the column types of the output
    geodataframe.
    :return: None
    """

    # get the geohash geometry
    gh_geo_series = df['gh']

    # use a lambda function to decode the geohash and then create a polygon
    def mpfb(x):
        return make_polygon_from_bounds(geohash.bbox(x))

    gh_geos = gh_geo_series.map(mpfb)

    # let's create a geodataframe
    wgs84_crs = 'epsg:4326'
    gh_gdf = gpd.GeoDataFrame(data=df, geometry=gh_geos, crs=wgs84_crs)

    # to disk
    o_file_pn = os.path.join(o_file_path, o_file_name)
    geometry_to_disk(gh_gdf=gh_gdf, output_fpn=o_file_pn,
                     o_file_encoding=o_file_encoding)

    return None


def run_it():

    ### todo: fix this stuff. It's janky.
    speedups.enable()

    """
    Function that does it all.
    :return:
    """

    # input directory
    i_db_path = 'H:/data/census_geography/states'
    i_db_name = 'tl_2018_us_state.db'
    db_pn = os.path.join(i_db_path, i_db_name)
    db_conn = sqlite3.connect(db_pn)

    # set up inputs.
    geo_id_fn = 'geoid'
    geo_fn = 'wkt'
    area_fn = 'xy_area'
    output_path = 'H:/project/fun_maps/data/geohash_grid_cells'
    geo_id_desc = 'states'
    intermediate_path = 'H:/project/fun_maps/data/geohash_grid_cells/intermediate'
    # intermediate_path = None
    geohash_level = 4
    pool_option = 'pool'

    sql = 'select * from tl_2018_us_state'

    # read in the data from a sqlite db
    geo_df = pd.read_sql(sql=sql, con=db_conn)

    # close the database connection
    db_conn.close()

    # let's fix this to store the apportioned overlaps and then the apportioned values.
    s_time = time.time()

    enumerate_geometry(geo_df=geo_df, geo_id_fn=geo_id_fn, area_fn=area_fn,
                       geo_fn=geo_fn, output_path=output_path, geo_id_desc=geo_id_desc,
                       geohash_level=geohash_level, intermediate_path=intermediate_path,
                       pool_option=pool_option)

    e_time = time.time()
    p_time = e_time - s_time
    p_time = round(p_time, 2)
    print(p_time, 'seconds.')

    i_file_path = output_path
    i_file_name = geo_id_desc + '_gh.txt'
    o_file_path = output_path
    o_file_name = geo_id_desc + '_gh.json'

    dtype_dict = {'gh': str, geo_id_fn: str, 'overlap_percent': float64}

    call_make_geohash_coverage(i_file_path=i_file_path, i_file_name=i_file_name,
                               o_file_path=o_file_path, o_file_name=o_file_name,
                               dtype_dict=dtype_dict)

    return None


if __name__ == '__main__':

    run_it()