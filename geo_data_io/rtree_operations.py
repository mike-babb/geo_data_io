################################################################################
# CREATE AND LOAD AN RTREE FROM A PANDAS DATAFRAME
################################################################################

# standard libraries
import itertools
import time

# external
import numpy as np
import ogr
import pandas as pd

from rtree import index

# custom
from df_operations import into_dict
from fc_df_spatial import *


# TODO: will need a function that loads from a single line. i.e. data too
# TODO: replace with shapely
# big to fit into memory
# put a function in here to create a grid of cells for use with the rtree.
# only do the point in polygon operation as a last resort

def intersection_test(source_geo, target_geo):
    """
    Simple, brute-force to see if one geometry touches another using OGR
    :param source_geo:
    :param target_geo:
    :return:
    """

    # integer to hold the outcome
    if source_geo.intersects(target_geo):
        intersects = 1
    else:
        intersects = 0

    return intersects


def pip_df(df, id_fn, wkt_fn, target_geo, outcome_fn):
    """
    Wrapper function for the point in polygon function above
    :param df: pandas dataframe
    :param id_fn: string. Name of the field that uniquely identifies each row.
    :param wkt_fn: string. Name of the field with wkt.
    :param target_geo: string or preferably, OGR geometry.
    :return: pandas dataframe
    """
    # TODO use shapely for this portion.

    temp_dict = {}

    def _pip_it(row, id_fn, wkt_fn, target_geo):
        """
        internal function used to apply to the dataframe
        :param row:
        :param id_fn:
        :param wkt_fn:
        :param target_geo:
        :return:
        """

        id_value = row[id_fn]  # id of the row
        source_geo = row[wkt_fn]  # point wkt
        source_geo = ogr.CreateGeometryFromWkt(
            source_geo)  # convert to geometry

        intersects = intersection_test(
            source_geo, target_geo)  # perform the calculation
        temp_dict[id_value] = intersects  # store the outcome

    def temp_lambda(x): return _pip_it(
        x, id_fn, wkt_fn, target_geo)  # define the function
    df.apply(temp_lambda, 1)  # apply it

    # recover the values
    def temp_lambda(x): return temp_dict[x]
    df[outcome_fn] = df[id_fn].map(temp_lambda)

    return df


def create_and_load_index(df, id_fn, x_min_fn, y_min_fn, x_max_fn, y_max_fn,
                          output_file_path_name=None):
    """
    This accepts either point geometry or polygon geometry.

    :return: rtree index
    """
    if output_file_path_name:
        idx = index.Index(output_file_path_name)
    else:
        idx = index.Index()

    def load_index(row):
        """
        Read from a pandas dataframe and load into the rtree
        :param row:
        :return:
        """

        # ID value
        id_val = row[id_fn]
        # bbox values
        x_min = row[x_min_fn]
        y_min = row[y_min_fn]
        x_max = row[x_max_fn]
        y_max = row[y_max_fn]

        # insert into the index
        # let's convert the values to a string
        idx.insert(id_val, (x_min, y_min, x_max, y_max))

        return None

    df.apply(load_index, 1)

    if output_file_path_name:
        idx.close()
        idx = None

    return idx


def single_line_query(query_tuple, idx, n_features=1,
                      query_isinstance='intersection'):

    if query_isinstance == 'intersection':
        # return the n_features intersected with the item
        results = list(idx.intersection(query_tuple))
    if query_isinstance == 'nearest':
        results = list(idx.nearest(query_tuple, n_features))

    # if there are multiple intersections, pick the first one.
    if results and query_isinstance == 'intersection':
        pass
    elif results and query_isinstance == 'nearest':
        if n_features > 1:
            results = results[:n_features]
        else:
            results = results[0]
    else:
        results = [-1]

    return results


def query_rtree(id_fn, id_val, x_min, y_min, x_max, y_max, idx,
                n_features=1, query_isinstance='intersection'):
    """
    Read from a pandas dataframe and query the rtree
    :param row:
    :return:
    """

    query_tuple = (x_min, y_min, x_max, y_max)

    results = single_line_query(query_tuple=query_tuple, idx=idx,
                                n_features=n_features,
                                query_isinstance=query_isinstance)

    id_list = [id_val] * len(results)
    output_dict = {'ref_id': id_list, 'non_ref_id': results}

    return output_dict


def determine_self_intersections(idx, df, id_fn, x_min_fn, y_min_fn,
                                 x_max_fn, y_max_fn):
    """
    For a given index, determine the self intersections
    :param df:
    :param id_fn:
    :param geo_fn:
    :param idx:
    :return:
    """

    # query the rtree. This used to be a lambda function that
    # is no longer kosher. So, an actual named function.
    # this way the user can still specif names of the bounding boxes
    # field names
    def do_query(row):
        """
        Internal function that
        :param row:
        :return:
        """

        id_val = row[id_fn]
        x_min = row[x_min_fn]
        y_min = row[y_min_fn]
        x_max = row[x_max_fn]
        y_max = row[y_max_fn]

        output_dict = query_rtree(id_fn=id_fn, id_val=id_val, x_min=x_min,
                                  y_min=y_min, x_max=x_max, y_max=y_max,
                                  idx=idx)
        return output_dict

    # apply the query r-tree function.
    intersection_list = df.apply(do_query, 1)
    # convert the resulting series to a list
    intersection_list = intersection_list.tolist()
    # convert each dictionary to a dataframe within the list.
    # 2018 03 20: I am honestly not sure why I can't create a dataframe from the list
    # of dictionaries.
    intersection_df_list = [pd.DataFrame.from_dict(
        data=x) for x in intersection_list]
    # delete the list
    del intersection_list

    # build the output dictionary
    adjacency_df = pd.concat(intersection_df_list)
    # delete the list of dataframes
    del intersection_df_list

    # let's do some joins
    # joins? That will blow up memory. But will it?
    geo_dict = into_dict(df=df, key_fn=id_fn, value_fns='geometry',
                         with_value_field_names=False)

    def do_intersection_test(row):
        ref_id = row['ref_id']
        nonref_id = row['non_ref_id']

        source_geo = geo_dict[ref_id]
        target_geo = geo_dict[nonref_id]

        # 0: no intersection
        # 1: intersection
        # 2: self-intersection

        if ref_id == nonref_id:
            result = 2
        else:
            result = intersection_test(source_geo, target_geo)

        return result

    adjacency_df['rtree_adjacent'] = 1
    adjacency_df['geo_adjacent'] = adjacency_df.apply(do_intersection_test, 1)

    # set the proper ids
    id_num_fn = id_fn[:]
    id_fn = id_fn[:-4]
    col_names = [id_fn, id_num_fn]
    id_df = df[col_names]

    col_names = ['ref_' + id_fn, 'ref_id']
    id_df.columns = col_names
    adjacency_df = pd.merge(left=adjacency_df, right=id_df)

    col_names = ['non_ref_' + id_fn, 'non_ref_id']
    id_df.columns = col_names
    adjacency_df = pd.merge(left=adjacency_df, right=id_df)

    # reorder the columns
    col_names = ['ref_' + id_fn, 'non_ref_' +
                 id_fn, 'rtree_adjacent', 'geo_adjacent']
    adjacency_df = adjacency_df[col_names]

    col_names = ['ref_id', 'non_ref_id', 'rtree_adjacent', 'geo_adjacent']
    adjacency_df.columns = col_names

    return adjacency_df


def brute_force_adjacency_check(geo_dict):
    # list to hold the county adjacency
    adjacency_dict = {}
    time_dict = {}

    for ref_id, ref_geo in geo_dict.items():

        start_time = time.time()
        # combine the pairs to see what has been examined

        for non_ref_id, non_ref_geo in geo_dict.items():

            refID_nonRefID_ij = ref_id + '_' + non_ref_id
            refID_nonRefID_ji = non_ref_id + '_' + ref_id

            # only proceed if the feature pair has not been examined
            # XY = YX
            if refID_nonRefID_ij not in adjacency_dict and \
                    refID_nonRefID_ji not in adjacency_dict:

                # a county is adjacent to itself
                if refID_nonRefID_ij == refID_nonRefID_ji:
                    adjacency = 1
                else:
                    adjacency = intersection_test(ref_geo, non_ref_geo)

                adjacency_dict[refID_nonRefID_ij] = adjacency
                adjacency_dict[refID_nonRefID_ji] = adjacency

                # and now, at this point, I could add some code to check for the length of the shared border

        time_end = time.time()
        time_diff = time_end - start_time
        time_dict[ref_id] = time_diff

        # increment the count
        if (len(time_dict) % 10) == 0:
            print_line = 'Processing completed on: ' + str(len(time_dict))
            print(print_line)

            # # build the time df and export some summary statistics
            # time_df = pd.DataFrame.from_dict(data=time_dict, orient='index')
            # time_df = time_df.reset_index()
            # temp_cn = ['refCountyFIPS', 'nSeconds']
            # time_df.columns = temp_cn
            #
            # # print the total time elapsed and the average time per county.
            # # it should be going down.
            # print time_df['nSeconds'].sum(), time_df['nSeconds'].mean()

    # build the adjacency df
    adjacency_df = pd.DataFrame.from_dict(data=adjacency_dict, orient='index')
    del adjacency_dict
    adjacency_df = adjacency_df.reset_index()
    temp_cn = ['ref_id_non_ref_id', 'adjacency']
    adjacency_df.columns = temp_cn

    # build the time df
    time_df = pd.DataFrame.from_dict(data=time_dict, orient='index')
    del time_dict
    time_df = time_df.reset_index()
    temp_cn = ['refCountyFIPS', 'nSeconds']
    time_df.columns = temp_cn

    return adjacency_df, time_df


def determine_adjacency(input_db_path, input_db_name, sql, id_fn,
                        geo_fn, x_min_fn, x_max_fn, y_min_fn, y_max_fn,
                        output_db_path, output_db_name, output_table_name):
    # get the county data!
    county_gdf = load_geo_data_from_sqlite(sql=sql, db_path=input_db_path,
                                           db_name=input_db_name)

    # generate a unique, numeric value for the field name.
    id_num_fn = id_fn + '_num'
    county_gdf[id_num_fn] = county_gdf[id_fn].map(hash)

    # create the index
    print('...creating and loading the index...')
    idx = create_and_load_index(df=county_gdf, id_fn=id_num_fn,
                                x_min_fn=x_min_fn, y_min_fn=y_min_fn,
                                x_max_fn=x_max_fn,
                                y_max_fn=y_max_fn)

    # determine self intersections
    print('...determining adjacency...')
    adjacency_df = determine_self_intersections(idx=idx, df=county_gdf,
                                                id_fn=id_num_fn,
                                                x_min_fn=x_min_fn,
                                                y_min_fn=y_min_fn,
                                                x_max_fn=x_max_fn,
                                                y_max_fn=y_max_fn)

    return adjacency_df


if __name__ == '__main__':

    county_list_db_path = 'H:/data/census_geography/county'
    county_list_db_name = 'tl_2017_us_county.db'
    county_list_db_path_name = os.path.join(
        county_list_db_path, county_list_db_name)

    # output sqlite db
    output_db_path = '/project/geog542_2013/county_adjacency'
    output_db_name = 'countyAdjacency.db'
    output_table_name = 'county_adjacency_rtree'

    sql = 'select GEOID, X_MIN, Y_MIN, X_MAX, Y_MAX, ' \
          'wkt from tl_2017_us_county where statefp = \'53\';'

    stime = time.time()
    adjacency_df = determine_adjacency(input_db_path=county_list_db_path,
                                       input_db_name=county_list_db_name,
                                       sql=sql, id_fn='GEOID', geo_fn='wkt',
                                       x_min_fn='X_MIN', x_max_fn='X_MAX', y_min_fn='Y_MIN',
                                       y_max_fn='Y_MAX', output_db_path=output_db_path,
                                       output_db_name=output_db_name, output_table_name=output_table_name)

    etime = time.time()
    ptime = etime - stime
    print('Took: ', round(ptime, 2), 'seconds.')

    print(adjacency_df.head())
    print(adjacency_df['rtree_adjacent'].sum(),
          adjacency_df['geo_adjacent'].sum(), len(adjacency_df))
