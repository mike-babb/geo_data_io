################################################################################
# GEOCODE ADDRESSES USING GOOGLE MAPS
# MIKE BABB
# babbm@uw.edu
# 2016 05 25
################################################################################

# standard libraries
import csv
import os
import sqlite3

# external
import googlemaps
import numpy as np
import pandas as pd

# load the API key
api_key_file_path = 'H:/Users/babbm/Documents/api_keys'
api_key_file_name = 'google_maps_api_key.txt'
api_key_fpn = os.path.join(api_key_file_path, api_key_file_name)

with open(file = api_key_fpn, mode = 'r') as my_api_key_file:
    api_key = my_api_key_file.readline()

# set the client key
gmaps = googlemaps.Client(key=api_key)

# list of match types to flag for manual inspection later
output_check = ["postal_code", "route", "locality political",
                "neighborhood political", ""]

def gmaps_geocode_addresses(address_string, address_hash):
    """
    Geocode an address using google maps
    :param address_string:
    :param address_hash:
    :return:
    """

    # send to the geocoder
    geocode_result = gmaps.geocode(address_string)

    formatted_address = ''
    lat = 0.0
    lng = 0.0
    geocode_type = ''
    address_is_good = 1
    if len(geocode_result) > 0:
        # print geocode_result
        # this has everything
        geocode_result = geocode_result[0]
        # the match type
        geocode_type = ' '.join(geocode_result['types'])

        # the lat long
        # print geocode_result.keys()
        if 'formatted_address' in geocode_result:
            formatted_address =  geocode_result['formatted_address']
        else:
            formatted_address = ''
        if 'geometry' in geocode_result:
            lat = geocode_result['geometry']['location']['lat']
            lng = geocode_result['geometry']['location']['lng']
        else:
            lat = 0.0
            lng = 0.0

        if geocode_type in output_check:
            address_is_good = 0

    # gather the results into a list and return
    output_list = [address_hash, formatted_address, lat, lng, geocode_type,
                  address_is_good]

    return output_list


def geocode_addresses(address_df, address_col_names, out_file_path,
                      out_file_name):
    """
    Read geocode addresses in a pandas dataframe.
    Each address is written to disk in a text file.
    Once all addresses are geocoded, the file results are stored in an Excel File
    and a SQLiteDB

    :param address_df:
    :param addressColumns:
    :param out_file_path:
    :param out_file_name:
    :return:
    """

    # clean up the output file name
    if out_file_name[-5:] == '.xlsx':
        out_file_name = out_file_name[:-5]

    if out_file_name[-3:] == '.db':
        out_file_name = out_file_name[:-3]

    # add the field for the address string
    address_df['address_string'] = ''


    # clean up the address columns
    for tcn in address_col_names:
        address_df[tcn] = address_df[tcn].fillna('').astype(str)
        # remove excess spaces
        address_df[tcn] = address_df[tcn].str.strip().str.upper()

        # count the number of strings
        double_string_count = address_df['address_string'].str.find(
            '  ').unique()

        # remove excess spaces inside the address string
        while len(double_string_count) > 1:
            address_df['address_string'] = address_df[
                'address_string'].str.replace('  ', ' ')

            double_string_count = address_df['address_string'].str.find(
                '  ').unique()

        # build the total address
        address_df['address_string'] += address_df[tcn] + ' '

    # hash the address_string
    address_df['address_hash'] = address_df['address_string'].map(hash)

    # get a list of tuples of the address_hash and the full address_string
    output_dict = {}

    # write each line to an intermediary output.
    # Sometimes the geocoding process might fail.
    # But we want to capture the intermediary output
    int_output_file = out_file_name + '.txt'
    int_output_file_pn = os.path.join(out_file_path, int_output_file)
    int_output_file = open(int_output_file_pn, 'wb')
    csvwriter = csv.writer(int_output_file, delimiter = '\t')

    rowcount = 0
    funny_characters = 0
    for row in address_df.iterrows():

        address_string = row[1]['address_string']
        address_hash = row[1]['address_hash']

        # GEOCODE
        try:
            # first try to geocode. catch the error.
            # accounts for an error with the geocoding process
            output_list = gmaps_geocode_addresses(address_string, address_hash)
        except:
            print('Address could not be geocoded.')
            output_list = [address_hash, '',-1,-1,'',0]

        # WRITE THE INTERMEDIATE RESULTS TO DISK
        # THIS MIGHT ALSO FAIL BECAUSE OF FUNNY CHARACTERS IN THE ADDRESS
        try:
            csvwriter.writerow(output_list)
        except:
            # we might not need to reset the output list.
            # If it geocoded fine, but the text has funny characters
            output_list_temp = [address_hash, '', -1, -1, '', 0]
            csvwriter.writerow(output_list_temp)
            funny_characters += 1

        # STORE THE RESULTS IN THE DICTIONARY
        output_dict[address_hash] = output_list[1:]
        # display progress
        rowcount += 1
        if (float(rowcount) % 1000.0) == 0:
            print('Geocoded', '{:,}'.format(rowcount), 'records...')
            # time.sleep(.1)

    # close the intermediary file
    int_output_file.close()

    ####
    # WRITE THE FULL GEOCODED RESULTS BACK OUT TO DISK
    ####
    col_names = ['formattedAddress', 'lat', 'lng', 'geocode_type',
                       'address_is_good']

    for i_tcn, tcn in enumerate(col_names):
        temp_lambda = lambda x: output_dict[x][i_tcn]
        address_df[tcn] = address_df['address_hash'].map(temp_lambda)

    # write it out to excel
    # excel should be able to handle the funny characters
    o_excel_file_name = out_file_name + '.xlsx'
    output_file_path_name = os.path.join(out_file_path, o_excel_file_name)
    writer = pd.ExcelWriter(output_file_path_name, engine='xlsxwriter')
    address_df.to_excel(excel_writer=writer, sheet_name = 'geocoded',
                       index=False,engine='xlsxwriter')
    # writer.save()
    writer.close()

    # write it out to SQLite
    o_db_name = out_file_name + '.db'
    output_file_path_name = os.path.join(out_file_path, o_db_name)
    o_db_conn = sqlite3.connect(output_file_path_name)

    # if there are funny characters, we need to specify a different connection
    if funny_characters > 0:
        o_db_conn.text_factory = str
    address_df.to_sql(name=out_file_name,con=o_db_conn,if_exists='replace',
                     index=False)
    o_db_conn.close()


def excel_file_geocode(i_excel_file_path, i_excel_file_name, i_excel_sheet_name,
                       address_col_names, out_file_path, out_file_name):
    """
    Read an excel file and geocode the addresses
    :param i_excel_file_path:
    :param i_excel_file_name:
    :param i_excel_sheet_name:
    :param address_col_names:
    :return:
    """

    i_excel_file_pn = os.path.join(i_excel_file_path, i_excel_file_name)

    address_df = pd.read_excel(io=i_excel_file_pn, sheetname=i_excel_sheet_name)

    # geocode it.
    geocode_addresses(address_df=address_df,address_col_names=address_col_names,
                      out_file_path=out_file_path,out_file_name=out_file_name)


    return None


def sqlite_geocode(db_path, db_name, sql, address_col_names, out_file_path,
                   out_file_name):
    """

    :param db_path:
    :param db_name:
    :param sql:
    :param address_col_names:
    :param out_file_path:
    :param out_file_name:
    :return:
    """

    # connect to the db and get the data
    db_path_name = os.path.join(db_path, db_name)
    db_conn = sqlite3.connect(db_path_name)
    address_df = pd.read_sql(sql=sql, con=db_conn)
    db_conn.close()

    # geocode it.
    geocode_addresses(address_df=address_df,
                      address_col_names=address_col_names,
                      out_file_path=out_file_path,
                      out_file_name=out_file_name)

    return None


if __name__ == '__main__':


    address_string = '4114 University Way NE Seattle WA 98105'
    address_hash = hash(address_string)

    testo = gmaps_geocode_addresses(address_string, address_hash)

    print(testo)
