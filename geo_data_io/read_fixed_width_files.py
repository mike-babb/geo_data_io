#
# READ IN A FIXED WIDTH FILE
# RETURN A DATA FRAME
#

# standard libraries
import os

# external libraries
import numpy as np
import pandas as pd


def remove_excess_spaces(my_string):
    """
    Simple Function to remove excess spaces from a string
    :param my_string:
    :return:
    """
    # remove excess spaces
    if len(my_string) > 1:
        while my_string.find('  ') != -1:
            my_string = my_string.replace('  ', ' ')

    # remove space from the beginning of the line
    my_string = my_string.strip()

    return my_string


def data_type_change(curr_column, new_dt, na_values_dict):
    # remove string characters
    if new_dt != str:
        # remove spaces - if any are left
        curr_column = curr_column.str.replace(' ', '')
        # remove the comma
        curr_column = curr_column.str.replace(',', '')
        # now, just in case there are funny values for NA
        if na_values_dict:
            for na_value, fill_na_value in na_values_dict.items():
                curr_column = curr_column.str.replace(na_value, fill_na_value)

        curr_column = curr_column.astype(new_dt)

    return curr_column


def read_fixed_width_file(input_file_path, input_file_name,
                          column_widths, new_column_names,
                          data_line_check_characters = None, change_data_type = False,
                          new_data_type_dict = None, na_values_dict = None):
    """
    Read in a fixed width file as a delimited file.
    Use string splitting techniques to split up the rows
    Return a formatted data frame.
    Remove excess rows based on a series of check characters
    :param input_file_path:
    :param input_file_name:
    :param data_line_check_characters:
    :param column_widths:
    :param new_column_names:
    :param change_data_type:
    :param new_data_type_dict:
    :param na_values:
    :param fill_na_values:
    :return:
    """

    input_fpn = os.path.join(input_file_path, input_file_name)
    col_names = ['temp_column']
    data_type_dict = {'temp_column': str}

    # assume the lines are delimited by a tild.
    print_line = 'Reading in ' + input_fpn + ' ...'
    print(print_line)

    df = pd.read_csv(filepath_or_buffer = input_fpn,
                     sep = '~', engine = 'c', dtype = data_type_dict,
                     header = None, names = col_names)

    # remove non-datalines
    if data_line_check_characters:
        df = df[df['temp_column'].str[:2] == data_line_check_characters]

    print_line = '...Splitting strings...'
    print(print_line)

    data_dict = {}

    for i_cw, cw in enumerate(column_widths[:-1]):
        # column widths
        s_start = cw
        s_end = column_widths[i_cw + 1]

        # column names
        cn = new_column_names[i_cw]
        curr_split = df['temp_column'].str[s_start:s_end]
        data_dict[cn] = curr_split

        curr_split = None

    # split the string to get the final column
    s_start = column_widths[-1]
    cn = new_column_names[-1]
    data_dict[cn] = df['temp_column'].str[s_start:]

    new_df = pd.DataFrame.from_dict(data=data_dict)

    # delete stuff to free up memory
    del df
    del data_dict

    # now, let's strip the whitespace from the string
    print_line = '...Removing whitespace...'
    print(print_line)
    new_df = new_df.applymap(str.strip)

    if change_data_type:
        print_line = '...Changing data types...'
        print(print_line)
        for cn, new_dt in new_data_type_dict.items():
            new_df[cn] = data_type_change(curr_column=new_df[cn],
                                          new_dt=new_dt,
                                          na_values_dict=na_values_dict)

    # reorder the columns
    new_df = new_df[new_column_names]
    return new_df


if __name__ == '__main__':
    # read in the geo file
    geo_file_path = 'H:/data/census_data/census2000/Washington/geo'
    geo_header_file_name = 'geo_header.txt'

    geo_file_list = os.listdir(geo_file_path)
    geo_file_list = [i for i in geo_file_list if i != geo_header_file_name]

    geo_header_fpn = os.path.join(geo_file_path, geo_header_file_name)
    geo_header_df = pd.read_csv(filepath_or_buffer=geo_header_fpn, sep='\t', header=None,
                                names=['col_name', 'col_width'])

    geo_col_names = geo_header_df['col_name'].tolist()
    geo_header_df['col_width'] = geo_header_df['col_width'] - 1
    geo_col_width = geo_header_df['col_width'].tolist()

    # load the geo files
    # this is very lazy - but just get it done now.
    geo_file_name = geo_file_list[0]
    geo_df = read_fixed_width_file(input_file_path=geo_file_path, input_file_name=geo_file_name,
                                   data_line_check_characters=None,
                                   column_widths=geo_col_width, new_column_names=geo_col_names)
    print(geo_df.head())

    testo = sorted(geo_df['SUMLEV'].unique().tolist())
    print(testo)

    geo_df = geo_df.loc[geo_df['SUMLEV'] == '140', :]
    print(len(geo_df))
