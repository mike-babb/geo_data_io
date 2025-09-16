# simple functions that format strings.

def remove_commas(my_line, comma_sequence):
    """
    Remove double commas and excess spaces
    :param my_line: str. a line of text
    :param comma_sequence:
    :return:
    """

    if comma_sequence == 1:
        # insert a comma before the state abbreviation
        # my_line = my_line[:34] + ',' + my_line[34:]

        my_line = my_line.replace('  ', ',') # convert double spaces to commas
        # convert double commas to single commas
        while my_line.find(',,') != -1:
            my_line = my_line.replace(',,', ',')
        # remove the first comma
        if my_line[0] == ',':
            my_line = my_line[1:]

    if comma_sequence == 2:
        # replace excess spaces after a comma
        my_line = my_line.replace(', ', ',')
        # replace excess spaces before a comma
        my_line = my_line.replace(' ,', ',')
        # replace double commas
        my_line = my_line.replace(',,', ',')

    return(my_line)


# remove characters that cause all hell
def remove_new_line_characters(temp_line):
    """
    Remove the new line characters from a file.
    :param temp_line:
    :return:
    """
    remove_list = ['\n', '\r', '\r\n']
    for ii in remove_list:
        temp_line = temp_line.replace(ii, '')

    return(temp_line)


def pretty_format(my_val, percent=False):
    """

    :param my_val:
    :return:
    """

    if percent:
        my_val = round(my_val, 4)
        # with a percent sign
        my_val = '{:.2%}'.format(my_val)

    if type(my_val) is int:
        my_val = '{:,}'.format(my_val)
    elif type(my_val) is float:
        my_val = round(my_val, 2)
        my_val = '{:,.2}'.format(my_val)
    else:
        my_val = my_val

    return my_val


if __name__ == '__main__':
    my_line = 'bubbles\n'
    print(my_line)
    my_line = remove_new_line_characters(my_line)
    print(my_line)

    a = 1211
    testo = pretty_format(a, percent=False)
    print(testo)
