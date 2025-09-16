# the data have rotted.
# need to examine each character.

# libraries
import os
import string


def check_characters(curr_line, good_char_set = 'default'):

    # \x0b: vertical tab
    #
    # \x0c: Form Feed

    if good_char_set.lower() == 'default':
        default_good_char_set = string.printable

        # this is a variable that DF columns are compared against for non-numeric
        # it will be expanded over time
        non_numeric_set = ('.', '*', '+')
        good_char_set = default_good_char_set


    good_char_set = set(list(good_char_set))
    # turn the line into a set and perform a difference
    curr_line_set = set(list(curr_line))
    diff_set = curr_line_set.issubset(good_char_set)

    return diff_set


def call_check_characters(file_pn, skip_lines = 0, good_char_set = 'default'):
    '''From time to time, data rot. This makes it difficult to read. Let
    alone analyze. So we have to check each character.
    A loop within in a loop.
    Once the bad characters on each line are discovered, use notepad++
    to edit / remove bad characters.
    Displays the file name and line number of the bad character.
    @param: file_pn: string. Path and name of a file.
    @param: skip_lines: integer. Sometimes the header is good.
    Skip the number of lines featuring the hearder. Default is not to skip.
    @param: good_char_set: Set. Set with good characters.
    Default is set within the function.
    '''

    # usually python is pretty good about opening up files.
    # the problem is when the encoding is really strange.
    # data come from different computers with different Operating Systems
    # from different years. Good luck.
    # I wish I had a good solution to this, but I don't.
    # the csv package does provide some assistance with encodings.


    # open the file
    # let's print the name of the file:
    file_path, file_name = os.path.split(file_pn)

    my_file = open(file_pn, 'r')

    curr_line_count = 1
    for line in my_file:
        # skip the header
        if curr_line_count > skip_lines:

            diff_set = check_characters(curr_line=line,good_char_set=good_char_set)

            if diff_set:
                pass
            else:
                print(file_name, curr_line_count)
        # increment the count
        curr_line_count += 1

    # close the file
    my_file.close()

    return None


if __name__ == '__main__':
    # working directory
    my_dir = 'H:/Project/geog542_2013/IRS_County_To_County_Migration/tables/2004_and_later/original/2008'

    # walk a directory
    file_name_list = os.listdir(my_dir)
    file_name_list.sort() # sort the list
    #print len(file_name_list) # how many files did we find?
    #print file_name_list[:10] # let's look at a few

    # send to the function.
    for file_name in file_name_list:
        fpn = os.path.join(my_dir, file_name)
        print(fpn)
        call_check_characters(file_pn=fpn, skip_lines = 0)


