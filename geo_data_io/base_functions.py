# useful functions to help with multiprocessing and enumeration

from formatting import pretty_format

def calc_tenths(n_items, n_splits=10, n_procs=6):
    """
    Simple function to determine how to divide up the work.
    We need to split the values so that each core gets the same amount.
    And we want to do this in tenths.
    :param n_items: Number of items in the list.
    :param n_splits: Number of groups to the divide the items into
    :param n_procs: Number of processors in the computer. Default is 8.
    :return: int. Number of items to send to each core.
    """

    # split in tenths
    split_amount = n_items / n_splits
    # each tenth then gets further split into 8 parts.
    # one part for each processor
    per_core = split_amount / n_procs
    # round to create an even number.
    per_core = round(per_core)

    # in case the rounding goes to zero, split amongst all processors.
    if per_core == 0:
        per_core = 1

    return per_core


def check_for_processing(n_features, i_id, per_core):
    """
    This is a function that instructs control flow. It's effectively a nice way
    of presenting and evaluating if statements. Much more elegant.
    :param n_features: Number of features to process.
    :param i_id: current iteration count.
    :param per_core: values to send to each core.
    :return: Boolean. continue_processing. If true, the algorithm continues.
    """

    if n_features == 1:
        # process if there is only one feature.
        continue_processing = True
    elif i_id > 0 and (i_id % (6 * per_core) == 0):
        # process if enough features have been accumulated.
        continue_processing = True
    elif i_id + 1 == n_features:
        # process if the total count of features has been reached.
        continue_processing = True
    else:
        # do not process.
        continue_processing = False

    return continue_processing


def progress_display(total_count, current_count, mod_value = 0.1,
                     verbose=True):
    ''' This function displays progress while enumerating items
    @param total_count: the total number of items to enumerate
    @param current_count: the current count of items enumerated
    @param mod_value: the threshold (0.XX) needed to display a value
    @param str_info: information to display to the user
    '''

    # the number of items that will be our divisor
    curr_mod = float(total_count) * float(mod_value)
    curr_mod = round(curr_mod, 0)
    if curr_mod == 0:
        curr_mod = 1

    output = None

    if total_count != 0:
        if (float(current_count) % float(curr_mod)) == 0.0 or current_count == total_count - 1 :
            curr_percent = float(current_count) / float(total_count)
            curr_percent = pretty_format(my_val=curr_percent, percent=True)
            if verbose:
                print('...', curr_percent, 'complete...')
                # print('...', pretty_format(current_count),'out of', pretty_format(total_count), '...')
            else:
                output = curr_percent

    return output


def n_combos(n_items):
    """ Calculate the number of unique combinations
    """
    total_items = (n_items * (n_items + 1)) / 2
    return total_items


def get_date_time_stamp(date_only = False, for_file_name=False):
    """ Formats a date time stamp as a string
    """
    import datetime
    d = datetime.datetime.now()
    if date_only:
        d_pretty = d.strftime("%Y-%m-%d")
    else:
        d_pretty = d.strftime("%Y-%m-%d %H:%M:%S")
    
    # clean up the format if stamping a file name with the time stamp
    if for_file_name:
        replace_list = [' ', ':', '-']
        for rs in replace_list:
            d_pretty = d_pretty.replace(rs, '_')

    return d_pretty

def get_n_processes_to_spin_up(list_of_values):
    """ This function helps with determining how many processes to start when
    parallel processing. 
    First, determine the number of cpus available. If there are more itmes in the list that needs
    processing, limit the number of cpus to the system reported number of cpus - 2
    Otherwise, spin up as many cpus as items in the list. 
    """
    import multiprocessing


    cpu_thread_count = multiprocessing.cpu_count
    if len(list_of_values) >= (cpu_thread_count - 2):
        n_processes = cpu_thread_count - 2
    else:
        n_processes = len(list_of_values)
    
    return(n_processes)


if __name__ == '__main__':

    progress_display(total_count=9643392, current_count=964339, mod_value=0.1)

    # print(n_combos(500))
