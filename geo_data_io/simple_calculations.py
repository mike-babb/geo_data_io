# Simple functions that aren't worth recreating everytime

def get_CAGR(b_value, e_value, n_years, round_val = True):
    """
    Calculate the compound annual growth rate
    :param b_value:
    :param e_value:
    :param n_years:
    :return:
    """

    cagr = ((e_value/b_value)**(1.0/n_years)) - 1
    if round_val:
        cagr = round(cagr, 4)

    return cagr

def percent_change(b_value, e_value, round_val = True):
    """
    Calculate the percent change between two values
    :param b_value:
    :param e_value:
    :return:
    """

    per_change = (e_value - b_value) / b_value

    if round_val:
        per_change = round(per_change, 4)

    return per_change

if __name__ == '__main__':

    print(get_CAGR(5, 10, 2))

    print(percent_change(5, 10))