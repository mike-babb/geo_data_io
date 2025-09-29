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


# calculate an r-squared value
def calc_r_squared(observed, estimated):
    import numpy as np
    r = np.corrcoef(observed, estimated)[0][1]
    r2 = r ** 2
    return r2

# calculate the root mean square error
def calc_RMSE(observed, estimated):
    import numpy as np
    res = (observed - estimated) ** 2
    rmse = round(np.sqrt(np.mean(res)), 3)

    return rmse



if __name__ == '__main__':

    print(get_CAGR(5, 10, 2))

    print(percent_change(5, 10))