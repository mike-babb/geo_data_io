# export data from a Microsoft Access Database

# standard libraries
import os

# external libraries
import pandas as pd
import pyodbc


def load_data_from_ms_access(sql, db_path, db_name):
    """
    Load data from MS Access into a pandas dataframe
    Must download the Microsoft Access Database Engine 2010 Redistributable from:
    https://www.microsoft.com/en-us/download/confirmation.aspx?id=13255
    :param db_path:
    :param db_name:
    :return:
    """

    db_path_name = os.path.join(db_path, db_name)
    driver = '{Microsoft Access Driver (*.mdb, *.accdb)}'
    con = pyodbc.connect('DRIVER={};DBQ={}'.format(driver, db_path_name))

    df = pd.read_sql(sql, con)

    return df

if __name__ == '__main__':

    db_path = 'H:/data/census_data/census2000'
    db_name = 'SF3.accdb'
    sql = 'select * from SF30001'

    df = load_data_from_ms_access(sql, db_path, db_name)
    print(df.head())