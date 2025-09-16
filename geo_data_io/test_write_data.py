# creates a dictionary of dictionaries from pandas dataframe.
# each row/field is a key value pair in the inner dictionary.

# standard libraries
import os
import sqlite3
import time

# external libraries
import numpy as np
import pandas as pd

# custom
from df_operations import write_data

# create data
my_df = {'a': [1, 2], 'b': [3, 4]}
my_df = pd.DataFrame(data=my_df)
file_path = 'H:/temp'
file_name = 'test.db'
table_name = 'test_table'

# export to sqlite
description = 'test writing data to db'
write_data(obj_to_store=my_df, file_path=file_path, file_name=file_name,
           table_name=table_name,
           description=description)
  
# export to excel
file_path = 'H:/temp'
file_name = 'test.xlsx'
description = 'test writing data to excel'
write_data(obj_to_store=my_df, file_path=file_path, file_name=file_name,
           table_name='', description=description)

# export to csv
file_path = 'H:/temp'
file_name = 'test.csv'
description = 'test writing data to csv'

write_data(obj_to_store=my_df, file_path=file_path, file_name=file_name,
           table_name='', description=description)

# export to text
file_path = 'H:/temp'
file_name = 'test.txt'
description = 'test writing data to text'

write_data(obj_to_store=my_df, file_path=file_path, file_name=file_name,
           table_name='', description=description)


# print(os.path.realpath(__file__))
