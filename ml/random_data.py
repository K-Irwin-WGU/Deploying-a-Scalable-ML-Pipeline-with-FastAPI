
# Import Packages

import os

import math
import random

import pandas as pd
import numpy as np


############################### Helper Functions ############################


def get_cat1_values(row):
    if row['category_1'] == 0:
        return 'Alpha'
    elif row['category_1'] == 1:
        return 'Alpha'
    else:
        return 'Beta'


def get_cat2_values(row):
    if row['category_2'] == 0:
        return 'A'
    elif row['category_2'] == 1:
        return 'B'
    elif row['category_2'] == 2:
        return 'C'
    elif row['category_2'] == 3:
        return 'D'
    elif row['category_2'] == 4:
        return 'E'
    else:
        return 'F'


def get_cat3_values(row):
    if row['category_3'] == 0:
        return 'North'
    elif row['category_3'] == 1:
        return 'North'
    elif row['category_3'] == 2:
        return 'South'
    elif row['category_3'] == 3:
        return 'East'
    else:
        return 'West'


def get_cat4_values(row):
    if row['category_4'] == 0:
        return 'Up'
    elif row['category_4'] == 1:
        return 'Up'
    elif row['category_4'] == 2:
        return 'Up'
    else:
        return 'Down'


def get_cat5_values(row):
    if row['category_5'] == 0:
        return 'Front'
    elif row['category_5'] == 1:
        return 'Back'
    elif row['category_5'] == 2:
        return 'Left'
    elif row['category_5'] == 3:
        return 'Right'
    else:
        return 'Center'


def get_cat6_values(row):
    if row['category_6'] == 0:
        return 'Black'
    elif row['category_6'] == 1:
        return 'Red'
    elif row['category_6'] == 2:
        return 'Orange'
    elif row['category_6'] == 3:
        return 'Yellow'
    elif row['category_6'] == 4:
        return 'Green'
    elif row['category_6'] == 5:
        return 'Blue'
    elif row['category_6'] == 6:
        return 'Indigo'
    elif row['category_6'] == 7:
        return 'Violet'
    else:
        return 'White'


def get_target_values(row):
    if row['target'] == 0:
        return 'foo'
    elif row['target'] == 1:
        return 'foo'
    else:
        return 'bar'





############################### Main Function ###############################


def random_data():

    '''
    Generates a DataFrame with random data for testing purposes.
    '''

    random.seed(42)    
    random_integer = random.randint(20000, 50000)

    list_1 = [random.randint(0, 2) for _ in range(random_integer)]
    list_2 = [random.randint(0, 5) for _ in range(random_integer)]
    list_3 = [random.randint(0, 4) for _ in range(random_integer)]
    list_4 = [random.randint(0, 3) for _ in range(random_integer)]
    list_5 = [random.randint(0, 5) for _ in range(random_integer)]
    list_6 = [random.randint(0, 9) for _ in range(random_integer)]
    list_7 = list(range(1,(random_integer + 1)))
    list_8 = list(range(random_integer,0,-1))
    list_9 = [random.randint(20, 80) for _ in range(random_integer)]
    list_10 = [random.randint(0, 2) for _ in range(random_integer)]

    data = {
        'category_1': list_1,
        'category_2': list_2,
        'category_3': list_3,
        'category_4': list_4,
        'category_5': list_5,
        'category_6': list_6,
        'feature_1': list_6,
        'feature_2': list_7,
        'feature_3': list_8,
        'feature_4': list_9,
        'target': list_10
    }
    
    df = pd.DataFrame(data)
    
    df['category_1'] = df.apply(get_cat1_values, axis=1)
    df['category_2'] = df.apply(get_cat2_values, axis=1)
    df['category_3'] = df.apply(get_cat3_values, axis=1)
    df['category_4'] = df.apply(get_cat4_values, axis=1)
    df['category_5'] = df.apply(get_cat5_values, axis=1)
    df['category_6'] = df.apply(get_cat6_values, axis=1)
    df['target'] = df.apply(get_target_values, axis=1)
    
    return df







# End of Page
