# -*- coding: utf-8 -*-
"""
Created on Mon Jan 10 21:40:22 2022

@author: ENGITO
"""

import pandas as pd
def read_data_from_Friedrichshagencsv():

    csv_KW = pd.read_csv('data/28-012112fb840e.csv',
                         parse_dates=["Zeit_Datetime"])
    csv_RT = pd.read_csv('data/28-012113544079.csv',
                         parse_dates=["Zeit_Datetime"])


    df_Friedrichshagen = pd.DataFrame(list(zip( csv_KW['Zeit_Datetime'], csv_KW['measurement_value'], csv_RT['measurement_value'])), columns=['Timestamp', 1, 2])

    return df_Friedrichshagen


def read_new_data_from_Friedrichshagencsv():

    csv_KW = pd.read_csv('data/28-012112fb840e_new.csv',
                         parse_dates=["Zeit_Datetime"])
    csv_RT = pd.read_csv('data/28-012113544079_new.csv',
                         parse_dates=["Zeit_Datetime"])


    df_Friedrichshagen = pd.DataFrame(list(zip( csv_KW['Zeit_Datetime'], csv_KW['measurement_value'], csv_RT['measurement_value'])), columns=['Timestamp', 1, 2])

    return df_Friedrichshagen
