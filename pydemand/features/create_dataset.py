import numpy as np
import pandas as pd
import logging

from .climate import get_temperature
from .gdp import get_gdp
from .population import get_population
from .hdi import get_hdi



def to_trainingsset(df):
    """
    Creates an example from each datapoint.
    Features are either based on datetime (df.index) or
    column name, i.e. the country, or both
    
    Features extracted from datetime
    - hour
    - weekday
   
    Features extracted from country
    - population
    - ...
    
    Features extracted from both
    - temperature
    - ...
    """ 

    dfcountry_list = list()
    for country in df.columns:


        country_df = df[[country]]
        if dfcountry_list:
            country_df[["hour", "weekday"]] = dfcountry_list[-1][["hour", "weekday"]]

        else:
            country_df["hour"] = df.index.hour
            country_df["weekday"] = df.index.dayofweek

            country_df["temperature"] = get_temperature(country, df.index)

            country_df["hdi"] = get_hdi(country)

            country_df["gdp"] = get_gdp(country, df.index)
            
            country_df["population"] = get_population(country)
        

        country_df

            
     
    
     
    
    