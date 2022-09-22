import pandas as pd
import os
from pathlib import Path
import logging
import wget
import pycountry

logging.basicConfig(level=logging.INFO, format='%(asctime)s:%(levelname)s:%(message)s')

from config import DOWNLOAD_BASE

TEMP_PATH = os.path.join(os.getcwd(), 'temp')
os.makedirs(TEMP_PATH, exist_ok=True)

def download_and_read(alpha_2):
    '''
    downloads a series from aws, reads and returns it as pandas.DataFrame object
    Also deletes the temporary file

    Args:
        alpha_2(float): CC code for country of interest
    
    Returns:
        pd.DataFrame with single column of respective country demand
    ''' 
    
    filename = '{}_2020.csv'.format(alpha_2)

    wget.download(DOWNLOAD_BASE.format(alpha_2), os.path.join(TEMP_PATH, filename))
    df = pd.read_csv(os.path.join(TEMP_PATH, filename), index_col=0, parse_dates=True)

    os.remove(os.path.join(TEMP_PATH, filename))
    return df


def get(countries=None,
        country=None,
        *, 
        carrier=None, 
        start=None, 
        end=None,
        ):
    '''
    obtains time series of energy demand

    Args:
        countries(str, list, pd.RangeIndex): Pass one or more countries for which demand is obtained
        country(str, list, pd.RangeIndex): same as countries
        carrier(None or str): Currently only carrier 'electricity' is available
        start(str or pd.Timestamp): timestamp of start of demand data (interpreted as local time)
        end(str or pd.Timestamp): timestamp of end of demand data (interpreted as local time)

    Returns:
        pd.DataFrame of demand series

    '''
    temp_path = Path(os.getcwd()) / 'temp'
    temp_path.mkdir(exist_ok=True)

    assert countries is not None or country is not None, 'No country passed'
    countries = countries or country

    if isinstance(countries, str):
        countries = [countries]

    carrier = carrier or 'electricity'

    logging.info(f'Getting {carrier} demand for countries '+
                 str(countries).replace('[', '').replace(']', '') + '.')

    dummy = download_and_read('AE')
    start = start or dummy.index[0]    
    end = end or dummy.index[-1]    
    
    index = pd.date_range(start, end, freq='h')

    assert index[0] >= dummy.index[0], 'Chosen start timestamp is not in 2020'
    assert index[-1] <= dummy.index[-1], 'Chosen end timestamp is not in 2020'

    demand = pd.DataFrame(index=index)

    for country in countries:

        country = pycountry.countries.search_fuzzy(country)[0]
        logging.info(f'Obtaining demand for {country.name}.')

        series = download_and_read(country.alpha_2)
        series = series[series.columns[0]].loc[index]

        demand[country.name] = series

    return demand

if __name__ == '__main__':
    
    print(get(['Germany', 'albania'], start='2020-04-01', end='2020-05-01'))