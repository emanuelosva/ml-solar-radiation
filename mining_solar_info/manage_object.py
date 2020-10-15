import logging
from math import trunc
import numpy as np
import pandas as pd
from urllib3.exceptions import MaxRetryError
from requests.exceptions import HTTPError
from solarpage_objects import SolarPage
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _new_scraper(latitude, longitude):
    solarpage = _fetching_solapage(latitude, longitude)
    if solarpage:
        df = _create_dataframe(solarpage)
        df = _wrangling_dataframe(df, solarpage)
        _save_to_csv(df)


def _fetching_solapage(latitude, longitude):
    logger.info(
        f'Start fetching solar_info for Lat:{latitude} & Lon:{longitude}'
    )
    solarpage = None
    try:
        solarpage = SolarPage(latitude, longitude)
    except (HTTPError, MaxRetryError):
        logger.warning('Error while fetching the solar_info', exc_info=False)

    return solarpage


def _create_dataframe(solarpage):
    _crude_data = solarpage.content.split('\r\n')
    latitude = float(solarpage.lat)
    longitude = float(solarpage.lon)
    altitude = float(_crude_data[2].split(': ')[1])

    _headers = _crude_data[16].split(',')
    _data = _crude_data[17:8776]
    _data = [row.split(',') for row in _data]

    df = pd.DataFrame(_data, columns=_headers)
    df['Lat'] = latitude
    df['Lon'] = longitude
    df['Alt'] = altitude

    return df


def _wrangling_dataframe(df, solarpage):
    logger.info('Wrangling the solar data')
    correction = trunc(solarpage.lon / 15)
    correction = np.timedelta64(correction, 'h')

    df['time(UTC)'] = (df
                       .apply(lambda row: row['time(UTC)'], axis=1)
                       .apply(lambda row: dts(row))
                       .apply(lambda date: str(f'{date[0]}-{date[1]}-{date[2]}T{date[3]}'))
                       .apply(lambda date: np.array(date, dtype='datetime64'))
                       .apply(lambda date: date + correction)
                       )

    df.rename(columns={'time(UTC)': 'Date'}, inplace=True)

    df['Month'] = df['Date'].dt.month
    df['Day'] = df['Date'].dt.day
    df['NumDay'] = df['Date'].dt.dayofyear
    df['Hour'] = df['Date'].dt.hour
    _headers = solarpage.headers()

    df = df[_headers]
    df[_headers[1:]] = np.array(df[_headers[1:]], dtype='float32')

    return df


def dts(ls):
    """Create de datetime format"""
    date_ = [ls[:4], ls[4:6], ls[6:8], ls[9:11]]
    return date_


def _save_to_csv(df):
    logger.info('Saving solar data...')
    df.to_csv('solar_data.csv', mode='a',
              encoding='utf-8', header=False, index=False)
