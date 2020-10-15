"""Script for minig info from pvgis for testing sets"""

# Request
from pvlib.irradiance import get_extra_radiation
from pvlib.solarposition import get_solarposition
from pvlib.atmosphere import gueymard94_pw
from pvlib.iotools import get_pvgis_tmy
import numpy as np
import pandas as pd


def get_from_pvgis(lat, lon, tz):
    """Return the pvgis data"""

    # PVGIS information
    info = get_pvgis_tmy(lat, lon)

    # Altitude
    alt = info[2]['location']['elevation']

    # Get th principal information
    df = info[0]

    # Convert UTC to time zone
    df.index = df.index.tz_convert(tz)
    df.index.name = f'Time {tz}'

    # Get the solar position
    solpos = get_solarposition(
        time=df.index,
        latitude=lat,
        longitude=lon,
        altitude=alt,
        pressure=df.SP,
        temperature=df.T2m
    )

    # Create the schema
    data = df
    data = data.drop(df.columns, axis=1)
    data['HourOfDay'] = df.index.hour
    data['Zenith'] = solpos['zenith']
    data['Temperature'] = df.T2m
    data['Humidity'] = df.RH
    data['WindSpeed'] = df.WS10m
    data['WindDirection'] = df.WD10m
    data['PrecipitableWater'] = gueymard94_pw(df.T2m, df.RH)
    data['Pressure'] = df.SP
    data['ExtraRadiation'] = get_extra_radiation(df.index)
    data['GHI'] = df['G(h)']

    return data


places = [
    {
        'name': 'torreon',
        'lat': 25.548597,
        'lon': -103.4719564,
        'tz': 'America/Mexico_City'
    },
    {
        'name': 'santaana',
        'lat': 30.5345495,
        'lon': -111.13968,
        'tz': 'America/Mexico_City'
    },
    {
        'name': 'madrid',
        'lat': 40.4378698,
        'lon': -3.8196212,
        'tz': 'Europe/Madrid'
    },
]


if __name__ == "__main__":

    for p in places:
        df = get_from_pvgis(p['lat'], p['lon'], p['tz'])
        filename = f"""./data/test_{p['name']}.csv"""
        df.to_csv(filename, index=False, encoding='utf-8', header=True)
