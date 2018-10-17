import os
from pathlib import Path
import pandas as pd
import numpy as np
import math

from parsing import split_tmp, split_wnd, split_ceil, split_vis, split_liquid_precip, split_snow


def see_maps_location(lat, lon):
    print(f'https://www.google.com.au/maps/search/{lat},{lon}')


def get_complete_station_years(path):
    """
    Figure out which stations have complete histories

    """
    station_years = pd.DataFrame()
    years = os.listdir(path/'raw')

    for y in years:
        this_station_year = pd.DataFrame.from_dict({
            'id':[s[:-4] for s in os.listdir(path/'raw'/f'{y}')],
            'year':y
        })
        station_years = pd.concat([station_years, this_station_year])

    files_per_station = station_years['id'].value_counts()
    stations_with_complete_history = files_per_station==len(station_years['year'].unique())
    is_complete_station_year = station_years['id'].isin(files_per_station[stations_with_complete_history].index)
    complete_station_years = station_years[is_complete_station_year].sort_values(['id','year'])
    complete_station_years.reset_index(inplace=True, drop=True)
    stations = complete_station_years['id'].unique()
    return stations, complete_station_years


def process_station_data(df):
    """
    Map the raw data from weather obs csv file to numeric columns in DataFrame
    """

    df.columns = map(str.lower, df.columns)

    timef = ['station','date','report_type']
    
    # parse out information from each of the relevant columns
    # data dictionary can be found at https://www.ncei.noaa.gov/data/global-hourly/doc/isd-format-document.pdf
    wnd, ceil, vis, tmp  = split_wnd(df), split_ceil(df), split_vis(df), split_tmp(df)
    wndf, ceilf, visf, tmpf = ['wnd_speed'], ['ceil','ceil_height'], ['vis_distance'], ['tmp']
    rain = split_liquid_precip(df)
    snow = split_snow(df)
    df['total_precip'] = rain['liquid_precip_depth_dimension'] + snow['snow_equivalent_water_depth_dimension']
    
    slim = pd.concat([
        df[timef],
        tmp['tmp'],
        rain['liquid_precip_depth_dimension'], snow['snow_equivalent_water_depth_dimension'], df['total_precip'],
        wnd[wndf], ceil[ceilf], vis[visf],
    ] , axis=1)

    # remove "Airways special report" records, 'SY-SA' records
    slim = slim[slim['report_type'] != 'SAOSP']

    # remove duplicated records by time
    slim = slim[~slim.date.duplicated()]

    slim.drop(['report_type'], axis=1, inplace=True)

    metadata = df[['station','latitude','longitude','elevation','name']].head(1)

    return metadata, slim


def get_all_station_data(path, station, years):
    """
    Sift through all the years with this station included, read the data, clean it
    """
    station_dfs = list()
    for year in years:
        this_year = pd.read_csv(
            path/'raw'/f'{year}'/f'{station}.csv',
            encoding='utf-8',
            parse_dates=['DATE'],
            low_memory=False,
            dtype={'STATION': 'object', 'LATITUDE': np.float32,'LONGITUDE': np.float32,
                   'ELEVATION': np.float32, 'NAME': str, 'REPORT_TYPE':str,
                   'TMP': str,
                  },
        )
        
        # don't use this station if any of the years have less than two observations per day
        if this_year.shape[0] < 365 * 2:
            metadata, _ = process_station_data(this_year)
        else:
            metadata, cleaned_data = process_station_data(this_year)
            cleaned_data['year'] = year
            station_dfs.append(cleaned_data)
    
    if len(station_dfs) > 0:
        station_data = pd.concat(station_dfs)

        # time series interpolation only works with datetime index
        station_data.set_index('date', inplace=True, drop=False)
        station_data = interpolate_measurements(station_data)
        station_data.station = station
        station_data.reset_index(inplace=True, drop=True)
    else:
        # filter out stations with less reliable
        station_data = None
    return metadata, station_data


def interpolate_measurements(station_data):
    """
    Create a baseline frequency of measurements, fill in the gaps
    """
    base = pd.DataFrame(
        index = pd.date_range(
            start=str(min(station_data.date).year), end=str(max(station_data.date).year+1),
            freq='H', closed='left'
        )
    )
    df = pd.merge(base, station_data, how='left', left_index=True, right_index=True)
    df['date'] = df.index.values
    
    df['tmp'] = df['tmp'].interpolate(method='time', limit_direction='both')
    # avoid warning about Nan mean operation
    if (df['vis_distance'].isnull().sum() == df['vis_distance'].shape[0]):
        df['vis_distance'].fillna(0)
    else:
        df['vis_distance'] = df['vis_distance'].fillna(df['vis_distance'].median())
    
    df['wnd_speed'] = df['wnd_speed'].interpolate(method='time', limit_direction='both')
    df['ceil'] = df['ceil'].fillna(0)
    df['ceil_height'] = df['ceil_height'].fillna(0)
    df['liquid_precip_depth_dimension'] = df['liquid_precip_depth_dimension'].fillna(0)
    df['snow_equivalent_water_depth_dimension'] = df['snow_equivalent_water_depth_dimension'].fillna(0)
    df['total_precip'] = df['total_precip'].fillna(0)
    
    return df


def collect_data_from_csvs(PATH, sample_size=None, shuffle=True):
    
    stations, station_years = get_complete_station_years(Path(PATH))
    if shuffle: np.random.shuffle(stations)
    
    if sample_size is not None:
        g = int(sample_size/10)
        if (sample_size < len(stations)):
            station_iterator = stations[0:int(sample_size)]
    else:
        g = 100
        station_iterator = stations
    
    c=0
    dfs = list()
    metas = list()
    print(f'Iterating through {len(station_iterator)} station file sets')
    for station in station_iterator:
        years = station_years['year'][station_years['id']==station]
        metadata, station_data = get_all_station_data(Path(PATH), station, years)
        
        if station_data is None:
            pass
        else:
            c+=1
            if c % g == 0:
                print(f'{c} - '+metadata.to_csv(None, header=False, index=False)[:-1])
            dfs.append(station_data)
            metas.append(metadata)
        
    metadata = pd.concat(metas)
    df = pd.concat(dfs)
    df = df.drop(['year'],axis=1)

    df.station = df.station.astype('category')
    metadata.station = metadata.station.astype('category')

    # get rid of stations with missing info (already having tried to interpolate)
    notnull_counts = df.groupby('station').apply(lambda c: c.notnull().sum())
    legit_stations = notnull_counts[(notnull_counts.apply(min, axis=1) == notnull_counts.apply(max).max())].index

    df = df[df.station.apply(lambda s: s in legit_stations)]
    metadata = metadata[metadata.station.apply(lambda s: s in legit_stations)]

    df.sort_values(['station','date'], inplace=True)
    df.reset_index(drop=True, inplace=True)

    metadata.sort_values(['station'], inplace=True)
    metadata.reset_index(drop=True, inplace=True)
    
    return df, metadata


def get_city_data(PATH, metadata, pop_threshold=1e6):

    print('Getting populous cities...')
    raw_cities = pd.read_csv(PATH,
                low_memory=False,
                encoding='utf-8',
                dtype={
                    'Country':'category',
                    'City': 'object',
                    'AccentCity': 'object',
                    'Region': 'category',
                    'Population': 'float32',
                    'Latitude': 'float32',
                    'Longitude': 'float32',
                })

    pop = raw_cities[raw_cities.Population > pop_threshold].copy()
    pop.sort_values('Population', ascending=False, inplace=True)
    cities = pop[~pop[['Latitude','Longitude']].duplicated()]
    cities.reset_index(drop=True, inplace=True)

    clos = cities.apply(find_closest_station, metadata=metadata, axis=1).apply(pd.Series)
    clos.columns=['station','closest_station_distance_km']
    mrgd = pd.merge(cities, clos, left_index=True, right_index=True, how='left')
    
    return mrgd.copy()


def distance(origin, destination):
    """
    Haversince distance from https://gist.github.com/rochacbruno/2883505
    Returns distance in kilometers
    """
    lat1, lon1 = origin
    lat2, lon2 = destination
    radius = 6371 # km radius of Earth

    dlat = math.radians(lat2-lat1)
    dlon = math.radians(lon2-lon1)
    a = math.sin(dlat/2) * math.sin(dlat/2) + math.cos(math.radians(lat1)) \
        * math.cos(math.radians(lat2)) * math.sin(dlon/2) * math.sin(dlon/2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    d = radius * c

    return d

def find_distance(m, coords):
    return distance((m['latitude'], m['longitude']), coords)

def find_closest_station(p, metadata):
    coords = (p.Latitude, p.Longitude)
    d = metadata.apply(find_distance, axis=1, coords=coords)
    return metadata.loc[d.idxmin()].station, min(d)


if __name__ == '__main__':

    PATH = f'/home/ubuntu/climate-classification/data'
    SAMPLE_SIZE = 4000

    df, metadata = collect_data_from_csvs(PATH, sample_size=SAMPLE_SIZE, shuffle=True)
    cities = get_city_data('./data/worldcitiespop.csv', metadata)

    closest_cities = cities.groupby('station').apply(lambda d: d.closest_station_distance_km.idxmin())
    ma = cities.loc[closest_cities]
    slim = df[df.station.apply(lambda s: s in ma.station.values)].copy()

    # need to reset categories so .groupby().apply() doesn't pick up the old ones
    for d in (slim, ma, cities):
        d['station'] = d['station'].astype(str).astype('category')
        d.reset_index(drop=True, inplace=True)

    print('Saving...')
    slim.to_feather(f'{PATH}/df')
    ma.to_feather(f'{PATH}/metadata')
    cities.to_feather(f'{PATH}/cities')

    print('Finished')
