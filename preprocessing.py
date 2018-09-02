import pandas as pd
import numpy as np
import matplotlib as plt

from datetime import datetime
from pathlib import Path
import re
import os
import sys

# steps to download data
# wget --header="Host: www.ncei.noaa.gov" --header="Connection: keep-alive" "https://www.ncei.noaa.gov/data/global-hourly/archive/1901.tar.gz" -O "1901.tar.gz" -c
# mkdir 1901 && tar -xzf 1901.tar.gz -C 1901
# rm 1901.tar.gz

def see_maps_location(lat, lon):
    print(f'https://www.google.com.au/maps/search/{lat},{lon}')


def split_wnd(df):
    
    unsplit = df['wnd'].str.split(',')
    wnd_metrics = pd.DataFrame.from_dict(
        dict(zip(df.index, unsplit)),
        orient='index',
        columns=[
            'wnd_direction', # The angle, measured in a clockwise direction, between true north and the direction from which the wind is blowing.
            'wnd_direction_code', # If type code (below) = V, then 999 indicates variable wind direction.
            'wnd_type_code', # If a value of 9 appears with a wind speed of 0000, this indicates calm winds.
            'wnd_speed', # meters per second. 9999 = Missing.
            'wnd_speed_code',
        ]
    )
    
    wnd_metrics['wnd_speed'] = wnd_metrics['wnd_speed'].replace('9999', np.nan)
    wnd_metrics['wnd_direction'] = wnd_metrics['wnd_direction'].replace('999', np.nan)
    
    wnd_metrics['wnd_speed'] = pd.to_numeric(wnd_metrics['wnd_speed'])
    wnd_metrics['wnd_direction'] = pd.to_numeric(wnd_metrics['wnd_direction'])
    wnd_metrics['wnd_direction_sin'] = np.sin(np.deg2rad(wnd_metrics['wnd_direction']))
    wnd_metrics['wnd_direction_cos'] = np.cos(np.deg2rad(wnd_metrics['wnd_direction']))
    
    return wnd_metrics


def split_ceil(df):
    
    unsplit = df['cig'].str.split(',')
    ceil = pd.DataFrame.from_dict(
        dict(zip(df.index, unsplit)),
        orient='index',
        columns=[
            'ceil_height', # Lowest clouds in meters. Unlimited = 22000.
            'ceil_code', # A quality status of a reported ceiling height dimension.
            'ceil_determination_code', # Method used to determine the ceiling.
            'ceil_cavok', # Whether the 'Ceiling and Visibility Okay' (CAVOK) condition has been reported.
        ]
    )
    
    
    ceil['ceil'] = ceil['ceil_height'] != '22000'
    ceil.loc[ceil['ceil_height'] == '99999','ceil'] = np.nan
    ceil['ceil_height'] = ceil['ceil_height'].replace(['99999','22000'], np.nan)
    ceil['ceil_height'] = (ceil['ceil_height']).astype(float)
    
    ceil['ceil_code'] = ceil['ceil_code'].replace('9', np.nan)
    ceil['ceil_determination_code'] = ceil['ceil_determination_code'].replace('9', np.nan)
    
    return ceil


def split_vis(df):
    unsplit = df['vis'].str.split(',')
    vis = pd.DataFrame.from_dict(
        dict(zip(df.index, unsplit)),
        orient='index',
        columns=[
            'vis_distance',  # Horizontal distance (in meters) at which an object can be seen and identified. # Missing = 999999. NOTE: Values greater than 160000 are entered as 160000.
            'vis_code', # Quality status.
            'vis_variability', # Denotes whether or not the reported visibility is variable. 9 = Missing.
            'vis_variability_code',
        ]
    )
    
    vis['vis_distance'] = vis['vis_distance'].replace('999999', np.nan)
    vis['vis_distance'] = pd.to_numeric(vis['vis_distance'])
    vis['vis_variability'] = (vis['vis_variability'] == 'V').astype(float)
    
    return vis


def split_tmp(df):
    unsplit = df['tmp'].str.split(',')
    tmp = pd.DataFrame.from_dict(
        dict(zip(df.index, unsplit)),
        orient='index',
        columns=[
            'tmp', # temps are in celsius, scaled up by 10
            'tmp_code',
        ]
    )

    nan = tmp['tmp'].apply(lambda t: np.nan if t=='+9999' else 1.0)
    sign = tmp['tmp'].apply(lambda t: 1.0 if t[0]=='+' else -1.0)
    value = tmp['tmp'].apply(lambda t: t[1:]).astype(float) / 10
    tmp['tmp'] = (nan * sign * value)
    
    return tmp


def add_datepart(df, fldname, drop=True, time=False):
    # straight from https://github.com/fastai/fastai/blob/master/fastai/structured.py#L76-L120
    fld = df[fldname]
    fld_dtype = fld.dtype
    if isinstance(fld_dtype, pd.core.dtypes.dtypes.DatetimeTZDtype):
        fld_dtype = np.datetime64

    if not np.issubdtype(fld_dtype, np.datetime64):
        df[fldname] = fld = pd.to_datetime(fld, infer_datetime_format=True)
    targ_pre = re.sub('[Dd]ate$', '', fldname)
    attr = ['Year', 'Month', 'Week', 'Day', 'Dayofweek', 'Dayofyear',
            'Is_month_end', 'Is_month_start', 'Is_quarter_end', 'Is_quarter_start', 'Is_year_end', 'Is_year_start']
    if time: attr = attr + ['Hour', 'Minute', 'Second']
    for n in attr: df[targ_pre + n] = getattr(fld.dt, n.lower())
    df[targ_pre + 'Elapsed'] = fld.astype(np.int64) // 10 ** 9
    if drop: df.drop(fldname, axis=1, inplace=True)


def save_station_data(d, slim, path):

    # denormalized features of this station.. anything that could be interesting
    d['num_obs'] = slim.shape[0]
    d['num_on_the_hour_obs'] = (slim['Minute']==0).sum()
    if d['num_obs'][0] > 8000:
        print(d.to_csv(None, header=False, index=False)[:-1])
    # TODO closest city

    slim.to_csv(
        (path/f"0-{d['station'][0]}.csv"),
        header = True, index = False
        )

    if (path/'stations.csv').is_file():
        with open((path/'stations.csv'), 'a') as file:
            d.to_csv(file, header=False, index=False)
    else:
        with open((path/'stations.csv'), 'a') as file:
            d.to_csv(file, header=True, index=False)


if __name__ == '__main__':

    if len(sys.argv) < 2:
        raise ValueError('No year!')
    else:
        year = int(sys.argv[1])

    path = Path(f'./data/{year}')
    station_files = [p for p in os.listdir(path) if not (p.startswith('0-') or p=='stations.csv')]


    for file in station_files:

        df = pd.read_csv(
            path/f'{file}',
            parse_dates=['DATE'],
            low_memory=False,
        )
        df.columns = map(str.lower, df.columns)

        if df.shape[0] < (3 * 365):
            # skip stations with too little data
            pass
        else:
            # merge them all together
            wnd, ceil, vis, tmp  = split_wnd(df), split_ceil(df), split_vis(df), split_tmp(df)
            wndf, ceilf, visf, tmpf = ['wnd_speed','wnd_direction_sin','wnd_direction_cos'], ['ceil','ceil_height'], ['vis_distance'], ['tmp']
            add_datepart(df, 'date', drop=False, time=True)
            timef = ['station','date','Year','Dayofyear','Hour','Minute','call_sign','report_type']

            # filter columns
            slim = pd.concat([df[timef],wnd[wndf],ceil[ceilf],vis[visf],tmp[tmpf]], axis=1)

            # some stations have multiple reporting call signs, take the most frequent one
            slim = slim[slim['call_sign'] == slim['call_sign'].value_counts().idxmax()]
            # remove "Airways special report" records
            slim = slim[slim['report_type'] != 'SAOSP']

            metadata = df[['station','latitude','longitude','elevation','name']].head(1)
            save_station_data(metadata, slim, path)

