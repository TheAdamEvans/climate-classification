import requests
import time
import os
from datetime import datetime, timedelta

import pandas as pd
import numpy as np
from ftplib import FTP
from netCDF4 import Dataset
from scipy.signal import convolve2d


def see_point_on_map(coords):
    if coords[1] > 180: coords = easting_to_westing(coords)
    print(f'https://www.google.com/maps/search/?api=1&query={coords[0]},{coords[1]}')


def download(ftp, filename):
    with open(filename, 'wb') as file:
        ftp.retrbinary(f'RETR {filename}', file.write, blocksize=4096)

def specific_humidity_to_relative_humidity(qair, temp, press):
    """
    straight from https://earthscience.stackexchange.com/a/2385
    https://github.com/PecanProject/pecan/blob/master/modules/data.atmosphere/R/metutils.R#L15-L35
    """
    es = 6.112 * np.exp((17.67 * temp) / (temp + 243.5))
    e = qair * press / (0.378 * qair + 0.622)
    rh = e / es
    rh[rh > 1] = 1
    rh[rh < 0] = 0
    return rh

# def easting_to_westing(coords):
#     """Take strictly positive longitude <360 and make abs(lon) <= 180"""
#     if coords[1] > 180.:
#         return (coords[0], coords[1] - 360.)
#     else: 
#         return coords

# def westing_to_easting(coords):
#     """Take abs(lon) <= 180 and make strictly positive longitude <360"""
#     if coords[1] < 0:
#         return (coords[0], coords[1] + 360.)
#     else: 
#         return coords

# def check_for_missing(a):
#     return bool(np.sum([d.mask for d in a]))


def df_from_nc(filename):
    var = filename.split('.')[0]
    dataset = Dataset(filename)
    standard_name = dataset.variables[var].standard_name
    
    lat = dataset.variables['lat'][:]
    lon = dataset.variables['lon'][:]
    tim = dataset.variables['time'][:]
    i = pd.MultiIndex.from_product([tim, lat, lon], names=['time','lat','lon'])

    if dataset.variables[var].ndim==4:
        # this is slicing the lowest level (only level often) of the atmosphere
        d = dataset.variables[var][:,0,:,:].ravel()
    elif dataset.variables[var].ndim==3:
        d = dataset.variables[var][:,:,:].ravel()
    else:
        raise ValueError(f'Too many dimensions in {filename}!')
        
    df = pd.DataFrame(d, index=i, columns=[standard_name])
    df.reset_index(inplace=True)
    return df

def download_relevant_files(year, vars_of_interest):
    
    with FTP('ftp.cdc.noaa.gov') as ftp:
        ftp.login()
        ftp.cwd('Datasets/ncep.reanalysis2/gaussian_grid')

        vars_available_for_download = ftp.nlst(f'*{year}.nc')
        filenames = [v for v in vars_available_for_download if v.startswith(vars_of_interest)]
        
        for filename in filenames:
            if os.path.isfile(f'{filename}'):
                print(f'{filename} exists already')
            else:
                start = time.time()
                download(ftp, filename)
                stop = time.time()
                print(f'Downloaded {filename} in {np.round(stop-start)} seconds')
    return filenames
    

def combine_vars(year, filenames, include):
    
    if os.path.isfile(f'{year}.feather'):
        print(f'Feather file for {year} exists already')
    else:
        print(f'Reading data for {year}...')
        dfs = list()
        start = time.time()
        for filename in filenames:
            partial = df_from_nc(filename)
            reduced = pd.merge(include, partial, on=['lat','lon'])
            dfs.append(reduced)
        
        # merge all the dfs together
        df = dfs[0]
        if len(dfs) > 1:
            for i in range(1,len(dfs)):
                # strictly speaking this could be a sort then concat
                df = pd.merge(df, dfs[i], on=['lat','lon','time'], how='left')

        df['time'] = df['time'].astype('int32')
        df['lat'] = df['lat'].astype('category')
        df['lon'] = df['lon'].astype('category')
        
        df.reset_index(drop=True, inplace=True)
        df.to_feather(f'{year}.feather')
        
        for filename in filenames:
            os.remove(filename)
        stop = time.time()
        print(f'Generated .feather file in {np.round(stop-start)} seconds')

    
def determine_included_grids():
    """
    To conserve storage space, this function identifies ~1/3 of grid coords to keep
    """
    
    if os.path.isfile('land.sfc.gauss.nc'):
        print(f'Land mask exists already')
    else:
        print('Downloading land mask...')
        with FTP('ftp.cdc.noaa.gov') as ftp:
            ftp.login()
            ftp.cwd('Datasets/ncep.reanalysis2/gaussian_grid')
            download(ftp, 'land.sfc.gauss.nc')
    land = Dataset('land.sfc.gauss.nc')
    
    # include the surrounding ocean for any land mass
    is_land = convolve2d(land['land'][0,:,:], np.ones((3,3)), 'same') > 0
    
    # exclude extreme latitudes
    reasonable_latitudes = (land['lat'][:] > -60) & (land['lat'][:] < 75)
    repeated_reasonable_latitudes = np.repeat(reasonable_latitudes, is_land.shape[1]).reshape(-1, is_land.shape[1])
    
    include = (is_land & repeated_reasonable_latitudes)
    lats, lons = np.where(include)
    
    include_grids = pd.DataFrame({
        'lat': land['lat'][lats],
        'lon': land['lon'][lons],
    })
    return include_grids

if __name__=="__main__":
    VARS_OF_INTEREST = (
        'air', # air temperature
        'shum', # specific humidity
        # 'vwnd', 'uwnd', # wind speed and direction
        # 'tcdc', # total cloud cover
        'prate', # precipatation rate
        #'weasd', # water equivalent snow depth
        'pres.sfc', # pressure at the surface
    )
    
    BEGIN_YEAR=1979
    END_YEAR=1989
    yrange=range(BEGIN_YEAR,END_YEAR+1)
    
    include = determine_included_grids()
    
    for year in yrange:
        filenames = download_relevant_files(year, VARS_OF_INTEREST)
        combine_vars(year, filenames, include)

    print('Aggregating...')
    df = pd.concat([pd.read_feather(f'{y}.feather') for y in yrange])
    
    # convert from Kelvin to Celcius
    df['air_temperature'] = df['air_temperature'].subtract(np.float32(273.15))
    
    # convert from a rate per kg/m^2 to mm
    df.loc[df['precipitation_rate'] < 0,'precipitation_rate'] = 0
    df['precipitation_rate'] = df['precipitation_rate'].multiply((1/4)*24*60*60)
    
    # convert from integer to datetime
    df['time'] = pd.to_datetime(
        df['time'], unit='h', utc=True,
        origin=datetime.strptime('1800-1-1 00:00:00','%Y-%m-%d %H:%M:%S')
    )
    # Convert from Pascals to hectopascals aka millibars
    df['air_pressure'] = df['air_pressure'].divide(100)
    
    # use relative instead of specific humidity
    df['relative_humidity'] = specific_humidity_to_relative_humidity(
        df['specific_humidity'], df['air_temperature'], df['air_pressure'])
    df.drop(['air_pressure','specific_humidity'],axis=1,inplace=True)

    df.reset_index(drop=True, inplace=True)
    df.to_feather('df.feather')
    [os.remove(f'{y}.feather') for y in yrange]
    
    print('Complete!')
