import re, os, sys, shutil
from pathlib import Path
import pandas as pd
import numpy as np

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
    
    wnd_metrics['wnd_speed'] = pd.to_numeric(wnd_metrics['wnd_speed']).astype('float32')
    wnd_metrics['wnd_direction'] = pd.to_numeric(wnd_metrics['wnd_direction']).astype('float32')
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
    ceil['ceil_height'] = (ceil['ceil_height']).astype('float32')
    
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
    vis['vis_distance'] = pd.to_numeric(vis['vis_distance']).astype('float32')
    vis['vis_variability'] = (vis['vis_variability'] == 'V').astype('float32')
    
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
    tmp['tmp'] = (nan * sign * value).astype('float32')
    
    return tmp


def split_liquid_precip(df):
    if 'aa1' in df.keys():
        unsplit = df['aa1'].fillna(',,,').str.split(',').copy()
        liquid_precip = pd.DataFrame.from_dict(
            dict(zip(df.index, unsplit)),
            orient='index',
            columns=[
                'liquid_precip_period_quantity_hours', # The quantity of time over which the LIQUID-PRECIPITATION was measured. (millimeters) 99 missing (milli)
                'liquid_precip_depth_dimension', # The depth of LIQUID-PRECIPITATION that is measured at the time of an observation.
                'liquid_precip_condition_code', # The code that denotes whether a LIQUID-PRECIPITATION depth dimension was a trace value. 9 missing.
                'liquid_precip_quality_code',
            ]
        )

        liquid_precip['liquid_precip_period_quantity_hours'] = liquid_precip['liquid_precip_period_quantity_hours'].replace('99', np.nan)
        liquid_precip['liquid_precip_depth_dimension'] = liquid_precip['liquid_precip_depth_dimension'].replace('9999',0.0)
        liquid_precip['liquid_precip_depth_dimension'] = pd.to_numeric(liquid_precip['liquid_precip_depth_dimension']).astype('float32')

        liquid_precip.liquid_precip_depth_dimension = liquid_precip.liquid_precip_depth_dimension.fillna(0)
    else:
        df['liquid_precip_period_quantity_hours'], df['liquid_precip_depth_dimension'] = 0, 0
        liquid_precip = df[['liquid_precip_period_quantity_hours','liquid_precip_depth_dimension']].copy()

    return liquid_precip


def split_snow(df):
    if 'aj1' in df.keys():
        unsplit = df['aj1'].fillna(',,,,,').str.split(',').copy()
        snow = pd.DataFrame.from_dict(
            dict(zip(df.index, unsplit)),
            orient='index',
            columns=[
                'snow_identifier', # The quantity of time over which the SNOW-DEPTH was measured. (millimeters) 99 missing
                'snow_depth_dimension', # The depth of SNOW-DEPTH that is measured at the time of an observation.
                'snow_condition_code', # The code that denotes whether a SNOW-DEPTH depth dimension was a trace value. 9 missing.
                'snow_quality_code',
                'snow_equivalent_water_depth_dimension',
                'snow_equivalent_water_condition_code'
            ]
        )

        snow['snow_equivalent_water_condition_code'] = snow['snow_equivalent_water_condition_code'].replace('9', np.nan)
        snow['snow_depth_dimension'] = snow['snow_depth_dimension'].replace('9999',0.0)
        snow['snow_depth_dimension'] = pd.to_numeric(snow['snow_depth_dimension']).astype('float32')

        
        snow['snow_equivalent_water_depth_dimension'] = snow['snow_equivalent_water_depth_dimension'].replace('',0)
        snow['snow_equivalent_water_depth_dimension'] = snow['snow_equivalent_water_depth_dimension'].replace('9999',0.0)
        snow['snow_equivalent_water_depth_dimension'] = pd.to_numeric(snow['snow_equivalent_water_depth_dimension']).astype('float32')
        
    else:
        df['snow_depth_dimension'] , df['snow_equivalent_water_depth_dimension'] = 0, 0
        snow = df[['snow_depth_dimension','snow_equivalent_water_depth_dimension']].copy()

    return snow