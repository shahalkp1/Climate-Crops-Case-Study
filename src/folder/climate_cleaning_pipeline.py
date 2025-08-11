import pandas as pd
import numpy as np
from pathlib import Path
from logzero import logger
import os

TEMP_PATH_RAW = os.getenv('TEMP_PATH')
RAIN_PATH_RAW = os.getenv('RAIN_PATH')
STATE = os.getenv('STATE')

if not TEMP_PATH_RAW or not RAIN_PATH_RAW:
    raise EnvironmentError("Environment variables TEMP_PATH and RAIN_PATH must be set.")

TEMP_PATH = Path(TEMP_PATH_RAW)
RAIN_PATH = Path(RAIN_PATH_RAW)
DISTRICTS = None
def remove_outliers_iqr(df, columns):
    logger.info('removing outliers')
    filtered_df = df.copy()
    for col in columns:
        Q1 = filtered_df[col].quantile(0.20)
        Q3 = filtered_df[col].quantile(0.80)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        filtered_df = filtered_df[(filtered_df[col] >= lower_bound) & (filtered_df[col] <= upper_bound)]
    return filtered_df

def seasonal_smooth_impute(df, column, window=7, method='spline', order=3, spike_threshold=2.0):
    logger.info('Applying smoothing')
    df = df.copy()
    df = df.loc[:, ~df.columns.duplicated()]
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df = df.sort_values('date').reset_index(drop=True)
    
    original_series = df[column].astype(float)
    nan_mask = original_series.isna()
    smoothed = original_series.rolling(window=window, center=True, min_periods=1).mean()

    try:
        interpolated = smoothed.interpolate(method=method, order=order, limit_direction='both')
    except:
        interpolated = smoothed.interpolate(method='linear', limit_direction='both')

    filled_series = original_series.copy()
    filled_series[nan_mask] = interpolated[nan_mask]

    diffs = np.abs(np.diff(filled_series))
    spikes = np.where(diffs > spike_threshold)[0]
    for idx in spikes:
        if nan_mask.iloc[idx] or nan_mask.iloc[idx + 1]:
            avg = (filled_series.iloc[idx] + filled_series.iloc[idx + 1]) / 2
            filled_series.iloc[idx] = avg
            filled_series.iloc[idx + 1] = avg

    return filled_series

def process_district(temp_df, rain_df, district):
    logger.info('processing for: '+str(district))
    
    temp_d = temp_df[temp_df['District'] == district].copy()
    temp_d = temp_d.sort_values(by='date').reset_index(drop=True)


    daily_median = temp_d.groupby('date')[['mean', 'min', 'max']].median().reset_index()
    cleaned_df = remove_outliers_iqr(daily_median, ['mean', 'min', 'max'])

    cleaned_df = cleaned_df.set_index('date').sort_index()
    full_index = pd.date_range(start=cleaned_df.index.min(), end=cleaned_df.index.max(), freq='D')
    cleaned_df = cleaned_df.reindex(full_index)
    cleaned_df = cleaned_df.reset_index().rename(columns={'index': 'date'})

    for col in ['mean', 'min', 'max']:
        cleaned_df[col] = seasonal_smooth_impute(cleaned_df, col)

    rain_d = rain_df[rain_df['District'] == district].copy()
    rain_d = rain_d.sort_values(by='date').reset_index(drop=True)
    rain_d['daily_district_median'] = rain_d.groupby(['date', 'District'])['rainfall_mm'].transform('median')
    rain_d = rain_d[['date', 'daily_district_median']].drop_duplicates()

    final_df = pd.merge(cleaned_df, rain_d, on='date', how='left')
    final_df.rename(columns={'daily_district_median': 'rainfall_mm'}, inplace=True)

    return final_df

def impute_rainfall(df, window=7):
    df = df.sort_values('date')
    
    nan_mask = df['rainfall_mm'].isna()
    df['rainfall_mm'] = df['rainfall_mm'].fillna(method='ffill').fillna(method='bfill')
    
    smoothed = df['rainfall_mm'].rolling(window=window, center=True, min_periods=1).median()
    
    df['rainfall_mm'] = np.where(
        (df['rainfall_mm'] > 0),
        smoothed,
        df['rainfall_mm']
    )
    
    return df


if __name__ == '__main__':
    temp_df = pd.read_csv(TEMP_PATH)
    temp_df.drop(columns=['Unnamed: 0'], inplace=True, errors='ignore')
    temp_df['date'] = pd.to_datetime(temp_df['date'], errors='coerce')
    temp_df['District'] = temp_df['District'].str.strip().str.upper()

    rain_df = pd.read_csv(RAIN_PATH)
    rain_df.drop(columns=['Unnamed: 0'], inplace=True, errors='ignore')
    rain_df['date'] = pd.to_datetime(rain_df['date'], errors='coerce')
    rain_df['District'] = rain_df['District'].str.strip().str.upper()

    rain_df = impute_rainfall(rain_df)

    all_districts = temp_df['District'].dropna().unique()
    to_process = DISTRICTS if DISTRICTS else all_districts

    combined_df = []

    for district in to_process:
        result_df = process_district(temp_df, rain_df, district)
        result_df.dropna(inplace=True)
        result_df['District'] = district 
        combined_df.append(result_df)

    final_combined_df = pd.concat(combined_df).sort_values('date').reset_index(drop=True)
    final_combined_df.to_csv(f"{STATE}combined.csv", index=False)