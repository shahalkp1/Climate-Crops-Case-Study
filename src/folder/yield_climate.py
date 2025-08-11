import calendar
import ee
import pandas as pd
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score
import numpy as np
from logzero import logger

# Authenticate and initialize Google Earth Engine
ee.Authenticate()
ee.Initialize()

# States mapping
states = {
    'MP': 'Madhya Pradesh',
    'MH': 'Maharashtra'
}

# Load climate data
MHclimate = pd.read_csv('MHcombined.csv')
MPclimate = pd.read_csv('MPcombined.csv')

# MODIS NDVI and landcover datasets
dataset = ee.ImageCollection("MODIS/006/MOD13Q1").select('NDVI')
landcover = ee.ImageCollection("MODIS/006/MCD12Q1").first().select('LC_Type1')
cropland_mask = landcover.eq(12).Or(landcover.eq(14))

# Parameters
start_year = 2019
end_year = 2023
max_date = datetime(2023, 12, 31).date()
crop_labels = ['SB', 'PA', 'WH', 'GM', 'CO']
all_data = []

def get_state_geometry(state_name):
    india_states = ee.FeatureCollection("FAO/GAUL_SIMPLIFIED_500m/2015/level1")
    return india_states.filter(ee.Filter.eq('ADM1_NAME', state_name)).geometry()

def assign_season(date):
    month = date.month
    if 6 <= month <= 9:
        return 'Kharif'
    if month >= 10 or month <= 3:
        return 'Rabi'
    return 'Off-season'

crop_seasons = {
    'SB': ('06-01', '09-30'),  
    'PA': ('10-01', '03-31'),  
    'WH': ('10-01', '03-31'),  
    'GM': ('06-01', '09-30'), 
    'CO': ('06-01', '09-30'),
}

def extract_ndvi_timeseries(state_code, state_name, start_year, end_year, crop_label):
    region = get_state_geometry(state_name)
    mask_clipped = cropland_mask.clip(region)
    
    season_start_str, season_end_str = crop_seasons[crop_label]
    results = []

    for year in range(start_year, end_year + 1):
        if season_start_str > season_end_str:  
            season_start_date = datetime.strptime(f"{year}-{season_start_str}", "%Y-%m-%d").date()
            season_end_date = datetime.strptime(f"{year+1}-{season_end_str}", "%Y-%m-%d").date()
        else:
            season_start_date = datetime.strptime(f"{year}-{season_start_str}", "%Y-%m-%d").date()
            season_end_date = datetime.strptime(f"{year}-{season_end_str}", "%Y-%m-%d").date()

        if season_start_date > max_date:
            continue
        if season_end_date > max_date:
            season_end_date = max_date

        filtered_collection = dataset.filterDate(
            season_start_date.strftime("%Y-%m-%d"),
            season_end_date.strftime("%Y-%m-%d")
        ).filterBounds(region)

        def get_mean_ndvi(image):
            date = image.date()
            masked_image = image.updateMask(mask_clipped)
            mean_dict = masked_image.reduceRegion(
                reducer=ee.Reducer.mean(),
                geometry=region,
                scale=500,
                maxPixels=1e13
            )
            ndvi_val = mean_dict.get('NDVI')
            return ee.Feature(None, {
                'State': state_code,
                'Crop': crop_label,
                'Date': date.format('YYYY-MM-dd'),
                'NDVI': ee.Number(ndvi_val).multiply(0.0001)
            })

        ndvi_features = filtered_collection.map(get_mean_ndvi)
        features_info = ndvi_features.getInfo().get('features', [])

        ndvi_map = {f['properties']['Date']: f['properties']['NDVI'] for f in features_info}

        total_days = (season_end_date - season_start_date).days + 1
        for i in range(total_days):
            current_date = season_start_date + timedelta(days=i)
            date_str = current_date.strftime("%Y-%m-%d")
            ndvi_value = ndvi_map.get(date_str, None)

            results.append({
                'State': state_code,
                'Crop': crop_label,
                'Date': date_str,
                'Year': current_date.year,
                'Month': current_date.month,
                'Season': assign_season(current_date),
                'NDVI': ndvi_value
            })

    return results

for crop_label in crop_labels:
    for state_code, state_name in states.items():
        data = extract_ndvi_timeseries(state_code, state_name, start_year, end_year, crop_label)
        all_data.extend(data)

df = pd.DataFrame(all_data)
monthly_mean = df.groupby(['State', 'Crop', 'Year', 'Month', 'Season'])['NDVI'].mean().reset_index()
baseline = monthly_mean.groupby(['State', 'Crop', 'Month', 'Season'])['NDVI'].mean().reset_index()
baseline = baseline.rename(columns={'NDVI': 'Baseline_NDVI'})
merged = monthly_mean.merge(baseline, on=['State', 'Crop', 'Month', 'Season'])
merged['NDVI_Anomaly'] = merged['NDVI'] - merged['Baseline_NDVI']

def process_climate_df(df, state_code):
    df['date'] = pd.to_datetime(df['date'])
    df = df[df['date'].dt.year <= 2023]  
    df['Year'] = df['date'].dt.year
    df['Month'] = df['date'].dt.month
    
    climate_agg = df.groupby(['Year', 'Month']).agg({
        'mean': 'mean',        
        'rainfall_mm': 'sum'     
    }).reset_index()
    
    climate_agg['State'] = state_code
    climate_agg.rename(columns={'mean': 'Avg_Temperature', 'rainfall_mm': 'Total_Rainfall'}, inplace=True)
    return climate_agg

MH_climate_processed = process_climate_df(MHclimate, 'MH')
MP_climate_processed = process_climate_df(MPclimate, 'MP')

combined_climate = pd.concat([MH_climate_processed, MP_climate_processed], ignore_index=True)

def month_in_season(month, season):
    if season == 'Kharif':
        return 6 <= month <= 9
    elif season == 'Rabi':
        return month >= 10 or month <= 3
    else:
        return False

unique_crop_season = merged[['State', 'Crop', 'Season']].drop_duplicates()

climate_expanded_list = []
for state in unique_crop_season['State'].unique():
    climate_sub = combined_climate[combined_climate['State'] == state].copy()
    crop_season_sub = unique_crop_season[unique_crop_season['State'] == state].copy()
    rows = []
    for _, cs_row in crop_season_sub.iterrows():
        season = cs_row['Season']
        crop = cs_row['Crop']
        filtered_climate = climate_sub[climate_sub['Month'].apply(lambda m: month_in_season(m, season))].copy()
        filtered_climate['Crop'] = crop
        filtered_climate['Season'] = season
        rows.append(filtered_climate)
    expanded = pd.concat(rows)
    climate_expanded_list.append(expanded)

climate_expanded = pd.concat(climate_expanded_list, ignore_index=True)

final_merged = pd.merge(
    climate_expanded,
    merged,
    on=['State', 'Crop', 'Year', 'Month', 'Season'],
    how='left'
)

final_merged['NDVI_Anomaly'] = final_merged['NDVI'] - final_merged['Baseline_NDVI']

df_ml = final_merged.copy()
label_encoders = {}
for col in ['State', 'Crop', 'Season']:
    le = LabelEncoder()
    df_ml[col] = le.fit_transform(df_ml[col].astype(str))
    label_encoders[col] = le

feature_cols = ['Year', 'Month', 'Avg_Temperature', 'Total_Rainfall', 'State', 'Crop', 'Season']

train_df = df_ml[df_ml['NDVI'].notna()]
predict_df = df_ml[df_ml['NDVI'].isna()]

X_train = train_df[feature_cols]
y_train = train_df['NDVI']
X_predict = predict_df[feature_cols]

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
print(f"CV RMSE: {np.sqrt(-cv_scores).mean():.4f}")

predicted_ndvi = model.predict(X_predict)
df_ml.loc[df_ml['NDVI'].isna(), 'NDVI'] = predicted_ndvi

df_ml['NDVI_Anomaly'] = df_ml['NDVI'] - df_ml['Baseline_NDVI']

# Inverse transform categorical columns to original labels
for col in ['State', 'Crop', 'Season']:
    le = label_encoders[col]
    df_ml[col] = le.inverse_transform(df_ml[col].astype(int))

# Save final output
df_ml.to_csv('combined.csv', index=False)
