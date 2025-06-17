# Converting the idle start time for mdt time into a timestamp and extracting the hour and day of the week from it
import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

def extract_datetime_features(df, datetime_col, timezone=None, drop_columns=False):
    """
    Converts a datetime column into a pandas datetime object and extracts hour and day of week.
    
    Parameters:
        df (pd.DataFrame): Input DataFrame.
        datetime_col (str): Name of the column with datetime values.
        timezone (str, optional): Timezone to localize or convert to (e.g., 'US/Mountain').
        drop_columns (bool, optional): If True, drops the 'idle_start_hour_mdt' column.

    Returns:
        pd.DataFrame: DataFrame with added 'idle_start_hour' column.
    """
    df = df.copy()
    
    # Convert column to datetime
    df['timestamp'] = pd.to_datetime(df[datetime_col], format='mixed', errors='coerce')
    
    # Convert to timezone if specified
    if timezone:
        df['timestamp'] = df['timestamp'].dt.tz_localize('UTC').dt.tz_convert(timezone)

    # Extract features
    df['idle_start_hour'] = df['timestamp'].dt.hour

    # Drop columns if specified
    if drop_columns:
        df.drop(columns=['idle_start_hour_mdt'], inplace=True)

    return df



def add_week_tracking(df, timestamp_col):
    """
    Adds week tracking features to a DataFrame based on a timestamp column.
    
    Parameters:
        df (pd.DataFrame): Input DataFrame.
        timestamp_col (str): Name of the timestamp column (must be datetime type or convertible).
    
    Returns:
        pd.DataFrame: DataFrame with 'week_start_date', 'week_end_date', and 'week_number' columns added.
    """
    df = df.copy()
    
    # Ensure datetime and remove timezone
    df[timestamp_col] = pd.to_datetime(df[timestamp_col], errors='coerce')
    df[timestamp_col] = df[timestamp_col].dt.tz_localize(None)

    # Get Monday-aligned week start
    df['week_start_date'] = df[timestamp_col].dt.to_period('W-MON').dt.start_time

    # Get first week start
    first_week = df['week_start_date'].min()

    # Calculate custom week number
    df['week_number'] = ((df['week_start_date'] - first_week).dt.days // 7) + 1

    # Calculate week end date (Sunday)
    df['week_end_date'] = df['week_start_date'] + pd.Timedelta(days=6)

    return df


def convert_idle_location_to_coords(df, x_col, y_col, lat_col='lat', lon_col='lon', drop_original=False):
    """
    Converts raw idle location data into latitude and longitude by dividing by 1e7.
    
    Parameters:
        df (pd.DataFrame): Input DataFrame.
        x_col (str): Column name for longitude values (typically 'idle_location_x').
        y_col (str): Column name for latitude values (typically 'idle_location_y').
        lat_col (str): Name for the output latitude column.
        lon_col (str): Name for the output longitude column.
    
    Returns:
        pd.DataFrame: DataFrame with added 'lat' and 'lon' columns.
    """
    df = df.copy()
    df[lat_col] = df[y_col] / 1e7
    df[lon_col] = df[x_col] / 1e7

    if drop_original:
        df.drop(columns=[x_col, y_col], inplace=True)

    return df



def cluster_downtime(data):
    coords = data[['lat', 'lon']].copy()  # Only use numeric coordinate columns

    # Scale the coordinates
    scaler = StandardScaler()
    coords_scaled = scaler.fit_transform(coords)

    # Apply DBSCAN clustering
    dbscan = DBSCAN(eps=0.028, min_samples=5)
    data = data.copy()
    data['cluster'] = dbscan.fit_predict(coords_scaled)

    return data  # Remove noise points