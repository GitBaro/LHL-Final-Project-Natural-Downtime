# Converting the idle start time for mdt time into a timestamp and extracting the hour and day of the week from it
import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import folium
from folium.plugins import FloatImage


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



def select_idle_features(df, feature_cols=None, latest_week_only=False, week_col='week_number'):
    """
    Selects a subset of columns from the DataFrame for idle analysis,
    with an optional filter for the latest week.

    Parameters:
        df (pd.DataFrame): Input DataFrame.
        feature_cols (list, optional): List of column names to select. 
                                       Defaults to ['lat', 'lon', 'idle_start_hour', 
                                       'idle_duration_hr', 'week_start_date', 'week_number'].
        latest_week_only (bool): If True, filter the DataFrame to include only the latest week.
        week_col (str): Column name representing week number.

    Returns:
        pd.DataFrame: DataFrame containing only the selected features (optionally from the latest week).
    """
    if feature_cols is None:
        feature_cols = ['lat', 'lon', 'idle_start_hour', 'idle_duration_hr', 'week_number']

    df = df.copy()
    
    if latest_week_only:
        if week_col not in df.columns:
            raise ValueError(f"Column '{week_col}' not found in DataFrame.")
        latest_week = df[week_col].max()
        df = df[df[week_col] == latest_week]

    return df[feature_cols].copy()




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



def summarize_cluster_downtime(df, cluster_col='cluster', idle_duration_col='idle_duration_hr'):
    """
    Summarizes natural downtime statistics for each cluster.
    
    Parameters:
        df (pd.DataFrame): Input DataFrame containing cluster and idle duration information.
        cluster_col (str): Column name representing cluster labels.
        idle_duration_col (str): Column name representing idle durations (in hours).
    
    Returns:
        pd.DataFrame: Summary DataFrame with total, average, and frequency of downtime per cluster.
    """
    df = df.copy()

    # Filter out noise clusters (typically labeled -1)
    valid_clusters = df[df[cluster_col] != -1]

    # Group and aggregate
    cluster_summary = valid_clusters.groupby(cluster_col).agg({
        idle_duration_col: ['sum', 'mean', 'count']
    }).reset_index()

    # Rename columns
    cluster_summary.columns = [cluster_col, 'total_downtime', 'avg_downtime', 'frequency']

    return cluster_summary



def get_best_cluster(cluster_summary_df, sort_by='total_downtime', ascending=False):
    """
    Identifies the best cluster based on a specified metric (e.g., total downtime).
    If all points are noise (e.g., cluster == -1), it still returns that as the 'best cluster'.

    Parameters:
        cluster_summary_df (pd.DataFrame): DataFrame summarizing cluster stats 
                                           (e.g., from summarize_cluster_downtime).
        sort_by (str): Column to sort by when ranking clusters.
        ascending (bool): Sort order; False to get the highest value as best.

    Returns:
        pd.Series: The row corresponding to the best cluster.
    """
    if cluster_summary_df.empty:
        raise ValueError("Cluster summary DataFrame is empty.")

    return cluster_summary_df.sort_values(by=sort_by, ascending=ascending).iloc[0]



def get_best_cluster_coords(df, best_cluster, cluster_col='cluster', lat_col='lat', lon_col='lon'):
    """
    Returns the average coordinates (latitude and longitude) for the given best cluster.
    
    Parameters:
        df (pd.DataFrame): DataFrame containing cluster and coordinate information.
        best_cluster (pd.Series or dict): Row with cluster information (must include key 'cluster').
        cluster_col (str): Column name identifying the cluster.
        lat_col (str): Column name for latitude.
        lon_col (str): Column name for longitude.
    
    Returns:
        pd.Series: A series with the average latitude and longitude for the best cluster.
    """
    best_cluster_id = best_cluster[cluster_col]
    
    best_coords = df[df[cluster_col] == best_cluster_id][[lat_col, lon_col]].mean()
    
    return best_coords




def plot_idle_clusters_on_map(
    df,
    cluster_col='cluster',
    lat_col='lat',
    lon_col='lon',
    label_cols=None,
    zoom_start=12,
    only_cluster=None,
    show_legend=True,
    latest_week_only=False,
    week_col='week_number'
):
    """
    Plots idle location clusters on a Folium map, with optional filters for cluster, week, labels, and legend.

    Parameters:
        df (pd.DataFrame): Input DataFrame with cluster labels, coordinates, and (optionally) week numbers.
        cluster_col (str): Column name for cluster labels.
        lat_col (str): Latitude column name.
        lon_col (str): Longitude column name.
        label_cols (list of str, optional): Columns to include as marker popups.
        zoom_start (int): Initial zoom level.
        only_cluster (int, optional): If specified, only plot points for this cluster.
        show_legend (bool): Whether to show a color legend for clusters.
        latest_week_only (bool): If True, filter to only the latest week based on `week_col`.
        week_col (str): Column name containing the week number.

    Returns:
        folium.Map: A Folium map object.
    """
    df = df.copy()

    # Filter to latest week if requested
    if latest_week_only:
        if week_col not in df.columns:
            raise ValueError(f"Column '{week_col}' not found in DataFrame for filtering latest week.")
        latest_week = df[week_col].max()
        df = df[df[week_col] == latest_week]

    # Filter to specific cluster if requested
    if only_cluster is not None:
        df = df[df[cluster_col] == only_cluster]

    if df.empty:
        raise ValueError("No data available to plot after filtering.")

    # Map center
    map_center = [df[lat_col].mean(), df[lon_col].mean()]
    m = folium.Map(location=map_center, zoom_start=zoom_start)

    # Cluster colors
    color_palette = [
        'red', 'blue', 'green', 'purple', 'orange', 
        'darkred', 'cadetblue', 'darkblue', 'lightgreen', 
        'pink', 'brown', 'black'
    ]
    
    cluster_ids = sorted(df[cluster_col].unique())
    cluster_color_map = {
        cluster_id: ('gray' if cluster_id == -1 else color_palette[i % len(color_palette)])
        for i, cluster_id in enumerate(cluster_ids)
    }

    # Plot markers
    for _, row in df.iterrows():
        cluster_id = row[cluster_col]
        color = cluster_color_map[cluster_id]
        
        popup_text = ""
        if label_cols:
            popup_text = "<br>".join(f"{col}: {row[col]}" for col in label_cols if col in row)
        
        folium.CircleMarker(
            location=[row[lat_col], row[lon_col]],
            radius=3,
            color=color,
            fill=True,
            fill_opacity=0.5,
            popup=popup_text if popup_text else None
        ).add_to(m)

    # Add legend
    if show_legend:
        legend_html = """
        <div style="position: fixed; 
                    bottom: 50px; left: 50px; width: 150px; height: auto; 
                    background-color: white; z-index:9999; 
                    border:2px solid grey; padding: 10px;">
            <strong>Cluster Colors</strong><br>
        """
        for cluster_id, color in cluster_color_map.items():
            legend_html += f"""
            <i style="background:{color}; width:10px; height:10px; display:inline-block; margin-right:5px;"></i>
            Cluster {cluster_id}<br>
            """
        legend_html += "</div>"
        m.get_root().html.add_child(folium.Element(legend_html))

    return m


def generate_service_booking_recommendation_text(
    features_df,
    cluster_col='cluster',
    week_col='week_number',
    include_latest_week=True
):
    """
    Generates business-readable recommendations (as text) for the best idle time and location
    to book mobile services based on vehicle idle clustering.

    Parameters:
        features_df (pd.DataFrame): DataFrame with 'lat', 'lon', 'idle_start_hour', 
                                    'idle_duration_hr', 'week_number', and 'cluster'.
        cluster_col (str): Column with cluster labels.
        week_col (str): Column for identifying week number.
        include_latest_week (bool): If True, include a recommendation for the most recent week.

    Returns:
        str: Multi-line recommendation text for both overall and (optionally) latest week.
    """

    def format_recommendation(data, label):
        clustered_data = data[data[cluster_col] != -1]

        # If all points are noise
        if clustered_data.empty:
            clustered_data = data
            noise_only = True
        else:
            noise_only = False

        # Summarize clusters
        summary = clustered_data.groupby(cluster_col).agg({
            'idle_duration_hr': ['sum', 'mean', 'count']
        }).reset_index()
        summary.columns = [cluster_col, 'total_downtime', 'avg_downtime', 'frequency']

        best_cluster_id = summary.sort_values('total_downtime', ascending=False).iloc[0][cluster_col]
        best_cluster_data = clustered_data[clustered_data[cluster_col] == best_cluster_id]

        # Best location
        coords = best_cluster_data[['lat', 'lon']].mean()
        lat, lon = round(coords['lat'], 5), round(coords['lon'], 5)

        # Best time
        idle_hour_counts = best_cluster_data['idle_start_hour'].value_counts().sort_values(ascending=False)
        best_hour = idle_hour_counts.index[0]
        hour_count = idle_hour_counts.iloc[0]

        # Average idle duration
        avg_idle_duration = round(best_cluster_data['idle_duration_hr'].mean(), 2)

        cluster_note = (
            " (Note: only noise points present, using fallback analysis)"
            if noise_only else ""
        )

        return print(
            f"📊 Recommendation based on *{label} data*{cluster_note}:\n"
            f"- 📍 **Location:** Near latitude {lat}, longitude {lon}\n"
            f"- ⏰ **Best time to book service:** Around **{best_hour}:00**\n"
            f"- 🔁 **Idle occurrences at this time:** {hour_count}\n"
            f"- 🕒 **Expected idle duration:** Approximately **{avg_idle_duration} hours**\n"
        )

    # Start building output
    output = format_recommendation(features_df, label="overall")

    if include_latest_week:
        latest_week = features_df[week_col].max()
        latest_data = features_df[features_df[week_col] == latest_week]
        output += "\n" + format_recommendation(latest_data, label=f"latest week (week {latest_week})")

    return output
