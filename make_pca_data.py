import polars as pl
import re

norm = pl.read_parquet('./data/normalized_data.parquet')

# Making Polars DataFrame where each row is a pitch
    # (i.e. x_marker_frame, y_marker_frame, z_marker_frame are columns made from x, y, z, frame, and markerID columns of df)
def create_pitch_df(df):
    # Create a Polars DataFrame where each row is a pitch
    pitch_df = df.pivot(
        index = ["userID", "sessionID", "pitchNum", "height", "weight", "pitchType", "pitchSpeed", "p_throws"],
        on = ['markerID', 'frame_norm'],
        values = ['x', 'y', 'z']
    )
    
    # Pattern to match x_{"C7",0} or similar
    pattern = re.compile(r'([xyz])_\{"([^"]+)",(\d+)\}')

    # Change from naming convention of x_{"C7",0} to x_C7_0 
    for col in pitch_df.columns:
        if col.startswith('x_') or col.startswith('y_') or col.startswith('z_'):
            match = pattern.match(col)
            axis, marker, frame = match.groups()
            new_col_name = f"{axis}_{marker}_{frame}"
            pitch_df = pitch_df.rename({col: new_col_name})

    return pitch_df

# Create PCA-ready DataFrame
pitch_df = create_pitch_df(norm)

# Save PCA-ready DataFrame to parquet file
print('Saving PCA-ready data to pca_ready_data.parquet')
pitch_df.write_parquet('./data/pca_ready_data.parquet', compression = 'lz4')