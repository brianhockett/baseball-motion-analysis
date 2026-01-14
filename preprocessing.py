from scipy.interpolate import interp1d
import pandas as pd
import polars as pl
import numpy as np
import json

# Load in data using Polars
df = pl.read_parquet('./data/pitching_data.parquet')

print("Loading in pitch marker mapping")

# Load in pitch_marker_mapping
with open("./data/pitch_marker_mapping.json", 'r') as f:
    pitch_marker_mapping = json.load(f)

print("Converting pitch marker mapping into a Polars DataFrame")
# Unroll the JSON mapping into a DataFrame
mapping_records = []
for pitch_key, labels in pitch_marker_mapping.items():
    for marker_id, marker_label in enumerate(labels):
        mapping_records.append({
            "pitch_key": pitch_key,
            "markerID": marker_id,
            "markerLabel": marker_label
        })
mapping_df = pl.DataFrame(mapping_records)

# Create a key to merge data with marker mapping
df = df.with_columns(
    (
    df['userID'].cast(pl.Utf8) + '_' +
    df['sessionID'].cast(pl.Utf8) + '_' +
    df['pitchNum'].cast(pl.Utf8)
    ).alias('pitch_key')
)

print("Merging data with marker mapping")
# Merge data with marker mapping
df = df.join(
    mapping_df,
    on = ['pitch_key', 'markerID'],
    how = 'inner' # Only keeping markers that have a label
)

print("Replacing markerID number with markerLabel description as markerID")
# Drop original ID, replace and rename with label
df = df.drop(['markerID'])
df = df.with_columns(
    df['markerLabel'].str.split('_')
    .list.get(-1)
    .alias('markerLabel')
)
df = df.rename({'markerLabel': 'markerID'})

print("Loading in poi_metrics, getting unique session-key pitcher-handedness combinations")
# Loading points of interest data, which contains pitcher handedness
poi_metrics = pl.read_csv("./data/poi_metrics.csv")[["session_pitch", "session", "p_throws", "pitch_speed_mph"]]
poi_metrics = poi_metrics.with_columns(
    poi_metrics["session"].cast(pl.Utf8)
    .alias('session_key')
)
# Drop duplicates
poi_metrics_unique = poi_metrics.unique(subset = ['session_key', 'p_throws'])

print("Merging marker data with handedness")
# Merging marker data with pitcher handedness data
df = df.with_columns(
    df['sessionID'].cast(pl.Utf8)
    .str.replace(r"^0+", "")
    .alias('session_key')
)
df = df.join(
    poi_metrics_unique,
    on = 'session_key',
    how = 'left'
)

def swap_lr(label):
    if label.startswith("L"):
        return "R" + label[1:]
    if label.startswith("R"):
        return "L" + label[1:]
    return label

print("Mirroring left-handed pitchers about the y-axis")
# Mirror lefties about the y-axis -- Now every pitcher is treated as being the same handedness (right-handed)
mask = pl.col("p_throws") == "L"

df = df.with_columns([
    pl.when(mask)
      .then(pl.col("y") * -1)
      .otherwise(pl.col("y"))
      .alias("y"),

    pl.when(mask)
      .then(pl.col("markerID").map_elements(swap_lr,  return_dtype = pl.Utf8))
      .otherwise(pl.col("markerID"))
      .alias("markerID"),
])

print("Removing markers that are not present in every pitch")
# Identify unique markers per pitch
pitch_marker_sets = (
    df.group_by(["userID", "sessionID", "pitchNum"]).agg(
        pl.col('markerID').unique()
    )
)
# All markers in original data
all_markers = set(df["markerID"].unique())

# Intersect all marker sets to find markers present in every pitch
common_markers = set.intersection(*[set(x) for x in pitch_marker_sets["markerID"]])

# Markers that were excluded
excluded_markers = all_markers - common_markers
print(f"Removed {excluded_markers} from marker set for consistency")

# Filter data to only include markers present in every pitch
df = df.filter(pl.col('markerID').is_in(list(common_markers)))

# Dropping redundant columns
df = df.drop(['session_key', 'pitch_key', 'session_pitch', 'session', 'pitch_speed_mph'])

# Saving data to parquet file
print('Saving cleaned data to cleaned_data.parquet')
df.write_parquet('./data/cleaned_data.parquet')

# Function to normalize frames [0, 100] for each marker in a pitch using interpolation
def normalize_frames_polars_interpolate(df: pl.DataFrame, N_TARGET_FRAMES: int = 101) -> pl.DataFrame:
    all_groups = []

    # Target normalized frame axis
    frame_norm = np.linspace(0, 100, N_TARGET_FRAMES)

    # Loop over each pitch explicitly
    for pitch_key, group in df.group_by(["userID", "sessionID", "pitchNum"]):
        # Process each marker separately
        for marker, marker_df in group.partition_by("markerID", as_dict=True).items():
            marker_df = marker_df.sort("frame")
            n_frames = marker_df.height
            if n_frames == 1:
                # single frame → replicate values
                interp_x = np.full(N_TARGET_FRAMES, marker_df["x"][0])
                interp_y = np.full(N_TARGET_FRAMES, marker_df["y"][0])
                interp_z = np.full(N_TARGET_FRAMES, marker_df["z"][0])
            else:
                # Original frame axis scaled 0–100
                orig_frames = np.linspace(0, 100, n_frames)
                interp_x = np.interp(frame_norm, orig_frames, marker_df["x"].to_numpy())
                interp_y = np.interp(frame_norm, orig_frames, marker_df["y"].to_numpy())
                interp_z = np.interp(frame_norm, orig_frames, marker_df["z"].to_numpy())

            # Build new DataFrame for this marker
            new_df = pl.DataFrame({
                "userID": pl.Series([marker_df["userID"][0]] * N_TARGET_FRAMES),
                "sessionID": pl.Series([marker_df["sessionID"][0]] * N_TARGET_FRAMES),
                "height": pl.Series([marker_df["height"][0]] * N_TARGET_FRAMES),
                "weight": pl.Series([marker_df["weight"][0]] * N_TARGET_FRAMES),
                "pitchNum": pl.Series([marker_df["pitchNum"][0]] * N_TARGET_FRAMES),
                "pitchType": pl.Series([marker_df["pitchType"][0]] * N_TARGET_FRAMES),
                "pitchSpeed": pl.Series([marker_df["pitchSpeed"][0]] * N_TARGET_FRAMES),
                "frame_norm": pl.Series(frame_norm).cast(pl.Int64),
                "x": pl.Series(interp_x),
                "y": pl.Series(interp_y),
                "z": pl.Series(interp_z),
                "markerID": pl.Series([marker][0] * N_TARGET_FRAMES),
                "p_throws": pl.Series([marker_df["p_throws"][0]] * N_TARGET_FRAMES),
            })
            all_groups.append(new_df)

    # Concatenate all markers and pitches
    return pl.concat(all_groups)

norm = normalize_frames_polars_interpolate(df)

# Save normalized data to parquet file
print('Saving normalized data to normalized_data.parquet')
norm.write_parquet('./data/normalized_data.parquet')

