import pandas as pd
import numpy as np
import json

print("Loading in original pitcher data")
# Load in original data
data = pd.read_parquet("./data/pitching_data.parquet")

print("Loading in pitch marker mapping")
# Load in pitch_marker_mapping
with open("./data/pitch_marker_mapping.json", 'r') as f:
    pitch_marker_mapping = json.load(f)

print("Converting pitch marker mapping into a dataframe")
# Unroll the JSON mapping into a DataFrame
mapping_records = []
for pitch_key, labels in pitch_marker_mapping.items():
    for marker_id, marker_label in enumerate(labels):
        mapping_records.append({
            "pitch_key": pitch_key,
            "markerID": marker_id,
            "markerLabel": marker_label
        })
mapping_df = pd.DataFrame(mapping_records)

# Create a key to merge data with marker mapping
data['pitch_key'] = (
    data['userID'].astype(str) + '_' +
    data['sessionID'].astype(str) + '_' +
    data['pitchNum'].astype(str)
)

print("Merging data with marker mapping")
# Merge data with marker mapping
data = pd.merge(
    data,
    mapping_df,
    on = ['pitch_key', 'markerID'],
    how ='inner' # Only keeping markers that have a label
)

print("Replacing markerID number with markerLabel description as markerID")
# Drop original ID, replace and rename with label
data.drop(columns=['markerID'], inplace = True)
data['markerLabel'] = data['markerLabel'].str.split('_').str[-1]
data.rename(columns={'markerLabel': 'markerID'}, inplace = True)

print("Loading in poi_metrics, getting unique session-key pitcher-handedness combinations")
# Loading points of interest data, which contains pitcher handedness
poi_metrics = pd.read_csv("./data/poi_metrics.csv")[["session_pitch", "session", "p_throws", "pitch_speed_mph"]]
poi_metrics["session_key"] = poi_metrics['session'].astype(str)
poi_metrics_unique = poi_metrics[["session_key", "p_throws"]].drop_duplicates()

print("Merging marker data with handedness")
# Merging marker data with pitcher handedness data
data["session_key"] = (data["sessionID"].astype(str).str.lstrip("0"))
data = data.merge(
    poi_metrics_unique,
    on = "session_key",
    how = "left"
)

print("Mirroring left-handed pitchers about they y-axis")
# Mirror lefties about the y-axis -- Now every pitcher is treated as being the same handedness (right-handed)
data.loc[data['p_throws'] == 'L', 'y'] *= -1

print("Removing markers that are not present in every pitch")
# Identify unique markers per pitch
pitch_marker_sets = (
    data.groupby(["userID", "sessionID", "pitchNum"])["markerID"]
    .unique()
)

# All markers in original data
all_markers = set(data["markerID"].unique())

# Intersect all marker sets to find markers present in every pitch
common_markers = set.intersection(*map(set, pitch_marker_sets))

# Markers that were excluded
excluded_markers = all_markers - common_markers
print(f"Removed {excluded_markers} from marker set for consistency")

# Filter data to only include markers present in every pitch
data = data[data["markerID"].isin(common_markers)]

# Dropping redundant columns
data = data.drop(columns = ["pitch_key", "session_key"])

print("Saving cleaned and enriched data to cleaned_data.parquet")
# Save cleaned and enriched data to cleaned_data.parquet
data.to_parquet("./data/cleaned_data.parquet", index = False)


print("Making pivot dataframe with each row having x_marker, y_marker, z_marker values for each marker")
# Getting pivot table
pivot_df = data.pivot(
    index = ["userID", "sessionID", "pitchNum", "height", "weight", "frame", "pitchType", "pitchSpeed", "p_throws"],
    columns = "markerID",
    values = ["x", "y", "z"]
)

# Put columns into single layer
pivot_df.columns = [f"{coord}_{marker}" for coord, marker in pivot_df.columns]
pivot_df.dropna(inplace = True)

print("Centering pitch motions on the y-axis")
# Get y-coordinate column names
y_cols = [c for c in pivot_df.columns if c.startswith("y_")]

# Compute each pitch's first-frame mean Y
first_frame_mean_y = (
    pivot_df
    .groupby(["userID", "sessionID", "pitchNum"])[y_cols]
    .transform(lambda g: g.iloc[0].mean())
)

# Subtract that baseline so each pitch starts at mean(y)=0
pivot_df[y_cols] = pivot_df[y_cols] - first_frame_mean_y

print("Saving pivot dataframe to pivot.parquet")
# Save pivot dataframe to pivot.parquet
pivot_df.to_parquet("./data/pivot.parquet")



