import ezc3d
import pandas as pd
import os
import json
import numpy as np

# Directory containing C3D files
data_dir = './data/c3d_files'
data = []
pitch_marker_mapping = {}  # (userID, sessionID, pitchNum) -> [marker_labels]

# List all C3D files first
files = [f for f in os.listdir(data_dir) if f.endswith('.c3d')]
total_files = len(files)
file_number = 1
# Iterate over all C3D files in the directory
for file in os.listdir(data_dir):
    if file.endswith('.c3d'):
        file_path = os.path.join(data_dir, file)

        # Load the C3D file
        c3d = ezc3d.c3d(file_path)
        points = c3d["data"]["points"]  # shape = (4, nMarkers, nFrames)


        # Extract metadata from filename
        parts = file.split('_')
        userID     = parts[0]
        sessionID  = parts[1]
        height     = parts[2]
        weight     = parts[3]
        pitchNum   = parts[4]
        pitchType  = parts[5]
        pitchSpeed = parts[6].split('.')[0]

        n_markers = points.shape[1]
        n_frames = points.shape[2]

        try:
            labels = c3d['parameters']['POINT']['LABELS']['value']
        except Exception as e:
            labels = [f'marker_{i}' for i in range(points.shape[1])]

        pitch_key = f"{userID}_{sessionID}_{pitchNum}"
        pitch_marker_mapping[pitch_key] = labels

        # Iterate over frames and markers to collect data
        for frame in range(n_frames):
            for marker_id in range(n_markers):
                x = points[0, marker_id, frame]
                y = points[1, marker_id, frame]
                z = points[2, marker_id, frame]
                residual = points[3, marker_id, frame]

                data.append([
                    userID, sessionID, height, weight,
                    pitchNum, pitchType, pitchSpeed,
                    frame, marker_id, x, y, z
                ])

        print(f"File {file_number}/{total_files} loaded into parquet file")
        file_number += 1

# Create DataFrame
df = pd.DataFrame(
    data,
    columns=[
        'userID','sessionID','height','weight',
        'pitchNum','pitchType','pitchSpeed',
        'frame','markerID','x','y','z'
    ]
)

# Fix pitchSpeed to be in mph
df["pitchSpeed"] = pd.to_numeric(df["pitchSpeed"], errors='coerce') / 10
df.to_parquet('./data/pitching_data.parquet', index=False)
print(f"Saved DataFrame to './data/pitching_data.parquet', total rows: {len(df)}")

with open('./data/pitch_marker_mapping.json', 'w') as f:
    json.dump(pitch_marker_mapping, f, indent=2)
print("Saved pitch marker labels to './data/pitch_marker_mapping.json'")