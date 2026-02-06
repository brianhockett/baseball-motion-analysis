import polars as pl
import numpy as np
import json
import re
from pathlib import Path

# Use lazy evaluation where possible
def create_pitch_df(df):
    """Create a Polars DataFrame where each row is a pitch"""
    pitch_df = df.pivot(
        index=["userID", "sessionID", "pitchNum", "height", "weight", "pitchType", "pitchSpeed", "p_throws"],
        on=['markerID', 'frame_norm'],
        values=['x', 'y', 'z']
    )
    
    # Rename columns more efficiently
    pattern = re.compile(r'([xyz])_\{"([^"]+)",(\d+)\}')
    rename_dict = {}
    
    for col in pitch_df.columns:
        if col[0] in ('x', 'y', 'z'):
            match = pattern.match(col)
            if match:
                axis, marker, frame = match.groups()
                new_col_name = f"{axis}_{marker}_{frame}"
                rename_dict[col] = new_col_name
    
    if rename_dict:
        pitch_df = pitch_df.rename(rename_dict)
    
    return pitch_df

def convert_real_df_to_json(df: pl.DataFrame, output_filename="./data/real_data.json"):
    """Convert dataframe to JSON format"""
    # Cast types upfront
    df = df.with_columns([
        pl.col("frame_norm").cast(pl.Int64),
        pl.col("pitchNum").cast(pl.Int64),
        pl.col("x").cast(pl.Float64),
        pl.col("y").cast(pl.Float64),
        pl.col("z").cast(pl.Float64),
    ])
    
    # Group and aggregate
    grouped = df.group_by(["userID", "sessionID", "pitchNum"]).agg([
        pl.first("pitchType").alias("pitchType"),
        pl.first("pitchSpeed").alias("pitchSpeed"),
        (pl.max("frame_norm") + 1).cast(pl.Int64).alias("max_frames"),
        pl.struct(["frame_norm", "markerID", "x", "y", "z"]).alias("points")
    ])
    
    pitches = []
    
    # More efficient iteration
    for row in grouped.iter_rows(named=True):
        frames = {}
        for p in row["points"]:
            f = int(p["frame_norm"])
            marker = p["markerID"]
            
            if f not in frames:
                frames[f] = {}
            
            frames[f][marker] = {
                "x": float(p["x"]),
                "y": float(p["y"]),
                "z": float(p["z"])
            }
        
        pitches.append({
            "id": f"{int(row['userID'])}_{int(row['sessionID'])}_{int(row['pitchNum'])}",
            "label": f"U:{row['userID']} S:{row['sessionID']} P:{row['pitchNum']} "
                     f"({row['pitchType']} @ {row['pitchSpeed']}mph)",
            "max_frames": int(row["max_frames"]),
            "frames": frames
        })
    
    # Write JSON
    with open(output_filename, "w") as f:
        json.dump(pitches, f)

def main():
    print("Loading data...")
    norm = pl.read_parquet('./data/normalized_data.parquet')
    
    print("Cleaning data...")
    # Check and remove nulls in one step
    norm = norm.filter(~pl.any_horizontal(pl.col(pl.Float64).is_null() | pl.col(pl.Float64).is_nan()))
    
    print("Creating PCA-ready dataframe...")
    df = create_pitch_df(norm)
    
    print("Writing PCA data...")
    df.write_parquet('./data/pca_ready.parquet')
    
    print("Converting to real data JSON...")
    convert_real_df_to_json(norm)
    
    print("Done!")

if __name__ == "__main__":
    main()