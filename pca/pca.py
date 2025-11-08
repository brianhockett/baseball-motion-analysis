from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.interpolate import interp1d
from scipy.signal import resample
import pandas as pd
import numpy as np
import joblib

pivot_df = pd.read_parquet("../data/pivot.parquet")

print("Fitting PCA")
# Doing PCA with 3 components
X = pivot_df.values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
pca = PCA(n_components = 3)

# Fitting the PCA
scores = pca.fit_transform(X_scaled)
print("Cumulative explained variance:", np.cumsum(pca.explained_variance_ratio_)*100)

# Save PCA and scaler
joblib.dump(pca, "./data/pca_model.joblib")
joblib.dump(scaler, "./data/scaler_model.joblib")

# Making dataframe to hold reconstruction data
scores_df = pivot_df.copy().reset_index()

# Adding score for each PCA
for i in range(scores.shape[1]):
    scores_df[f"PC{i+1}"] = scores[:, i]

# Drop marker columns for clarity
feature_cols = [col for col in scores_df.columns if col.startswith(('x_', 'y_', 'z_'))]
scores_df.drop(columns = feature_cols, inplace = True)

# Save PCA results to pca_results.parquet
print("Save PCA results as pca_results.parquet")
scores_df.to_parquet("./data/pca_results.parquet")

## Normalizing Motion to 100 frames

# Number of frames wanted
N_TARGET_FRAMES = 101

def normalize_raw(group):
    # Get frames 0 to 100
    frame_norm = np.linspace(0, 100, N_TARGET_FRAMES)
    normalized_data = {"frame_norm": frame_norm}

    # Identify all motion coordinate columns (x_*, y_*, z_*)
    motion_cols = [c for c in group.columns if c.startswith(("x_", "y_", "z_"))]

    # Original frame axis (assumed sequential)
    orig_frames = np.linspace(0, 100, len(group))

    # Interpolate each coordinate column
    for col in motion_cols:
        f = interp1d(orig_frames, group[col].values, kind="linear")
        normalized_data[col] = f(frame_norm)

    # Add metadata
    for meta_col in ["userID", "sessionID", "pitchNum", "height", "weight", "pitchType", "pitchSpeed"]:
        normalized_data[meta_col] = group[meta_col].iloc[0]

    return pd.DataFrame(normalized_data)

# Normalizing data to be viewed as 0-100% of motion
def normalize_pca(group):
    # Get frames 0 to 100
    frame_norm = np.linspace(0, 100, N_TARGET_FRAMES)
    normalized_data = {"frame_norm": frame_norm}

    # Original frame axis (assumed sequential)
    orig_frames = np.linspace(0, 100, len(group))

    # Resample PCA values according to normalized frames
    for pc in ["PC1", "PC2", "PC3"]:
        f = interp1d(orig_frames, group[pc].values, kind = 'linear')
        normalized_data[pc] = f(frame_norm)

    # Add metadata
    for col in ["userID", "sessionID", "pitchNum", "height", "weight", "pitchType", "pitchSpeed"]:
        normalized_data[col] = group[col].iloc[0]

    return pd.DataFrame(normalized_data)

# Get normalized raw dataframe
normalized_raw = (
    pivot_df.reset_index()
    .groupby(["userID", "sessionID", "pitchNum", "height", "weight", "pitchType", "pitchSpeed"], group_keys=False)
    .apply(normalize_raw)
)

# Saving normalized raw data
print("Save normalized PCA results as normalized_pca.parquet")
normalized_raw.to_parquet("./data/normalized_raw.parquet")

# Get normalized pca dataframe
normalized_pca = (
    scores_df
    .groupby(["userID", "sessionID", "pitchNum", "height", "weight", "pitchType", "pitchSpeed"], group_keys=False)
    .apply(normalize_pca)
)

# Saving normalized PCA results
print("Save normalized PCA results as normalized_pca.parquet")
normalized_pca.to_parquet("./data/normalized_pca.parquet")