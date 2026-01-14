# Baseball Pitching Motion Analysis

## Data Overview

`pitching_data.parquet` - Raw 3D motion capture data where each row represents a single frame of a pitcher's motion. Columns include: 'userID', 'sessionID', 'height', 'weight', 'pitchNum', 'pitchType', 'pitchSpeed', 'frame', 'markerID', 'x', 'y', 'z'

`pitch_marker_mapping.json` - JSON mapping that associates markerID numbers with descriptive labels for anatomical body part locations (e.g., "Left_Shoulder", "Right_Elbow"). Organized by pitch_key (composite key of userID_sessionID_pitchNum).

`poi_metrics.csv` - Points of interest metrics data containing columns: 'session_pitch', 'session', 'p_throws', 'pitch_speed_mph'. Used to extract pitcher handedness (p_throws: "L" or "R") for each session.

`cleaned_data.parquet` - Processed motion capture data after cleaning and standardization. All left-handed pitchers are mirrored about the y-axis so all pitchers are normalized as right-handed. Only includes markers that appear in every pitch within a session. Columns: 'userID', 'sessionID', 'height', 'weight', 'pitchNum', 'pitchType', 'pitchSpeed', 'frame', 'markerID', 'x', 'y', 'z', 'p_throws'

`normalized_data.parquet` - Final processed dataset with frame normalization applied. Each marker trajectory within a pitch is interpolated to a standard 101 frames (0-100 normalized frame scale). Columns: 'userID', 'sessionID', 'height', 'weight', 'pitchNum', 'pitchType', 'pitchSpeed', 'frame_norm', 'x', 'y', 'z', 'markerID', 'p_throws'

`df` in `pca.ipynb` - Wide-format feature matrix prepared for machine learning and dimensionality reduction. Each row represents a single pitch, with one row per unique combination of userID, sessionID, pitchNum, height, weight, pitchType, pitchSpeed, and p_throws. Features are constructed by pivoting the normalized motion data into columns with naming convention {axis}_{markerID}_{frame_norm} (e.g., x_C7_0, y_LeftShoulder_50, z_RightElbow_100), where axis is x, y, or z coordinate, markerID is the body part label, and frame_norm is the normalized frame number (0-100). This format enables direct input to PCA and other statistical modeling techniques.