# Imports
import matplotlib.pyplot as plt
import polars as pl
import pandas as pd
import numpy as np
import json
import re

print("Loading data...")
# Load normalized data and PCA-ready DataFrame from parquet files
norm = pl.read_parquet('./data/normalized_data.parquet')
df = pl.read_parquet('./data/pca_ready.parquet')

# Filter rows where any value is null or NaN in feature columns (starting at column 8)
# This ensures clean data for PCA computation
df = df.filter(
    ~(
    pl.any_horizontal(
        pl.nth(range(8, len(df.columns))).is_null() | 
        pl.nth(range(8, len(df.columns))).is_nan()
    )
    )
)

# Extract feature columns and transpose to shape (n_features, n_pitches)
# Each row represents a feature, each column represents a pitch observation
X = df.select(pl.nth(range(8, len(df.columns)))).to_numpy().T

# Calculate mean across all pitches and center the data
# Centering is essential for PCA to work correctly
X_mean = np.mean(X, axis=1, keepdims=True)
X_centered = X - X_mean

# Compute covariance matrix and perform eigenvalue decomposition
# Eigenvalues represent variance along each principal component direction
cov_mat = (X_centered.T @ X_centered) / (X_centered.shape[1] - 1)
eigenvalues, eigenvectors = np.linalg.eigh(cov_mat)

# Sort eigenvalues and eigenvectors in descending order by magnitude
# This orders components by how much variance they explain
idx = eigenvalues.argsort()[::-1]
eigenvalues = eigenvalues[idx]
eigenvectors = eigenvectors[:, idx]

print(f"Covariance matrix shape: {cov_mat.shape}")
print("Top 5 eigenvalues:")
for i, eval, evec in zip(range(len(eigenvectors)), eigenvalues, eigenvectors.T):
    if i < 5:
        print(f"  PC{i+1}: {np.round(eval,3)}")

# Calculate explained variance ratio and cumulative variance
# This helps determine how many components we need to retain signal
total_variance = np.sum(eigenvalues)
explained_variance_ratio = eigenvalues / total_variance
cumulative_variance = np.cumsum(explained_variance_ratio)

# Find the number of components needed to explain 80% of variance
print("\nFinding components for 80% variance threshold...")
for i, (eval, ratio) in enumerate(zip(eigenvalues, explained_variance_ratio)):
    if cumulative_variance[i] > 0.80:
        print(f"80% variance reached at PC{i+1} with {cumulative_variance[i] * 100:.2f}% total")
        # Extract the top k principal components and their eigenvalues
        k = i + 1
        U = eigenvectors[:, :k]
        Lambda = eigenvalues[:k]
        # Small epsilon prevents division by zero in downstream calculations
        eps = 1e-12
        # Compute principal component scores by projecting centered data onto principal components
        V = X_centered @ U / np.sqrt((X_centered.shape[1] - 1) * (Lambda + eps))
        break

print("Exporting PCA data to JSON...")
# Package all PCA results for export to visualization
pca_export = {
    'X_mean': X_mean.squeeze().tolist(),
    'V': V.tolist(),
    'Lambda': Lambda.tolist(),
    'feature_names': df.columns[8:],
    'marker_labels': norm.select(pl.col('markerID')).unique().to_series().to_list(),
    'n_frames': 101,
}

# Save PCA metadata to JSON for visualization to load
with open('data/pca_data.json', 'w') as f:
    json.dump(pca_export, f)


def get_pca_scores_for_pitch(pitch_features, X_mean, V_components, Lambda):
    """
    Compute variance-standardized PCA scores for a single pitch.
    Projects the pitch onto the principal components and normalizes by explained variance.
    """
    # Center the pitch using global mean
    pitch_vector = pitch_features.reshape(-1, 1)
    pitch_centered = pitch_vector - X_mean
    
    # Project onto principal components to get raw scores
    raw_scores = V_components.T @ pitch_centered
    
    # Normalize by explained variance (eigenvalues) to get z-scores
    z_scores = raw_scores.flatten() / np.sqrt(Lambda)
    
    return z_scores


def compute_pca_scores_dataframe(df_pitches, X_mean, V_components, Lambda, n_training_samples):
    """
    Compute PCA scores for all pitches in the dataframe.
    Returns a dataframe with metadata columns and PC score columns.
    """
    # Extract metadata and feature columns
    feature_cols = df_pitches.columns[8:]
    metadata_cols = ["userID", "sessionID", "pitchNum", "pitchSpeed"]
    
    metadata = df_pitches.select(metadata_cols)
    features_np = df_pitches.select(feature_cols).to_numpy()
    
    # Compute PCA scores for each pitch
    pca_scores_list = []
    for pitch_features in features_np:
        scores = get_pca_scores_for_pitch(pitch_features, X_mean, V_components, Lambda)
        pca_scores_list.append(scores)
    
    # Convert scores array to dataframe with PC columns
    pca_scores_array = np.array(pca_scores_list)
    k = pca_scores_array.shape[1]
    pc_column_names = [f"PC{i+1}_Score" for i in range(k)]
    
    pc_df = pl.DataFrame({
        pc_column_names[i]: pca_scores_array[:, i].tolist() 
        for i in range(k)
    })
    
    # Combine metadata with PC scores
    result_df = pl.concat([metadata, pc_df], how="horizontal")
    
    return result_df


def export_pca_scores_to_json(pca_df, output_filename="data/pca_scores.json"):
    """
    Export PCA scores to JSON format for use in HTML visualization.
    Matches the non-padded ID format used in real_data.json.
    """
    # Extract PC score columns from dataframe
    pc_columns = [col for col in pca_df.columns if col.startswith('PC') and col.endswith('_Score')]
    
    pca_scores = []
    for row in pca_df.iter_rows(named=True):
        # Format ID to match the simple format: "1_1_5"
        # We use int() to strip any leading zeros that might exist in the raw data
        u_id = int(row['userID'])
        s_id = int(row['sessionID'])
        p_num = int(row['pitchNum'])
        pitch_id = f"{u_id}_{s_id}_{p_num}"
        
        pc_dict = {
            'id': pitch_id,
            'userID': str(row['userID']),
            'sessionID': str(row['sessionID']),
            'pitchNum': int(row['pitchNum']),
            'pitchSpeed': float(row['pitchSpeed']),
            'pc_scores': [float(row[col]) for col in pc_columns]
        }
        pca_scores.append(pc_dict)
    
    # Write to JSON file
    with open(output_filename, 'w') as f:
        json.dump(pca_scores, f, indent=2)
    
    print(f"Exported {len(pca_scores)} PCA scores to {output_filename}")


# Main execution: compute and save PCA scores
print("\nComputing PCA scores for all pitches...")

# Recompute V_components with the selected k components
eps = 1e-12
V_components = X_centered @ U / np.sqrt((X_centered.shape[1] - 1) * (Lambda + eps))

# Generate PCA scores dataframe for all pitches
pca_df = compute_pca_scores_dataframe(df, X_mean, V_components, Lambda, X_centered.shape[1])

# Export scores to JSON for visualization
export_pca_scores_to_json(pca_df, "data/pca_scores.json")


print("Creating and Saving PC - Pitch Speed Correlation Plot")
# Get PC column names (all columns from index 4 onwards)
pc_columns = pca_df.columns[4:]

# Bootstrapping Parameters
N_BOOTSTRAP = 5000
CI = 95
alpha = (100 - CI) / 2

# Random number generator
rng = np.random.default_rng(42)
n_rows = len(pca_df)

# Dictionary to hold bootstrap correlation results for each PC
boot_corrs = {col: [] for col in pc_columns}

# Bootstrap sampling and correlation calculation
for _ in range(N_BOOTSTRAP):
    idx = rng.integers(0, n_rows, size=n_rows)
    sample = pca_df[idx]
    for col in pc_columns:
        corr = sample.select(pl.corr(col, 'pitchSpeed')).item()
        boot_corrs[col].append(corr)

# Calculate means and confidence intervals for each PC's correlation with pitch speed
means      = [np.mean(boot_corrs[col]) for col in pc_columns]
lowers     = [np.percentile(boot_corrs[col], alpha) for col in pc_columns]
uppers     = [np.percentile(boot_corrs[col], 100 - alpha) for col in pc_columns]
errors_neg = [m - l for m, l in zip(means, lowers)]
errors_pos = [u - m for m, u in zip(means, uppers)]

# Initialize the plot
fig, ax = plt.subplots(figsize=(12, 7))

# X positions for each PC
x = list(range(len(pc_columns)))

# Error bars
ax.errorbar(x, means,
            yerr=[errors_neg, errors_pos],
            fmt='none',
            color='#C9A535',
            linewidth=2,
            capsize=6,
            capthick=2,
            zorder=2,
            label=f'{CI}% Bootstrap CI')

# Mean points
ax.scatter(x, means,
           color='#1f2f45',
           s=80,
           zorder=3,
           edgecolors='black',
           linewidths=0.8,
           label='Bootstrap mean')

# Labels to the right of each mean point
for i, mean in enumerate(means):
    ax.text(i + 0.15, mean,
            f'{mean:.3f}',
            ha='left',
            va='center',
            fontsize=10, fontweight='bold', color='#333333')

# Styling
ax.set_xlabel('Principal Component', fontsize=13, fontweight='bold')
ax.set_ylabel('Correlation with Pitch Speed', fontsize=13, fontweight='bold')
ax.set_title(f'PCA Component Correlation with Pitch Speed\n'
             f'(Bootstrapped Means ± {CI}% CI, n = {N_BOOTSTRAP})',
             fontsize=15, fontweight='bold', pad=20)
ax.set_xticks(x)
ax.set_xticklabels([col.replace('_Score', '') for col in pc_columns], fontsize=11)
ax.axhline(y=0, color='black', linestyle='-', linewidth=1.5)
ax.grid(axis='y', alpha=0.5, linestyle='--', linewidth=0.7, zorder=1)
ax.set_ylim(min(lowers) - 0.15, max(uppers) + 0.15)
ax.legend(fontsize=11, loc='upper right')

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_linewidth(1.5)
ax.spines['bottom'].set_linewidth(1.5)

plt.tight_layout()
plt.savefig('./images/pc_speed_correlation_bootstrapped.png', dpi=300, bbox_inches='tight')

print("Creating and Saving PCA Reconstruction Plot")

scores_all = pca_df.select(pl.col("^PC.*_Score$")).to_numpy()
reconstructed_all = X_mean.T + (scores_all * np.sqrt(Lambda)) @ V_components.T
real_all = df.select(pl.nth(range(8, len(df.columns)))).to_numpy()

def row_corr(A, B):
    ma, mb = A.mean(1, keepdims = True), B.mean(1, keepdims = True)
    am, bm = A - ma, B - mb
    return (am * bm).sum(1) / np.sqrt((am**2).sum(1) * (bm**2).sum(1))

corrs = row_corr(real_all, reconstructed_all)
rmses = np.sqrt(((real_all - reconstructed_all)**2).mean(1)) / real_all.std(1)

# Creating subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (16, 7))

# Styling parameters
hist_kwargs = {
    'bins': 50, 
    'color': '#1f2f45', 
    'alpha': 0.8, 
    'edgecolor': 'black', 
    'linewidth': 1.2
}

# Correlation between reconstructed motions and real motions plot
ax1.hist(corrs, **hist_kwargs)
ax1.axvline(corrs.mean(), color = '#dba86f', linestyle = '--', linewidth = 2, 
            label = f'Mean: {corrs.mean():.3f}')

ax1.set_title('Reconstruction Correlation', fontsize = 15, fontweight = 'bold', pad = 15)
ax1.set_xlabel('Correlation Coefficient', fontsize = 13, fontweight = 'bold')
ax1.set_ylabel('Pitch Count', fontsize = 13, fontweight = 'bold')

# Normalized RMSE plot
ax2.hist(rmses, **hist_kwargs)
ax2.axvline(rmses.mean(), color = '#dba86f', linestyle = '--', linewidth = 2, 
            label = f'Mean: {rmses.mean():.3f}')

ax2.set_title('Normalized RMSE', fontsize = 15, fontweight = 'bold', pad = 15)
ax2.set_xlabel('nRMSE (Standardized)', fontsize = 13, fontweight = 'bold')

# Apply consistent styling to both axes
for ax in [ax1, ax2]:
    ax.grid(axis = 'y', alpha = 0.5, linestyle = '--', linewidth = 0.7)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['bottom'].set_linewidth(1.5)
    ax.legend(frameon = False, fontsize = 11)

plt.tight_layout()
plt.savefig('./images/pca_reconstruction.png', dpi = 300, bbox_inches = 'tight')