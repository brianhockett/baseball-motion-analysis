import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns

def plot_pca(normalized_df, poi_to_color, poi_to_pcs, pitches_of_interest):
    pcs = ['PC1', 'PC2', 'PC3']
    
        # Initialize subplots
    fig, axes = plt.subplots(nrows = 3, ncols = 1, figsize = (14, 16), sharex = True)

    # Loop through each PC
    for i, pc in enumerate(pcs):
        ax = axes[i]

        # Plot mean line
        mean_pc = normalized_df.groupby("frame_norm")[pc].mean()
        ax.plot(mean_pc.index, mean_pc.values, color = "black", linewidth = 3, label = "PC Mean", zorder = 3)

        # Loop over each pitch
        for (sess, uid, pid, height, weight, pitchType, pitchSpeed), grp in normalized_df.groupby(
            ["sessionID", "userID", "pitchNum", "height", "weight", "pitchType", "pitchSpeed"], sort=False):

            key = (uid, sess, pid)
            # Color based on whether pitch is a pitch of interest (min or max for the PC)
            if key in poi_to_color and pc in poi_to_pcs[key]:
                color = poi_to_color[key]
                lw = 2
                zorder = 2
                label = f"POI {pitches_of_interest.index(key)+1}"
            else:
                color = "lightgray"
                lw = 1
                zorder = 1
                label = ''

            # Plot pitch line
            ax.plot(grp["frame_norm"], grp[pc], color=color, linewidth=lw, zorder=zorder, label=label)

        ax.set_title(f"{pc}", loc="left")
        sns.despine(ax = ax, top = True, right = True)

    # Create a collective legend
    handles, labels = [], []
    for poi in pitches_of_interest:
        handles.append(ax.plot([], [], color = poi_to_color[poi], lw = 2)[0])
        labels.append(f"POI {pitches_of_interest.index(poi) + 1}: {poi}")
    handles.append(ax.plot([], [], color="black", lw = 3)[0])
    labels.append("PC Mean")

    fig.legend(handles, labels, loc = "upper center", frameon = False, ncol = 3,
            bbox_to_anchor=(0.5, 0.99))

    axes[-1].set_xlabel("Frame (Normalized %)")
    fig.suptitle("PCA Trajectories over Normalized Pitch Frames", ha="center", weight='bold', y = 1)
    plt.tight_layout()
    plt.savefig("./plots/pca_plot.png")
    plt.show()

def plot_diffs(merged, poi_to_color, poi_to_pcs, poi_records):
    pcs = ['PC1', 'PC2', 'PC3']
    # LaTeX for w_i
    deltas = [r"$\omega_1$", r"$\omega_2$", r"$\omega_3$"]
    pca_to_delta = dict(zip(pcs, deltas))

    # Initialize subplots
    fig, axes = plt.subplots(nrows = 3, ncols = 1, figsize = (14,16), sharex = True)

    # Loop over each PC
    for i, pc in enumerate(pcs):
        ax = axes[i]

        # Plot mean line
        mean_pc = merged.groupby("frame_norm")[f"{pc}_diff"].mean()
        ax.plot(mean_pc.index, mean_pc.values, color="black", linewidth=2, label="Mean", zorder=3)

        # Loop over each pitch
        for (sess, uid, pid, height, weight, pitchType, pitchSpeed), grp in merged.groupby(
            ["sessionID","userID","pitchNum","height","weight","pitchType","pitchSpeed"], sort=False):

            key = (uid, sess, pid)
            # Color based on whether pitch is a POI (min/max for that PC)
            if key in poi_to_color and pc in poi_to_pcs[key]:
                color = poi_to_color[key]
                lw = 2
                zorder = 2
                label = f"POI {poi_records.index(next(r for r in poi_records if r['pitch_of_interest']==key))+1}"
            else:
                color = "lightgray"
                lw = 1
                zorder = 1
                label = ''

            # Plot pitch line
            ax.plot(grp["frame_norm"], grp[f"{pc}_diff"], color=color, linewidth=lw, zorder=zorder, label=label)

        ax.set_title(f"{pca_to_delta[pc]}", loc="left")
        sns.despine(ax=ax, top=True, right=True)

    # Create a collective legend
    handles, labels = [], []
    for r in poi_records:
        poi = r["pitch_of_interest"]
        handles.append(ax.plot([], [], color = poi_to_color[poi], lw = 2)[0])
        labels.append(f"POI {r['POI']}: {poi}")
    handles.append(ax.plot([], [], color = "black", lw = 2)[0])
    labels.append("Mean")

    fig.legend(handles, labels, loc = "upper center", frameon = False, ncol = 3, bbox_to_anchor = (0.5,0.975))

    axes[-1].set_xlabel("Frame (Normalized %)")
    fig.suptitle(f"PCA Difference From Mean ($\omega$) Trajectories over Normalized Pitch Frames", 
                ha="center", weight='bold', y=1.0)
    fig.text(0.5, 0.975, r"$\omega_i = PC_i - \overline{PC_i}$", ha='center', fontsize=10)
    plt.tight_layout()
    plt.savefig("./plots/diff_plot.png")
    plt.show()

def plot_pc_contribution_skeleton(marker_contrib, pc):
    # Marker list
    markers = ['C7', 'CLAV', 'LANK', 'LASI', 'LBHD', 'LELB', 'LFHD', 'LFIN',
        'LFRM', 'LHEE', 'LKNE', 'LMANK', 'LMELB', 'LMKNE', 'LPSI', 'LSHO',
        'LTHI', 'LTIB', 'LTOE', 'LUPA', 'LWRA', 'LWRB', 'RANK', 'RASI',
        'RBAK', 'RBHD', 'RELB', 'RFHD', 'RFIN', 'RFRM', 'RHEE', 'RKNE',
        'RMANK', 'RMELB', 'RMKNE', 'RPSI', 'RSHO', 'RTHI', 'RTIB', 'RTOE',
        'RUPA', 'RWRA', 'RWRB', 'STRN', 'T10']

    # Dictionary for marker to index
    num_markers = len(markers)
    marker_to_idx = {m:i for i,m in enumerate(markers)}


    # Creating fake points to act as skeleton
    points = np.zeros((3, num_markers))

    # Head
    points[:, marker_to_idx['LBHD']] = [0, 0, 1.8]
    points[:, marker_to_idx['LFHD']] = [0.1, 0, 1.8]
    points[:, marker_to_idx['RBHD']] = [-0.1, 0, 1.8]
    points[:, marker_to_idx['RFHD']] = [0, 0.01, 1.8]
    points[:, marker_to_idx['C7']]   = [0, 0, 1.6]
    points[:, marker_to_idx['STRN']] = [0, 0, 1.4]
    points[:, marker_to_idx['CLAV']] = [0, 0, 1.5]

    # Shoulders
    points[:, marker_to_idx['LSHO']] = [0.3, 0, 1.5]
    points[:, marker_to_idx['RSHO']] = [-0.3, 0, 1.5]

    # Arms
    points[:, marker_to_idx['LUPA']] = [0.5, 0, 1.4]
    points[:, marker_to_idx['LELB']] = [0.7, 0, 1.2]
    points[:, marker_to_idx['LFRM']] = [0.9, 0, 1.0]
    points[:, marker_to_idx['LWRA']] = [1.0, 0, 0.8]
    points[:, marker_to_idx['LWRB']] = [1.05, 0, 0.7]
    points[:, marker_to_idx['LFIN']] = [1.1, 0, 0.6]

    points[:, marker_to_idx['RUPA']] = [-0.5, 0, 1.4]
    points[:, marker_to_idx['RELB']] = [-0.7, 0, 1.2]
    points[:, marker_to_idx['RFRM']] = [-0.9, 0, 1.0]
    points[:, marker_to_idx['RWRA']] = [-1.0, 0, 0.8]
    points[:, marker_to_idx['RWRB']] = [-1.05, 0, 0.7]
    points[:, marker_to_idx['RFIN']] = [-1.1, 0, 0.6]

    points[:, marker_to_idx['LMELB']] = [0.68, -0.05, 1.2]
    points[:, marker_to_idx['RMELB']] = [-0.68, -0.05, 1.2] 

    # Torso / hips
    points[:, marker_to_idx['LASI']] = [0.2, 0, 1.0]
    points[:, marker_to_idx['RASI']] = [-0.2, 0, 1.0]
    points[:, marker_to_idx['LPSI']] = [0.2, -0.01, 0.95]
    points[:, marker_to_idx['RPSI']] = [-0.2, -0.01, 0.95]
    points[:, marker_to_idx['T10']] = [0, 0, 1.3]
    points[:, marker_to_idx['RBAK']] = [-0.1, -0.05, 1.35]

    # Legs
    points[:, marker_to_idx['LTHI']] = [0.2, 0, 0.6]
    points[:, marker_to_idx['LKNE']] = [0.2, 0, 0.3]
    points[:, marker_to_idx['LTIB']] = [0.2, 0, 0.1]
    points[:, marker_to_idx['LANK']] = [0.2, 0, 0]
    points[:, marker_to_idx['LHEE']] = [0.15, 0, -0.05]
    points[:, marker_to_idx['LTOE']] = [0.25, 0, -0.05]
    points[:, marker_to_idx['LMANK']] = [0.2, -0.01, 0]

    points[:, marker_to_idx['RTHI']] = [-0.2, 0, 0.6]
    points[:, marker_to_idx['RKNE']] = [-0.2, 0, 0.3]
    points[:, marker_to_idx['RTIB']] = [-0.2, 0, 0.1]
    points[:, marker_to_idx['RANK']] = [-0.2, 0, 0]
    points[:, marker_to_idx['RHEE']] = [-0.15, 0, -0.05]
    points[:, marker_to_idx['RTOE']] = [-0.25, 0, -0.05]
    points[:, marker_to_idx['RMANK']] = [-0.2, -0.01, 0]

    points[:, marker_to_idx['LMKNE']] = [0.18, -0.05, 0.3]
    points[:, marker_to_idx['RMKNE']] = [-0.18, -0.05, 0.3]


    # Contribtuions for given PC
    pc_contrib = (
        marker_contrib[marker_contrib["PC"] == pc]
        .set_index('marker')['total_contrib']
        .reindex(markers)
        .fillna(0)              
    )

    # Normalize contributions for color mapping (0-1 range)
    contrib_normalized = (pc_contrib - pc_contrib.min()) / (pc_contrib.max() - pc_contrib.min() + 1e-8)

    # Define skeleton connections
    def get_idx(label):
        return marker_to_idx.get(label)

    connections_labels = [
        (get_idx('LBHD'), get_idx('LFHD')),
        (get_idx('RBHD'), get_idx('RFHD')),
        (get_idx('LFHD'), get_idx('RFHD')),
        (get_idx('LBHD'), get_idx('RBHD')),
        (get_idx('C7'), get_idx('STRN')),
        (get_idx('C7'), get_idx('CLAV')),
        (get_idx('STRN'), get_idx('CLAV')),
        (get_idx('CLAV'), get_idx('LSHO')),
        (get_idx('CLAV'), get_idx('RSHO')),
        (get_idx('LSHO'), get_idx('RSHO')),
        (get_idx('LSHO'), get_idx('LASI')),
        (get_idx('RSHO'), get_idx('RASI')),
        (get_idx('LASI'), get_idx('RASI')),
        (get_idx('LPSI'), get_idx('RPSI')),
        (get_idx('LASI'), get_idx('LPSI')),
        (get_idx('RASI'), get_idx('RPSI')),
        (get_idx('LSHO'), get_idx('LUPA')),
        (get_idx('LUPA'), get_idx('LELB')),
        (get_idx('LELB'), get_idx('LFRM')),
        (get_idx('LFRM'), get_idx('LWRA')),
        (get_idx('LWRA'), get_idx('LWRB')),
        (get_idx('LWRB'), get_idx('LFIN')),
        (get_idx('RSHO'), get_idx('RUPA')),
        (get_idx('RUPA'), get_idx('RELB')),
        (get_idx('RELB'), get_idx('RFRM')),
        (get_idx('RFRM'), get_idx('RWRA')),
        (get_idx('RWRA'), get_idx('RWRB')),
        (get_idx('RWRB'), get_idx('RFIN')),
        (get_idx('LASI'), get_idx('LTHI')),
        (get_idx('LTHI'), get_idx('LKNE')),
        (get_idx('LKNE'), get_idx('LTIB')),
        (get_idx('LTIB'), get_idx('LANK')),
        (get_idx('LANK'), get_idx('LHEE')),
        (get_idx('LANK'), get_idx('LTOE')),
        (get_idx('LHEE'), get_idx('LTOE')),
        (get_idx('LANK'), get_idx('LMANK')),
        (get_idx('RASI'), get_idx('RTHI')),
        (get_idx('RTHI'), get_idx('RKNE')),
        (get_idx('RKNE'), get_idx('RTIB')),
        (get_idx('RTIB'), get_idx('RANK')),
        (get_idx('RANK'), get_idx('RHEE')),
        (get_idx('RANK'), get_idx('RTOE')),
        (get_idx('RHEE'), get_idx('RTOE')),
        (get_idx('RANK'), get_idx('RMANK'))
    ]

    # Plot
    fig = plt.figure(figsize=(12, 10), facecolor='white')
    ax = fig.add_subplot(111, facecolor='#f8f9fa')

    # Plot skeleton lines first (so they appear behind markers)
    for p1, p2 in connections_labels:
        ax.plot(
            [points[0, p1], points[0, p2]],
            [points[2, p1], points[2, p2]],
            color='#cccccc',
            lw=2,
            alpha=0.9,
            zorder=1
        )

    # Plot markers with PC coloring
    scatter = ax.scatter(
        points[0],
        points[2],
        c=contrib_normalized,
        s=contrib_normalized*150 + 30,  # Size scales with contribution
        cmap='viridis',
        alpha=0.85,
        edgecolors='white',
        linewidth=1.5,
        zorder=3
    )

    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax, label='Normalized Contribution')

    # Optionally add marker labels
    show_labels = False
    if show_labels:
        for m in markers:
            idx = marker_to_idx[m]
            x = points[0, idx]
            z = points[2, idx]
            ax.text(x, z+0.08, m, fontsize=7, ha='center', alpha=0.6)
    
    ax.set_xlabel('X (m)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Z (m)', fontsize=11, fontweight='bold')
    ax.set_title(f'Skeleton Marker Contributions: {pc}', fontsize=13, fontweight='bold', pad=20)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.2, linestyle='--', linewidth=0.5)
    ax.margins(0.1)
    
    plt.tight_layout()
    plt.savefig(f"./plots/{pc}_contribution_skeleton.png")
    plt.show()
