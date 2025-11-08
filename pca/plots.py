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

