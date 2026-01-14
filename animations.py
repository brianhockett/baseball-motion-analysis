from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
import pandas as pd
import numpy as np


def animate(sel, animation_type = 'real', name = None):

    # Ensure chronological frame order
    sel = sel.sort_values('frame')

    # For animation of real data: extract metadata and recenter on y-axis
    if animation_type == 'real':
        # Get first frame (0)
        first_frame = sel['frame'].min()

        # Compute first-frame mean 
        first_frame_mean_y = sel[sel['frame'] == first_frame]['y'].mean()

        # Subtract first-frame mean from all frames (centers all pitches to same starting point mean(y) = 0)
        sel['y'] = sel['y'] - first_frame_mean_y

        # Get metadata out of sel
        userID = sel['userID'].iloc[0]
        sessionID = sel['sessionID'].iloc[0]
        pitchNum = sel['pitchNum'].iloc[0]
        height = sel['height'].iloc[0]
        weight = sel['weight'].iloc[0]
        pitchType = sel['pitchType'].iloc[0]
        pitchSpeed = str(int(sel['pitchSpeed'].iloc[0]*10))
        hand = sel["p_throws"].iloc[0]

    # Build marker label-to-index mapping directly from data
    unique_markers = sorted(sel['markerID'].unique())
    unique_frames = sorted(sel['frame'].unique())
    num_markers = len(unique_markers)
    num_frames = len(unique_frames)

    marker_to_idx = {marker: i for i, marker in enumerate(unique_markers)}
    frame_to_idx = {frame: i for i, frame in enumerate(unique_frames)}
    
    points = np.full((3, num_markers, num_frames), np.nan)
    
    # Get x, y, z location of each point
    for _, row in sel.iterrows():
        m_idx = marker_to_idx[row['markerID']]
        f_idx = frame_to_idx[row['frame']]
        points[0, m_idx, f_idx] = row['x']
        points[1, m_idx, f_idx] = row['y']
        points[2, m_idx, f_idx] = row['z']

    def get_idx(label):
        return marker_to_idx.get(label)

    # Define connections
    skeleton_connections_labels = [
        # Head and Neck
        (get_idx('LBHD'), get_idx('LFHD')), # Left Head
        (get_idx('RBHD'), get_idx('RFHD')), # Right Head
        (get_idx('LFHD'), get_idx('RFHD')), # Forehead
        (get_idx('LBHD'), get_idx('RBHD')), # Back of Head
        (get_idx('C7'), get_idx('STRN')),  # Neck to Sternum (approx. upper torso center)
        (get_idx('C7'), get_idx('CLAV')),  # C7 to Clavicle (upper spine to shoulder girdle)
        
        # Torso
        (get_idx('STRN'), get_idx('CLAV')), # Sternum to Clavicle
        (get_idx('CLAV'), get_idx('LSHO')), # Clavicle to Left Shoulder
        (get_idx('CLAV'), get_idx('RSHO')), # Clavicle to Right Shoulder
        (get_idx('LSHO'), get_idx('RSHO')), # Across shoulders
        (get_idx('LSHO'), get_idx('LASI')), # Left Shoulder to Left ASIS (approx. body side)
        (get_idx('RSHO'), get_idx('RASI')), # Right Shoulder to Right ASIS
        (get_idx('LASI'), get_idx('RASI')), # Across hips
        (get_idx('LPSI'), get_idx('RPSI')), # Across posterior hips (if available, good for pelvis)
        (get_idx('LASI'), get_idx('LPSI')), # Left ASIS to Left PSIS
        (get_idx('RASI'), get_idx('RPSI')), # Right ASIS to Right PSIS
        
        # Left Arm
        (get_idx('LSHO'), get_idx('LUPA')), # Shoulder to Upper Arm (Bicep)
        (get_idx('LUPA'), get_idx('LELB')), # Upper Arm to Elbow
        (get_idx('LELB'), get_idx('LFRM')), # Elbow to Forearm (Radius/Ulna)
        (get_idx('LFRM'), get_idx('LWRA')), # Forearm to Wrist A
        (get_idx('LWRA'), get_idx('LWRB')), # Wrist A to Wrist B
        (get_idx('LWRB'), get_idx('LFIN')), # Wrist B to Finger
        
        # Right Arm
        (get_idx('RSHO'), get_idx('RUPA')), # Shoulder to Upper Arm (Bicep)
        (get_idx('RUPA'), get_idx('RELB')), # Upper Arm to Elbow
        (get_idx('RELB'), get_idx('RFRM')), # Elbow to Forearm (Radius/Ulna)
        (get_idx('RFRM'), get_idx('RWRA')), # Forearm to Wrist A
        (get_idx('RWRA'), get_idx('RWRB')), # Wrist A to Wrist B
        (get_idx('RWRB'), get_idx('RFIN')), # Wrist B to Finger

        # Left Leg
        (get_idx('LASI'), get_idx('LTHI')), # ASIS to Thigh (femur)
        (get_idx('LTHI'), get_idx('LKNE')), # Thigh to Knee
        (get_idx('LKNE'), get_idx('LTIB')), # Knee to Tibia
        (get_idx('LTIB'), get_idx('LANK')), # Tibia to Ankle
        (get_idx('LANK'), get_idx('LHEE')), # Ankle to Heel
        (get_idx('LANK'), get_idx('LTOE')), # Ankle to Toe
        (get_idx('LHEE'), get_idx('LTOE')), # Heel to Toe
        (get_idx('LANK'), get_idx('LMANK')), # Left Ankle to Medial Ankle (if LMANK is medial)
        
        # Right Leg
        (get_idx('RASI'), get_idx('RTHI')), # ASIS to Thigh (femur)
        (get_idx('RTHI'), get_idx('RKNE')), # Thigh to Knee
        (get_idx('RKNE'), get_idx('RTIB')), # Knee to Tibia
        (get_idx('RTIB'), get_idx('RANK')), # Tibia to Ankle
        (get_idx('RANK'), get_idx('RHEE')), # Ankle to Heel
        (get_idx('RANK'), get_idx('RTOE')), # Ankle to Toe
        (get_idx('RHEE'), get_idx('RTOE')), # Heel to Toe
        (get_idx('RANK'), get_idx('RMANK')) # Right Ankle to Medial Ankle (if RMANK is medial)
    ]

    # Filter out any connections where one or both of the markers were not found
    skeleton_connections = [conn for conn in skeleton_connections_labels if all(c is not None for c in conn)]

    # Set up plot
    fig = plt.figure(figsize = (10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Set consistent axis limits and viewing angle
    x_min, x_max = np.nanmin(points[0, :, :]), np.nanmax(points[0, :, :])
    y_min, y_max = np.nanmin(points[1, :, :]), np.nanmax(points[1, :, :])
    z_min, z_max = np.nanmin(points[2, :, :]), np.nanmax(points[2, :, :])
        
    # Add a small buffer to the limits
    buffer = 0.1 * (max(x_max-x_min, y_max-y_min, z_max-z_min) or 1) 
    ax.set_xlim([x_min - buffer, x_max + buffer])
    ax.set_ylim([y_min - buffer, y_max + buffer])
    ax.set_zlim([z_min - buffer, z_max + buffer])

    ax.set_xlabel('X Axis'); ax.set_ylabel('Y Axis'); ax.set_zlabel('Z Axis')
    ax.view_init(elev = 30, azim = -55)


    # Initialize scatter plot for the markers (frame = 0)
    scatter = ax.scatter(points[0, :, 0], points[1, :, 0], points[2, :, 0], alpha = 0.75, color = 'royalblue', label = 'Markers')
    title = ax.set_title('Frame 0')

    # Initialize line plots for the skeleton (frmae = 0)
    lines = []
    for p1_idx, p2_idx in skeleton_connections:
        x = points[0, [p1_idx, p2_idx], 0]
        y = points[1, [p1_idx, p2_idx], 0]
        z = points[2, [p1_idx, p2_idx], 0]
        line, = ax.plot(x, y, z, color='lightcoral', lw=2) 
        lines.append(line)

    # Update function for animation
    def update(frame):

        # Update marker positions
        scatter._offsets3d = (points[0, :, frame], points[1, :, frame], points[2, :, frame])
        if animation_type == 'real':
            title.set_text(f'Handedness: {hand} | Frame: {frame} | Percentage of Motion: {(frame/num_frames) * 100:.1f}%')
        else:
            title.set_text(f'| Frame: {frame} | Percentage of Motion: {(frame/num_frames) * 100:.1f}%')

        # Update skeleton line positions
        for i, (p1_idx, p2_idx) in enumerate(skeleton_connections):
            x = points[0, [p1_idx, p2_idx], frame]
            y = points[1, [p1_idx, p2_idx], frame]
            z = points[2, [p1_idx, p2_idx], frame]
            
            lines[i].set_data(x, y)
            lines[i].set_3d_properties(z)
        
        # Print out progress every 50 frames
        if frame % 100 == 0:
            print(f"On frame {frame} of animation")

        # Return updates
        return [scatter, title] + lines

    # Create and save animation
    num_frames = points.shape[2]
    ani = FuncAnimation(fig, update, frames=num_frames, interval=15, blit=False)

    print("Saving animation... please wait.")
    if animation_type == 'real':
        ani.save(f'./pca/animations/{animation_type}/{userID}_{sessionID}_{pitchNum}.mp4', writer ='ffmpeg', fps = 60)
        print(f"Animation saved as {userID}_{sessionID}_{pitchNum}.mp4")
    else:
        ani.save(f'./pca/animations/{animation_type}/{name}.mp4', writer ='ffmpeg', fps = 60)
        print(f"Animation saved as {name}.mp4")