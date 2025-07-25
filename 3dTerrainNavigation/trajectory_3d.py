import pandas as pd
import plotly.graph_objects as go
from terrain import Terrain # Make sure to import your Terrain environment

def visualize_3d_trajectory():
    """
    Creates an interactive 3D Plotly visualization of the agent's
    trajectory on the terrain surface.
    """
    # 1. Re-create the environment to get the terrain map, start, and goals
    #    (Use the same seed as training to ensure the map is identical)
    print("Recreating the environment...")
    env = Terrain(seed=69)
    terrain_data = env.terrain.cpu().numpy()
    goals = env.goals.cpu().numpy()
    start_pos = env.start.cpu().numpy()

    # 2. Load the saved trajectory data
    try:
        print("Loading trajectory data from best_trajectory.csv...")
        trajectory_df = pd.read_csv("./terrain_data/best_trajectory.csv")
    except FileNotFoundError:
        print("❌ Error: best_trajectory.csv not found. Please run the training script first to generate it.")
        return

    # 3. Get the elevation (z-coordinate) for each point in the trajectory
    trajectory_z = [terrain_data[int(row['x']), int(row['y'])] for index, row in trajectory_df.iterrows()]

    # 4. Create the 3D visualization
    print("Generating 3D plot...")
    fig = go.Figure()

    # Add the 3D terrain surface
    fig.add_trace(go.Surface(
        z=terrain_data,
        colorscale='earth',
        opacity=0.8,
        showscale=False,
        name='Terrain'
    ))

    # Add the agent's trajectory path over the surface
    fig.add_trace(go.Scatter3d(
        x=trajectory_df['y'], # Note: Plotly's axes might need swapping (x -> y)
        y=trajectory_df['x'], # depending on your coordinate system.
        z=trajectory_z,
        mode='lines+markers',
        line=dict(color='white', width=4),
        marker=dict(
            size=4,
            color=trajectory_df['reward'],
            colorscale='RdYlGn', # Red (low reward) to Green (high reward)
            showscale=True,
            colorbar=dict(title='Step Reward', x=1.15)
        ),
        name='Agent Path'
    ))

    # Add markers for Start and Goals
    fig.add_trace(go.Scatter3d(
        x=[start_pos[1]], y=[start_pos[0]], z=[terrain_data[start_pos[0], start_pos[1]]],
        mode='markers',
        marker=dict(color='blue', size=10, symbol='diamond'),
        name='Start'
    ))
    fig.add_trace(go.Scatter3d(
        x=goals[:, 1], y=goals[:, 0], z=[terrain_data[g[0], g[1]] for g in goals],
        mode='markers',
        marker=dict(color='red', size=8, symbol='x'),
        name='Goals'
    ))

    # 5. Update layout for a professional look
    fig.update_layout(
        title='Agent 3D Trajectory (Best Episode)',
        scene=dict(
            xaxis_title='Y Coordinate',
            yaxis_title='X Coordinate',
            zaxis_title='Elevation',
            aspectratio=dict(x=1, y=1, z=0.4) # Adjust z-axis scale for better viewing
        ),
        width=1000,
        height=800,
        margin=dict(l=0, r=0, b=0, t=40)
    )

    fig.show()
    fig.write_html("trajectory_visualization_3d.html")
    print("✅ Successfully saved interactive 3D trajectory plot to trajectory_visualization_3d.html")

if __name__ == '__main__':
    visualize_3d_trajectory()