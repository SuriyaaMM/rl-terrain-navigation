import pandas as pd
import plotly.graph_objects as go
from terrain import Terrain # Import your Terrain environment

def visualize_trajectory():
    """
    Creates a Plotly visualization of the agent's trajectory on the terrain.
    """
    # 1. Re-create the environment to get the terrain map and goal locations
    #    (use the same seed so it's the same map!)
    env = Terrain()
    terrain_data = env.terrain.cpu().numpy()
    goals = env.goals.cpu().numpy()
    start_pos = env.start.cpu().numpy()

    # 2. Load the saved trajectory data
    try:
        trajectory_df = pd.read_csv("./terrain_data/best_trajectory.csv")
    except FileNotFoundError:
        print("Error: best_trajectory.csv not found. Please run the training script first.")
        return

    # 3. Create the visualization
    fig = go.Figure()

    # Add the terrain as a heatmap background
    fig.add_trace(go.Heatmap(
        z=terrain_data,
        colorscale='earth',
        showscale=False,
        name='Terrain'
    ))

    # Add the agent's trajectory path
    # The path is colored by the reward at each step
    fig.add_trace(go.Scatter(
        x=trajectory_df['y'], # Note: Plotly's y-axis corresponds to array columns (our y)
        y=trajectory_df['x'], # Plotly's x-axis corresponds to array rows (our x)
        mode='lines+markers',
        line=dict(width=3, color='rgba(255, 255, 255, 0.7)'), # A white line connecting dots
        marker=dict(
            color=trajectory_df['reward'],
            colorscale='RdYlGn', # Red (bad) to Green (good)
            showscale=True,
            colorbar=dict(title='Step Reward'),
            size=6
        ),
        name='Agent Path'
    ))

    # Add Start and Goal markers
    fig.add_trace(go.Scatter(
        x=[start_pos[1]], y=[start_pos[0]],
        mode='markers',
        marker=dict(color='blue', size=12, symbol='star'),
        name='Start'
    ))
    fig.add_trace(go.Scatter(
        x=goals[:, 1], y=goals[:, 0],
        mode='markers',
        marker=dict(color='red', size=12, symbol='x'),
        name='Goals'
    ))

    # Update layout for a clean, professional look
    fig.update_layout(
        title='Agent Trajectory on Terrain (Best Episode)',
        xaxis_title='Y Coordinate',
        yaxis_title='X Coordinate',
        yaxis=dict(autorange="reversed"), # Match numpy array indexing
        template='plotly_dark',
        width=800,
        height=800
    )

    fig.show()
    fig.write_image("trajectory_visualization.svg")
    print("Saved interactive trajectory plot to trajectory_visualization.html")

if __name__ == '__main__':
    visualize_trajectory()