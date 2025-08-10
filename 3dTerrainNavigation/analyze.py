import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import re
import os

# WITH LOVE FROM GEMINI 2.5 PRO :)

# --- Configuration ---
CLIP_COEFF = 0.1
DATA_FOLDER = "terrain_data"
REPLY_FILE = os.path.join(DATA_FOLDER, "reply_df.csv")
PPO_FILE = os.path.join(DATA_FOLDER, "ppo_df.csv")
OUTPUT_FILE = "training_analysis_plotly.html"
IMAGE_OUTPUT_FILE = "training_analysis.svg"

def clean_and_visualize_plotly():
    """
    Loads training data from CSV files, cleans it, and generates
    interactive analysis plots using Plotly.
    """
    try:
        # --- 1. Load Data ---
        print(f"Loading data from {REPLY_FILE} and {PPO_FILE}...")
        reply_df = pd.read_csv(REPLY_FILE)
        ppo_df = pd.read_csv(PPO_FILE)

        # --- 2. Clean Data ---
        print("Cleaning and preparing data...")
        if 'policy_loss' in ppo_df.columns and ppo_df['policy_loss'].dtype == 'object':
            ppo_df['policy_loss'] = ppo_df['policy_loss'].apply(
                lambda x: float(re.search(r'tensor\((.*?)[,\)]', x).group(1)) if isinstance(x, str) else x
            )
        for col in ppo_df.columns:
            if col != 'Unnamed: 0':
                ppo_df[col] = pd.to_numeric(ppo_df[col], errors='coerce')
        for col in reply_df.columns:
            if col != 'Unnamed: 0':
                reply_df[col] = pd.to_numeric(reply_df[col], errors='coerce')
        ppo_df.dropna(inplace=True)
        reply_df.dropna(inplace=True)

        # --- 3. Generate Plots ---
        print("Generating Plotly figure...")
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=(
                'Agent Performance: Average Reward',
                "Critic's Prediction: Average State Value",
                'Exploration: Policy Entropy',
                'Training Stability: Total Loss',
                'Loss Components',
                'Policy Update Magnitude'
            )
        )

        # Plot 1: Average Reward
        fig.add_trace(go.Scatter(x=reply_df.index, y=reply_df['rewards'], mode='lines', name='Mean Reward', line=dict(color='green', width=1), opacity=0.8), row=1, col=1)
        if not reply_df['rewards'].empty:
            smoothed_rewards = reply_df['rewards'].rolling(window=5, min_periods=1).mean()
            fig.add_trace(go.Scatter(x=smoothed_rewards.index, y=smoothed_rewards, mode='lines', name='Smoothed Reward', line=dict(color='darkgreen', dash='dash')), row=1, col=1)

        # Plot 2: Average State Value
        fig.add_trace(go.Scatter(x=reply_df.index, y=reply_df['state_values'], mode='lines', name='State Value', line=dict(color='blue')), row=1, col=2)

        # Plot 3: Policy Entropy
        fig.add_trace(go.Scatter(x=reply_df.index, y=reply_df['entropies'], mode='lines', name='Entropy', line=dict(color='purple')), row=2, col=1)

        # Plot 4: Total Loss
        fig.add_trace(go.Scatter(x=ppo_df.index, y=ppo_df['loss_hist'], mode='lines', name='Total Loss', line=dict(color='red', width=1), opacity=0.8), row=2, col=2)
        if not ppo_df['loss_hist'].empty:
            smoothed_loss = ppo_df['loss_hist'].rolling(window=10, min_periods=1).mean()
            fig.add_trace(go.Scatter(x=smoothed_loss.index, y=smoothed_loss, mode='lines', name='Smoothed Loss', line=dict(color='darkred', dash='dash')), row=2, col=2)

        # Plot 5: Loss Components
        fig.add_trace(go.Scatter(x=ppo_df.index, y=ppo_df['policy_loss'], mode='lines', name='Policy Loss', line=dict(color='orange')), row=3, col=1)
        fig.add_trace(go.Scatter(x=ppo_df.index, y=ppo_df['value_loss'], mode='lines', name='Value Loss', line=dict(color='dodgerblue')), row=3, col=1)

        # Plot 6: Policy Update Ratio
        fig.add_trace(go.Scatter(x=ppo_df.index, y=ppo_df['ratios'], mode='lines', name='Mean Ratio', line=dict(color='cyan')), row=3, col=2)
        fig.add_hline(y=1.0 + CLIP_COEFF, line_dash="dash", line_color="black", annotation_text=f"Upper Clip ({1.0 + CLIP_COEFF})", row=3, col=2)
        fig.add_hline(y=1.0 - CLIP_COEFF, line_dash="dash", line_color="black", annotation_text=f"Lower Clip ({1.0 - CLIP_COEFF})", row=3, col=2)

        # --- 4. Update Layout and Save ---
        fig.update_layout(
            height=1200, width=1200,
            title_text="PPO Training Analysis",
            template='plotly_white',
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        fig.update_xaxes(title_text="Training Iteration", row=1, col=1)
        fig.update_xaxes(title_text="Training Iteration", row=1, col=2)
        fig.update_xaxes(title_text="Training Iteration", row=2, col=1)
        fig.update_xaxes(title_text="PPO Update Step", row=2, col=2)
        fig.update_xaxes(title_text="PPO Update Step", row=3, col=1)
        fig.update_xaxes(title_text="PPO Update Step", row=3, col=2)

        fig.write_html(OUTPUT_FILE)
        fig.write_image(IMAGE_OUTPUT_FILE, scale=1, format="svg", width=1200, height=1200)
        print(f"✅ Successfully generated interactive plot and saved to {OUTPUT_FILE}")
        # To open the plot directly in your browser, uncomment the following line:
        # fig.show()

    except FileNotFoundError:
        print(f"❌ Error: Could not find data files.")
        print(f"Please ensure '{REPLY_FILE}' and '{PPO_FILE}' exist.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == '__main__':
    clean_and_visualize_plotly()